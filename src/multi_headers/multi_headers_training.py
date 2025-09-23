# MIT License
#
# Copyright (c) 2025 Songnian Qian
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# 🧩 Modified WikiText Training with Split FFN and Individual LM Headers
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2Config, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import os
from pathlib import Path
from tqdm import tqdm
import sys
import time
import signal
import argparse
import json
import math
from multi_headers_model import CustomMultiHeaderGPT2Model, TokenAwareCrossEntropyLoss

project_root = Path(__file__).parent.parent 
src_path = project_root  
if src_path.exists():
    sys.path.insert(0, str(src_path))
    print(f"📁 Added to Python path: {src_path}")
else:
    print(f"⚠️ src folder not found at: {src_path}")

from utils.dataset import WikiTextDataset

# Global interrupt flag
training_interrupted = False

def signal_handler(signum, frame):
    global training_interrupted
    print("\n⏸️ Training interruption requested...")
    print("🔄 Will save checkpoint and exit after current batch...")
    training_interrupted = True

signal.signal(signal.SIGINT, signal_handler)

def _add_bool_arg(parser, name, default=True, help_text=""):
    # adds --name / --no-name (Python 3.10+ supports BooleanOptionalAction,
    # but this helper works everywhere)
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})

def get_args():
    parser = argparse.ArgumentParser(description="WikiText Multi-Perceptron Training")
    
    # ---- Runtime / IO ----
    parser.add_argument("--pretrained_model", type=str, default="gpt2",
                        help="HF model name or path (e.g., gpt2, gpt2-medium)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Where to save checkpoints. If None, auto-selects a sensible path.")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark suite at end.")
    parser.add_argument("--benchmark_only", action="store_true", help="Only run benchmark (no training).")
    parser.add_argument("--benchmark_checkpoint", type=str, default=None,
                        help="Checkpoint path for benchmarking.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")

    # ---- Training core ----
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=0, help="0 = use epochs")
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.0, help="(kept for compatibility; backbone LR usually 0.0)")

    # ---- Mixture-of-Heads (enabled by default) ----
    _add_bool_arg(parser, "use_mixture", default=True, help_text="Enable differentiable mixture-of-heads.")
    parser.add_argument("--gate_hidden", type=int, default=256, help="Hidden size of gating MLP.")
    parser.add_argument("--freeze_head0_steps", type=int, default=5000, help="Warmup steps to keep head-0 frozen.")
    parser.add_argument("--mixture_sparse_after", type=int, default=20000,
                        help="Global step after which to use sparse (hard) gating.")
    parser.add_argument("--head_lr", type=float, default=5e-6, help="LR for trainable heads (not head-0).")
    parser.add_argument("--gate_lr", type=float, default=1e-5, help="LR for the gating MLP.")

    parser.add_argument("--moe_gate_entropy_coef", type=float, default=0.0,
        help="Warmup-only entropy floor for the gate; 0.001 is a good starting value.")
    parser.add_argument("--entropy_target_frac", type=float, default=0.5,
        help="Target fraction of max gate entropy during warmup (0..1).")
   
    # ---- (Optional) Token-aware / extras you already had ----
    _add_bool_arg(parser, "use_freq_lr", default=False, help_text="Enable frequency-based LR scaling.")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--min_factor", type=float, default=0.3)

    # Resume functionality
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save checkpoints")

    # Training parameters
    parser.add_argument("--max_samples", type=int, default=2000000, help="Maximum number of samples")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--window_size", type=int, default=1, help="Window size for targets")

    # Model parameters
    parser.add_argument("--n_lm_perceptrons", type=int, default=4, help="Number of LM head perceptrons")
    
    # Pretrained model options
    parser.add_argument("--use_pretrained", dest="use_pretrained", action="store_true",  default=True)
    parser.add_argument("--no-pretrained",  dest="use_pretrained", action="store_false")

    # make header balance after load pretrain
    parser.add_argument("--head_only_phase", action="store_true",
                        help="Train only the specified LM heads; freeze trunk and other heads.")
    parser.add_argument("--head_only_ids", type=str, default="1,2,3",
                    help="Comma-separated head indices to train when --head_only_phase is set.")

    # Saving options
    parser.add_argument("--log_every", type=int, default=100, help="Log every N steps")

    #parser.add_argument("--header_balance_steps", type=int, default=5000, help="Steps to force balance across LM headers")

     # Token-aware CE (per-token weights + optional anneal)
    parser.add_argument("--token_weights", type=str, default="token_weights.pt",
                        help="Path to per-token weight vector; set to '' to disable")
    parser.add_argument("--token_alpha", type=float, default=1.0, help="Legacy: constant CE weight (1.0=pure CE)")
    parser.add_argument("--token_alpha_start", type=float, default=1.0, help="Start CE weight (overrides token_alpha)")
    parser.add_argument("--token_alpha_end",   type=float, default=1.0, help="End CE weight")
    parser.add_argument("--token_anneal_steps", type=int, default=None, help="Steps to anneal start→end")

    # Auto training knobs
    parser.add_argument("--auto", action="store_true", default=True,
                        help="Auto-configure steps/warmup/eval cadence")
    parser.add_argument("--total_steps", type=int, default=30000,
                        help="Train for exactly this many optimizer steps if set")
    parser.add_argument("--scheduler", type=str, default="linear",
                        choices=["linear", "cosine"], help="LR schedule with warmup")
  
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="If --warmup_steps is None and --auto is set, use ratio*total_steps")
    
    # === Phase control ===
    parser.add_argument("--phase", choices=["auto", "A", "B", "C"], default="auto",
        help="Training phase. 'auto' uses step thresholds below.")
    parser.add_argument("--phase_a_end", type=int, default=30000,
        help="Global step at which Phase A ends.")
    parser.add_argument("--phase_b_end", type=int, default=80000,
        help="Global step at which Phase B ends. Phase C starts after this.")

    # === Gating schedule (taus) ===
    parser.add_argument("--mixture_tau_start", type=float, default=1.0,
        help="Tau at the start of Phase A.")
    parser.add_argument("--mixture_tau_mid", type=float, default=0.50,
        help="Tau to use during Phase B (or target when annealing A→B).")
    parser.add_argument("--mixture_tau_end", type=float, default=0.25,
        help="Tau target for Phase C.")

    parser.add_argument("--mixture_tau_sched", choices=["const", "warmup", "linear"],
        default="linear",
        help="How to schedule tau within each phase: const|warmup|linear.")

    # === Sparse routing (Phase C) ===
    parser.add_argument("--top_k", type=int, default=1,
        help="Top-k heads to mix per token when sparse=True. 1 = fastest.")
    # toggle pair so you can force it either way on CLI
    parser.add_argument("--compute_only_selected", dest="compute_only_selected",
        action="store_true", help="Compute only selected heads in sparse mode.")
    parser.add_argument("--no-compute_only_selected", dest="compute_only_selected",
        action="store_false", help="Disable fast path; compute all heads.")
    parser.set_defaults(compute_only_selected=True)

    # === Regularizers ===
    parser.add_argument("--moe_load_balance_coef", type=float, default=0.02,
        help="KL(usage || uniform) weight.")
    parser.add_argument("--moe_diversity_coef", type=float, default=0.0,
        help="JS-divergence across heads weight (requires all-head logits).")
    parser.add_argument("--moe_kl_to_hf_coef", type=float, default=0.0,
        help="Tiny tether to anchor head distribution (stability).")
    parser.add_argument("--eval_stride", type=int, default=512,
        help="Stride for HF-style PPL validator (contiguous text).")
    
    parser.add_argument("--tw-alpha-max",  type=float, default=1.0)
    parser.add_argument("--tw-ramp-steps", type=int,   default=20000)
    parser.add_argument("--tw-min",        type=float, default=0.3)
    parser.add_argument("--tw-max",        type=float, default=5.0)
    parser.add_argument("--tw-blend", type=float, default=1.0)   # 0.0=off, 1.0=fully weighted
    parser.add_argument("--tw-ramp-reset", action="store_true")  # start ramp from 0 at this run

    return parser.parse_args()

# Parse arguments
args = get_args()

# Environment setup
RUN_MODE = "colab" if "COLAB_GPU" in os.environ else "local"

if RUN_MODE == "colab":
    BASE_PATH = Path("/content/drive/MyDrive/Project1")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except ImportError:
        pass
else:
    BASE_PATH = Path("C:/Machine Learning/Project1")

BASE_PATH.mkdir(parents=True, exist_ok=True)

print(f"🔧 Running in {RUN_MODE} mode")
print(f"📁 Base path: {BASE_PATH}")

# Setup checkpoint directory
if args.checkpoint_dir:
    checkpoint_dir = Path(args.checkpoint_dir)
else:
    checkpoint_dir = BASE_PATH / "multi-headers_training_checkpoint"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
print(f"💾 Checkpoints will be saved to: {checkpoint_dir}")

def create_collate_fn(tokenizer):
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        targets = torch.stack([item['target'] for item in batch])
        
        # Create labels for language modeling
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'targets': targets
        }
    
    return collate_fn

class WikiTextTrainer:
    def __init__(self, args, model, tokenizer, device, checkpoint_dir=None):
        self.args = args
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.checkpoint_dir = checkpoint_dir or args.checkpoint_dir
        self.global_step = 0

        # ---- knobs (with safe defaults if not in args) ----
        self.tw_alpha_max  = float(getattr(args, "tw_alpha_max", 1.0))   # ramp power
        self.tw_ramp_steps = int(getattr(args, "tw_ramp_steps", 20000))  # steps to reach alpha_max
        self.tw_min        = float(getattr(args, "tw_min", 0.3))         # clamp lower
        self.tw_max        = float(getattr(args, "tw_max", 5.0))         # clamp upper
        self.tw_blend = float(getattr(args, "tw_blend", 1.0))
        self.tw_ramp_anchor = self.global_step if getattr(args, "tw_ramp_reset", False) else 0

        # ---- load token_weights.pt ----
        try:
            base = Path(globals().get("BASE_PATH", self.checkpoint_dir))
        except Exception:
            base = Path(self.checkpoint_dir)
        tw_path = Path(base) / "token_weights.pt"

        self.token_weight_vec = None  # [vocab] or None

        if tw_path.exists():
            raw = torch.load(tw_path, map_location="cpu")

            def _to_tensor(x): return torch.as_tensor(x, dtype=torch.float32)
            cfg = getattr(self, "config", getattr(self.model, "config", None))
            vocab = int(getattr(cfg, "vocab_size", 50257))

            if isinstance(raw, dict):
                vals = None
                for k in ["ranks","token_ranks","weights","freq","rank","token_weights"]:
                    if k in raw:
                        vals = _to_tensor(raw[k]); break
                if vals is None:
                    vals = torch.full((vocab,), float("nan"))
                    for k, v in raw.items():
                        try:
                            i = int(k)
                            if 0 <= i < vocab:
                                vals[i] = float(v)
                        except Exception:
                            pass
            else:
                vals = _to_tensor(raw)

            # resize/pad/trim
            if vals.numel() < vocab:
                vals = torch.cat([vals, torch.full((vocab - vals.numel(),), float("nan"))], dim=0)
            elif vals.numel() > vocab:
                vals = vals[:vocab]

            # --- orientation heuristic & safe fill (no torch.nanmin/nanmax) ---
            finite = vals[torch.isfinite(vals)]
            if finite.numel():
                mn = finite.min()
                mx = finite.max()
                looks_like_freq = (float(mn) >= 0.0 and float(mx) <= 1.5)  # small values => likely frequencies
            else:
                looks_like_freq = False

            # If values look like frequencies in [0..~1.5], invert so larger => rarer
            if looks_like_freq:
                vals = 1.0 / torch.clamp(vals, min=1e-12)

            # Recompute finite after possible inversion
            finite = vals[torch.isfinite(vals)]
            fill = finite.median() if finite.numel() else torch.tensor(1.0)

            # Replace NaN/Inf with median of finite values (compat: no torch.nan_to_num required)
            if hasattr(torch, "nan_to_num"):
                vals = torch.nan_to_num(vals, nan=float(fill), posinf=float(fill), neginf=float(fill))
            else:
                vals = torch.where(torch.isfinite(vals), vals, torch.full_like(vals, float(fill)))

            # Normalize mean≈1 and clamp for stability
            vals = vals / vals.mean().clamp(min=1e-12)
            vals = torch.clamp(vals, self.tw_min, self.tw_max)
            self.token_weight_vec = vals.to(self.device)
            self.token_weights    = self.token_weight_vec 
            print(f"✅ Loaded token weights: {tw_path} (mean≈1, clamp[{self.tw_min},{self.tw_max}])")
        else:
            print("⚠️ token_weights.pt not found — per-token weighting disabled.")
            self.token_weight_vec = None
            self.token_weights    = None              

        self.model.train()

    def _gold_weights(self, gold_token_ids: torch.Tensor) -> torch.Tensor:
        if self.token_weight_vec is None:
            return torch.ones_like(gold_token_ids, dtype=torch.float32, device=gold_token_ids.device)

        # ramp from 0 -> tw_alpha_max over tw_ramp_steps, starting at anchor
        step_since_anchor = max(0, int(getattr(self, "global_step", 0)) - int(getattr(self, "tw_ramp_anchor", 0)))
        alpha = float(self.tw_alpha_max) * min(1.0, step_since_anchor / max(1, int(self.tw_ramp_steps)))

        w = self.token_weight_vec[gold_token_ids]  # [N], float32 on correct device

        # soften/sharpen by alpha (alpha in [0,1] keeps extremes bounded)
        if alpha != 1.0:
            w = torch.clamp(w, self.tw_min, self.tw_max) ** alpha

        # (optional but recommended) keep batch mean ~1 to avoid LR drift
        w_mean = w.mean().clamp(min=1e-12)
        w = w / w_mean

        # final safety clamp
        return torch.clamp(w, self.tw_min, self.tw_max)

    def weighted_ce_from_logits(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        logits: [B, S, V], labels: [B, S] with -100 ignored
        Returns: (loss, stats_dict)
        """
        # next-token shift
        shift_logits = logits[:, :-1, :]            # [B, S-1, V]
        shift_labels = labels[:, 1:].contiguous()   # [B, S-1]
        valid = shift_labels.ne(-100)

        if not valid.any():
            return (shift_logits.new_tensor(0.0), {"tokens": 0})

        step_logits = shift_logits[valid]           # [N, V]
        gold_ids    = shift_labels[valid]           # [N]

        logprobs  = F.log_softmax(step_logits, dim=-1)
        gold_logp = logprobs.gather(1, gold_ids.unsqueeze(1)).squeeze(1)  # [N]
        nll_vec   = -gold_logp                                            # [N]

        tw = self._gold_weights(gold_ids)                                  # [N]
        loss = (nll_vec * tw).sum() / tw.sum().clamp(min=1e-12)

        return loss, {
            "tokens": int(gold_ids.numel()),
            "tw_mean": float(tw.mean().item()),
        }

    def token_aware_loss_from_logmix(
        self,
        log_mix: torch.Tensor,   # [B,S,V]  (LOG-probs from the mixture)
        labels:  torch.Tensor,   # [B,S]    (-100 masked)
        *,
        ignore_index: int = -100,
        w_min: float = 0.0,
        w_max: float | None = None,
        pow_gamma: float = 1.0,
    ) -> torch.Tensor:
        """
        Weighted NLL on mixture log-probs.
        Normalizes by the sum of weights over valid tokens so the scale is stable.
        """
        assert log_mix.dim() == 3, f"log_mix must be [B,S,V], got {tuple(log_mix.shape)}"
        B, S, V = log_mix.shape

        # Shift to next-token targets
        gold  = labels[:, 1:].contiguous()         # [B,S-1]
        logp  = log_mix[:, :-1, :].contiguous()    # [B,S-1,V]

        # Flatten
        gold_f = gold.view(-1)                     # [N]
        logp_f = logp.view(-1, V)                  # [N,V]
        valid  = (gold_f != ignore_index)

        # Base per-token NLL (unreduced) on valid positions
        nll = torch.zeros_like(gold_f, dtype=logp_f.dtype, device=logp_f.device)
        if valid.any():
            nll_valid = F.nll_loss(logp_f[valid], gold_f[valid], reduction="none")
            nll[valid] = nll_valid

        # Build weights for gold tokens
        if self.token_weights is None:
            w = torch.ones_like(gold_f, dtype=logp_f.dtype, device=logp_f.device)[valid]
        else:
            if isinstance(self.token_weights, torch.Tensor):
                lut = self.token_weights.to(device=logp_f.device, dtype=logp_f.dtype)
                # Expect shape [V]; if dict/tensor mismatch occurs, fallback to ones
                if lut.numel() != V:
                    w = torch.ones_like(gold_f, dtype=logp_f.dtype, device=logp_f.device)[valid]
                else:
                    w = lut[gold_f[valid]]
            else:
                # dict{id->weight} → LUT (cheap for occasional calls; fine per step too)
                lut = torch.ones(V, dtype=logp_f.dtype, device=logp_f.device)
                for k, v in self.token_weights.items():
                    idx = int(k)
                    if 0 <= idx < V:
                        lut[idx] = float(v)
                w = lut[gold_f[valid]]

        # Optional shaping/clamping
        if pow_gamma != 1.0:
            w = torch.pow(w, pow_gamma)
        if w_min is not None:
            w = torch.clamp(w, min=w_min)
        if w_max is not None:
            w = torch.clamp(w, max=w_max)

        # Weighted mean over valid tokens
        nll_valid = nll[valid]
        denom = w.sum().clamp_min(1e-8)
        loss = (nll_valid * w).sum() / denom
        return loss

    def build_optimizer_mixture(args, model):
        # heads 1..P-1 only
        head_params = []
        for i, head in enumerate(model.lm_head.perceptrons):
            if i == 0:
                continue  # keep head-0 as HF anchor
            head_params += [p for p in head.parameters() if p.requires_grad]

        gate_params = list(model.gate.parameters()) if hasattr(model, "gate") else []

        # backbone should be requires_grad=False during this phase
        opt = torch.optim.AdamW(
            [
                {"params": head_params, "lr": args.head_lr, "weight_decay": 0.01},
                {"params": gate_params, "lr": args.gate_lr, "weight_decay": 0.0},
                # (optional) include backbone with lr=args.lr if/when you unfreeze it:
                # {"params": model.transformer.parameters(), "lr": args.lr, "weight_decay": 0.01},
            ],
            betas=(0.9, 0.999),
        )
        # one-time debug
        for i,g in enumerate(opt.param_groups):
            n = sum(p.numel() for p in g["params"])
            print(f"🧪 Opt group {i}: lr={g['lr']:.2e}, wd={g.get('weight_decay',0)}, params={n:,}")
        return opt

    def build_optimizer_head_only(self, lr: float, train_head_ids: list[int]):
        params = []
        # only include parameters for the selected heads
        for i, head in enumerate(self.model.lm_head.perceptrons):
            if i in train_head_ids:
                params += list(head.parameters())
        # trunk is frozen by _freeze_for_head_only, so don't pass it here
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)

    def build_optimizer_full(self, lr: float):
        # train all heads; keep trunk effectively frozen by giving it lr=0.0
        return torch.optim.AdamW(
            [
                {"params": self.model.transformer.parameters(), "lr": 0.0, "weight_decay": 0.01},
                {"params": self.model.lm_head.parameters(),     "lr": lr,  "weight_decay": 0.0},
            ],
            lr=lr, betas=(0.9, 0.999)
        )

    def _init_hf_anchor_and_freeze_head0(self):
        hf = GPT2LMHeadModel.from_pretrained(self.args.pretrained_model)
        self.model.transformer.load_state_dict(hf.transformer.state_dict(), strict=True)
        with torch.no_grad():
            self.model.lm_head.perceptrons[0].weight.copy_(hf.lm_head.weight)
            if getattr(self.model.lm_head.perceptrons[0], "bias", None) is not None:
                self.model.lm_head.perceptrons[0].bias.zero_()

        # 2) freeze head-0 as anchor for the warmup phase
        for p in self.model.lm_head.perceptrons[0].parameters():
            p.requires_grad = False

        # (optional) quick sanity print
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        print(f"Params — trainable: {trainable:,} | frozen: {frozen:,}")

    def _build_optimizer(self):
        head_params = []
        gate_params = []
        other_params = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad: 
                continue
            if n.startswith("lm_head."):
                head_params.append(p)
            elif n.startswith("gate"):
                gate_params.append(p)
            else:
                other_params.append(p)

        groups = []
        if len(other_params):
            groups.append({"params": other_params, "lr": self.args.lr})
        if len(head_params):
            groups.append({"params": head_params, "lr": self.args.head_lr})
        if len(gate_params):
            groups.append({"params": gate_params, "lr": self.args.gate_lr})

        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=getattr(self.args,"weight_decay",0.0), fused=True)
        return opt

    def _override_group_lrs(self):
        # call this right after loading optimizer state_dict on resume
        i = 0
        for g in self.optimizer.param_groups:
            # heuristic: order = other, head, gate
            if i == 0: g["lr"] = self.args.lr
            elif i == 1: g["lr"] = self.args.head_lr
            elif i == 2: g["lr"] = self.args.gate_lr
            i += 1

    def maybe_unfreeze_head0_bias(self):
        # call this inside your train loop when global_step hits freeze_head0_steps
        if self.global_step == self.args.freeze_head0_steps:
            head0 = self.model.lm_head.perceptrons[0]
            if getattr(head0, "bias", None) is not None and not head0.bias.requires_grad:
                head0.bias.requires_grad = True
                # tiny LR for head-0 bias
                self.optimizer.add_param_group({"params": [head0.bias], "lr": 1e-6, "weight_decay": 0.0})
                print("🔓 Unfroze head-0 bias (bias-only) with lr=1e-6")

    def evaluate_perplexity_mixture(self, dataloader):
        self.model.eval()
        nll_sum, tok_count = 0.0, 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids      = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels         = batch['labels'].to(self.device)

                # Use the mixture exactly as in training; if you’re past
                # mixture_sparse_after, it'll use sparse gating with low tau
                k = min(1.0, float(self.global_step) / max(1, self.args.freeze_head0_steps))
                sparse = (self.global_step >= self.args.mixture_sparse_after)
                tau = (self.args.mixture_tau_start - (self.args.mixture_tau_start - self.args.mixture_tau_end) * k)

                _, log_mix, _, _ = self.forward_with_mixture(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    sparse=sparse,
                    tau=tau,
                    hf_anchor_head=0,
                    kl_to_hf_coef=0.0,   # no reg at eval
                    lb_coef=0.0,
                    div_coef=0.0
                )

                # sum NLL over valid next-token positions
                shift_log_mix = log_mix[:, :-1, :]
                shift_labels  = labels[:, 1:].contiguous()
                nll = F.nll_loss(
                    shift_log_mix.reshape(-1, shift_log_mix.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum"
                )
                nll_sum += nll.item()
                tok_count += (shift_labels != -100).sum().item()

        ppl = math.exp(nll_sum / max(1, tok_count))
        return ppl

    def forward_with_mixture(
        self,
        input_ids,
        attention_mask,
        labels,
        *,
        sparse: bool = False,
        tau: float = 1.0,
        top_k: int = 1,                     # NEW: choose 1 (fastest) or 2,3...
        compute_only_selected: bool = True, # NEW: only compute selected heads when sparse
        hf_anchor_head: int = 0,
        kl_to_hf_coef: float = 0.0,
        lb_coef: float = 0.0,
        div_coef: float = 0.0,
    ):
        """
        Returns:
        loss:      scalar
        log_mix:   [B,S,V]   (log-probs over vocab)
        pi:        [B,S,P]   (mixture weights; one-hot-ish if sparse+hard)
        all_logits:[B,S,V,P] or None  (None in fast sparse top-1 path)
        """
        # 1) Backbone once
        out = self.model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden = out.last_hidden_state                      # [B,S,E]
        B, S, E = hidden.shape
        P = self.model.lm_head.num_perceptrons
        V = self.model.lm_head.perceptrons[0].weight.size(0)

        # 2) Gate
        gate_logits = self.model.gate(hidden)               # [B,S,P]

        # --------- SOFT MIXTURE (Phase A/B) ---------
        if not sparse:
            # compute all heads once (no double forward)
            all_logits = self.model.lm_head(hidden, return_all_logits=True)  # [B,S,V,P]
            log_probs_h = F.log_softmax(all_logits, dim=2)                   # [B,S,V,P]
            log_pi      = F.log_softmax(gate_logits, dim=2)                  # [B,S,P]
            pi          = log_pi.exp()
            log_mix     = torch.logsumexp(log_pi.unsqueeze(2) + log_probs_h, dim=3)  # [B,S,V]

        # --------- SPARSE / HARD (Phase C) ---------
        else:
            # straight-through Gumbel; tau ~ 0.35→0.25
            pi_hard = F.gumbel_softmax(gate_logits, tau=tau, hard=True, dim=-1)  # [B,S,P]
            # keep a soft version for regularizers if you like (optional)
            # pi_soft = F.softmax(gate_logits / tau, dim=-1)
            pi = pi_hard  # use hard one-hot in forward; ST gives grads to gate

            if top_k == 1 and compute_only_selected:
                # FAST PATH: compute only heads that are actually selected anywhere
                sel = pi.argmax(dim=-1)                                       # [B,S]
                uniq = torch.unique(sel)
                selected_logits = torch.zeros(B, S, V, device=hidden.device, dtype=hidden.dtype)

                # Build mixture via ∑_h pi[...,h] * logits_h, but only over heads present
                for h in uniq.tolist():
                    logits_h = self.model.lm_head.perceptrons[h](hidden)      # [B,S,V]
                    weight   = pi[..., h].unsqueeze(-1)                       # [B,S,1] (one-hot)
                    selected_logits = selected_logits + weight * logits_h     # keeps ST grads to gate

                log_mix = F.log_softmax(selected_logits, dim=2)               # [B,S,V]
                all_logits = None                                             # we didn't compute all heads

            else:
                # GENERAL SPARSE: top-k mixture (k>=1) using all heads once
                all_logits = self.model.lm_head(hidden, return_all_logits=True)      # [B,S,V,P]
                log_probs_h = F.log_softmax(all_logits, dim=2)                       # [B,S,V,P]
                k = min(max(1, top_k), P)
                topk_idx = gate_logits.topk(k=k, dim=-1).indices                      # [B,S,k]
                log_pi_full = F.log_softmax(gate_logits, dim=2)                       # [B,S,P]
                # gather selected heads
                log_pi_sel   = torch.gather(log_pi_full, 2, topk_idx)                 # [B,S,k]
                gather_idx   = topk_idx.unsqueeze(2).expand(-1, -1, V, -1)            # [B,S,V,k]
                log_probs_sel= torch.gather(log_probs_h, 3, gather_idx)               # [B,S,V,k]
                log_mix      = torch.logsumexp(log_pi_sel.unsqueeze(2) + log_probs_sel, dim=3)
                pi = log_pi_full.exp()  # for regularizers/diagnostics

        # 3) Data loss on next-token targets (NLL on log-probs)
        gold = labels[:, 1:].contiguous()                        # [B,S-1]
        logp = log_mix[:, :-1, :].contiguous()                   # [B,S-1,V]
        nll  = F.nll_loss(
            logp.view(-1, V),
            gold.view(-1),
            ignore_index=-100,
            reduction="mean"
        )
        loss = nll

        # 4) Regularizers
        # 4a) Load-balance (gate usage ~ uniform)
        if lb_coef > 0.0:
            usage = pi[:, :-1, :].mean(dim=(0,1))                # [P]
            target = torch.full_like(usage, 1.0 / P)
            lb = F.kl_div((usage + 1e-9).log(), target, reduction="batchmean")
            loss = loss + lb_coef * lb

        # 4b) Diversity across heads (JS) — only if we have all heads
        if div_coef > 0.0 and all_logits is not None:
            log_probs_h = F.log_softmax(all_logits, dim=2)       # [B,S,V,P]
            probs_h = log_probs_h.exp()
            m = probs_h.mean(dim=3, keepdim=True)
            kl_h_m = (probs_h * (log_probs_h - m.clamp_min(1e-9).log())).sum(2).mean()
            kl_m_h = (m * (m.clamp_min(1e-9).log() - probs_h.clamp_min(1e-9).log())).sum(2).mean()
            js = 0.5 * (kl_h_m + kl_m_h)
            loss = loss + div_coef * js

        # 4c) Tiny tether to HF anchor head (cheap & works in fast path)
        if kl_to_hf_coef > 0.0:
            anchor_logits = self.model.lm_head.perceptrons[hf_anchor_head](hidden)  # [B,S,V]
            anchor_logp   = F.log_softmax(anchor_logits, dim=2)
            kl = F.kl_div(log_mix, anchor_logp, log_target=True, reduction="batchmean")
            loss = loss + kl_to_hf_coef * kl

        return loss, log_mix, pi, all_logits

    def _freeze_for_head_only(self, train_head_ids: list[int]):
        # freeze transformer trunk
        for p in self.model.transformer.parameters():
            p.requires_grad = False
        # freeze all heads except chosen ones
        for i, head in enumerate(self.model.lm_head.perceptrons):
            req = i in train_head_ids
            for p in head.parameters():
                p.requires_grad = req

    def _unfreeze_all(self):
        for p in self.model.transformer.parameters():
            p.requires_grad = True
        for head in self.model.lm_head.perceptrons:
            for p in head.parameters():
                p.requires_grad = True

    def build_optimizer(self, args):
        import torch

        if args.head_only_phase:
            train_ids = [int(x) for x in args.head_only_ids.split(",") if x.strip() != ""]
            lm_head_params = []
            for i, head in enumerate(self.model.lm_head.perceptrons):
                if i in train_ids:
                    lm_head_params += list(head.parameters())
            optimizer = torch.optim.AdamW(
                lm_head_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0
            )
        else:
            optimizer = torch.optim.AdamW([
                {"params": self.model.transformer.parameters(), "lr": args.lr * 0.0, "weight_decay": 0.01},
                {"params": self.model.lm_head.parameters(),     "lr": args.lr,        "weight_decay": 0.0},
            ], lr=args.lr, betas=(0.9, 0.999))
        return optimizer

    def select_best_perceptron_raw_logits(self, all_logits):
            """
            🆕 Raw logits head selection (same as benchmark script)
            all_logits: [B, S, V, P]
            Returns: [B, S] head indices
            """
            # Direct softmax on raw logits (no z-score normalization)
            probs = F.softmax(all_logits.float(), dim=2)           # [B, S, V, P]
            conf, _ = probs.max(dim=2)                             # [B, S, P] - max prob per head
            best_head = conf.argmax(dim=-1)                        # [B, S] - head with highest max prob
            return best_head

    def select_best_perceptron_max_prob(self, all_logits):
        """Select the best perceptron for each position based on MAX PROBABILITY (same as inference)"""
        # all_logits: [B, S, V, P]
        
        batch_size, seq_len, vocab_size, num_perceptrons = all_logits.shape
        
        # Use the same logic as inference: max probability per perceptron
        probs = F.softmax(all_logits, dim=2)  # [B, S, V, P]
        max_probs, _ = probs.max(dim=2)  # [B, S, P] - max prob for each perceptron
        best_perceptrons = max_probs.argmax(dim=-1)  # [B, S] - best perceptron index
    
        return best_perceptrons
    
    def select_best_perceptron_entropy(self, all_logits):
        """Select perceptron with lowest entropy (highest confidence)"""
        # all_logits: [B, S, V, P]
        probs = F.softmax(all_logits, dim=2).clamp_min(1e-9)  # [B,S,V,P]
        entropy = -(probs * probs.log()).sum(dim=2)           # [B,S,P]
        return entropy.argmin(dim=-1)                         # [B,S]
    
    def select_best_perceptron(self, all_logits, labels):
        """Select the best perceptron for each position based on prediction accuracy"""
        # all_logits: [B, S, V, P]
        # labels: [B, S]
        
        batch_size, seq_len, vocab_size, num_perceptrons = all_logits.shape
        
        # Get predictions for each perceptron
        predictions = all_logits.argmax(dim=2)  # [B, S, P]
        
        # Create mask for valid positions (not padding)
        valid_mask = (labels != -100)  # [B, S]
        
        # Initialize selection indices
        selected_indices = torch.zeros(batch_size, seq_len, dtype=torch.long, device=all_logits.device)
        
        for b in range(batch_size):
            for s in range(seq_len):
                if valid_mask[b, s]:
                    # Check which perceptrons got the correct prediction
                    correct_perceptrons = (predictions[b, s, :] == labels[b, s])
                    
                    if correct_perceptrons.any():
                        # If multiple perceptrons are correct, choose the one with highest confidence
                        correct_indices = torch.where(correct_perceptrons)[0]
                        if len(correct_indices) > 1:
                            # Use softmax to get confidence scores
                            probs = F.softmax(all_logits[b, s, :, correct_indices], dim=0)
                            max_probs, _ = probs.max(dim=0)
                            best_among_correct = correct_indices[max_probs.argmax()]
                            selected_indices[b, s] = best_among_correct
                        else:
                            selected_indices[b, s] = correct_indices[0]
                    else:
                        # If no perceptron is correct, choose the one with highest confidence
                        probs = F.softmax(all_logits[b, s, :, :], dim=0)
                        max_probs, _ = probs.max(dim=0)
                        selected_indices[b, s] = max_probs.argmax()
                else:
                    # For invalid positions, just use the first perceptron
                    selected_indices[b, s] = 0
        
        return selected_indices

    def select_best_perceptron_best_gold(self, shift_logits_all, shift_labels, ignore_index=-100, allowed_heads=None):
        """
        shift_logits_all: [B, T, V, P] logits for positions 1..S-1
        shift_labels:     [B, T]
        If allowed_heads is not None, restrict argmax to those heads.
        """
        import torch
        import torch.nn.functional as F

        B, T, V, P = shift_logits_all.shape
        logp = F.log_softmax(shift_logits_all.float(), dim=2)            # [B,T,V,P]
        valid = shift_labels.ne(ignore_index)
        gold  = shift_labels.masked_fill(~valid, 0)
        gold_idx = gold.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, P)
        gold_logp_all = logp.gather(2, gold_idx).squeeze(2)              # [B,T,P]

        if allowed_heads is not None:
            allow = torch.zeros(P, dtype=torch.bool, device=shift_logits_all.device)
            allow[allowed_heads] = True
            mask = (~allow).view(1, 1, P).expand(B, T, P)
            gold_logp_all = gold_logp_all.masked_fill(mask, float("-inf"))

        best = gold_logp_all.argmax(dim=-1)                              # [B,T]
        best = best.masked_fill(~valid, 0)
        return best

    def _phase_mixture_kwargs(self):
        """
        Decide sparse/tau/top_k/compute_only_selected for the current global_step,
        using only CLI args. Returns a dict for forward_with_mixture(**kwargs).
        """
        a_end = self.args.phase_a_end
        b_end = self.args.phase_b_end
        step  = int(self.global_step)

        # Resolve phase
        if self.args.phase == "A" or (self.args.phase == "auto" and step < a_end):
            phase = "A"
        elif self.args.phase == "B" or (self.args.phase == "auto" and step < b_end):
            phase = "B"
        else:
            phase = "C"

        # Tau schedule per phase
        def sched_tau(t0, t1, s_begin, s_end):
            mode = self.args.mixture_tau_sched
            if mode == "const" or s_end <= s_begin:
                return t1  # constant target
            if mode == "warmup":
                # warmup from t0->t1 only within [s_begin, s_begin+(s_end-s_begin)/5]
                warm_end = s_begin + max(1, (s_end - s_begin)//5)
                k = 0.0 if step <= s_begin else 1.0 if step >= warm_end else (step - s_begin) / max(1, warm_end - s_begin)
                return t0 * (1-k) + t1 * k
            # linear
            k = 0.0 if step <= s_begin else 1.0 if step >= s_end else (step - s_begin) / max(1, s_end - s_begin)
            return t0 * (1-k) + t1 * k

        if phase == "A":
            sparse = False
            tau    = sched_tau(self.args.mixture_tau_start, self.args.mixture_tau_mid, 0, a_end)
            top_k  = 1
            cos    = False  # irrelevant in soft
        elif phase == "B":
            sparse = False
            tau    = sched_tau(self.args.mixture_tau_mid, self.args.mixture_tau_mid, a_end, b_end)
            top_k  = 1
            cos    = False
        else:  # Phase C
            sparse = True
            tau    = sched_tau(self.args.mixture_tau_mid, self.args.mixture_tau_end, b_end, self.args.total_steps or (b_end + 1))
            top_k  = max(1, int(self.args.top_k))
            cos    = bool(self.args.compute_only_selected)

        return dict(
            sparse=sparse,
            tau=float(tau),
            top_k=top_k,
            compute_only_selected=cos,
            hf_anchor_head=0,
            kl_to_hf_coef=self.args.moe_kl_to_hf_coef,
            lb_coef=self.args.moe_load_balance_coef,
            div_coef=self.args.moe_diversity_coef,
        )

    def train_step(self, batch, optimizer, criterion=None, token_freqs=None,
               min_factor=0.3, alpha=0.5, allowed_heads=None):

        args = self.args  # or however you access args

        input_ids      = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels         = batch['labels'].to(self.device)

        optimizer.zero_grad()

        # ========== MIXTURE PATH ==========
        if args.use_mixture:
            # 1) Decide mixture behavior from CLI (phase/taus/sparse/top_k/fast-path)
            mix_kwargs = self._phase_mixture_kwargs()  # -> sparse, tau, top_k, compute_only_selected, hf_anchor_head, lb_coef, div_coef, kl_to_hf_coef

            # 2) Forward WITHOUT LB/JS regs; (anchor-KL: keep or zero depending on your phase)
            fw_kwargs = dict(mix_kwargs)
            fw_kwargs["lb_coef"]  = 0.0
            fw_kwargs["div_coef"] = 0.0
            # If you want anchor-KL OFF here, uncomment next line; else keep mix_kwargs value:
            # fw_kwargs["kl_to_hf_coef"] = 0.0

            _, log_mix, pi, all_logits = self.forward_with_mixture(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **fw_kwargs
            )

            # 3) Data loss (token-aware NLL on mixture log-probs)
            gold = labels[:, 1:].contiguous()                 # [B,S-1]
            logp = log_mix[:, :-1, :].contiguous()            # [B,S-1,V]
            V = logp.size(-1)

            if getattr(self, "token_weight_vec", None) is not None:
                data_loss = self.token_aware_loss_from_logmix(log_mix, labels, ignore_index=-100)
            else:
                data_loss = F.nll_loss(
                    logp.view(-1, V),
                    gold.view(-1),
                    ignore_index=-100,
                    reduction="mean"
                )
            loss = data_loss

            # 4) Regularizers (use the SAME coefs the phase helper decided)
            # 4a) Load-balance (works with fast path; uses pi only)
            if mix_kwargs.get("lb_coef", 0.0) > 0.0:
                usage  = pi[:, :-1, :].mean(dim=(0, 1))       # [P]
                target = torch.full_like(usage, 1.0 / usage.numel())
                lb = F.kl_div((usage + 1e-9).log(), target, reduction="batchmean")
                loss = loss + mix_kwargs["lb_coef"] * lb

            # 4b) Diversity (JS across heads) — only when we have all heads
            if mix_kwargs.get("div_coef", 0.0) > 0.0 and all_logits is not None:
                probs_h = F.softmax(all_logits, dim=2)        # [B,S,V,P]
                m = probs_h.mean(dim=3, keepdim=True)
                log_ph = (probs_h + 1e-9).log()
                log_m  = (m + 1e-9).log()
                kl_h_m = (probs_h * (log_ph - log_m)).sum(2).mean()
                kl_m_h = (m * (log_m - log_ph)).sum(2).mean()
                js = 0.5 * (kl_h_m + kl_m_h)
                loss = loss + mix_kwargs["div_coef"] * js

            # 5) Metrics (computed from the mixture)
            with torch.no_grad():
                valid = (gold != -100)                         # [B,S-1]
                preds = logp.argmax(dim=-1)                    # [B,S-1]
                denom = valid.sum().float().clamp_min(1.0)
                accuracy = ((preds == gold) & valid).sum().float() / denom

                top5_idx = logp.topk(5, dim=-1).indices       # [B,S-1,5]
                top5_accuracy = ((top5_idx == gold.unsqueeze(-1)) & valid.unsqueeze(-1)).any(dim=-1).sum().float() / denom

            # 6) Diagnostics & epoch-level usage accumulation (works in soft and hard)
            with torch.no_grad():
                pi_shift = pi[:, :-1, :]                      # [B,S-1,P]
                soft_usage_step = (pi_shift * valid.unsqueeze(-1)).sum(dim=(0, 1))  # [P]
                if not hasattr(self, "_usage_soft") or self._usage_soft is None:
                    self._usage_soft   = soft_usage_step.detach().clone()
                    self._usage_tokens = int(valid.sum().item())
                else:
                    self._usage_soft   += soft_usage_step.detach()
                    self._usage_tokens += int(valid.sum().item())

            if (self.global_step % getattr(self.args, "log_every", 2000)) == 0 and getattr(self, "token_weight_vec", None) is not None:
                with torch.no_grad():
                    # Recompute briefly just for stats (cheap)
                    logp  = log_mix[:, :-1, :]
                    gold  = labels[:, 1:]
                    valid = gold.ne(-100)
                    if valid.any():
                        gold_ids = gold[valid]
                        tw = self._gold_weights(gold_ids)
                        print(f"[tw] step={self.global_step} mean={tw.mean():.3f} "
                            f"p90={tw.quantile(0.9).item():.3f} p99={tw.quantile(0.99).item():.3f} N={tw.numel()}")

            if (self.global_step % getattr(args, "log_every", 500) == 0):
                with torch.no_grad():
                    denom_s = soft_usage_step.sum().clamp_min(1e-9)
                    soft_usage_pct = (soft_usage_step / denom_s).tolist()

                    hard_sel  = pi_shift.argmax(dim=-1)       # [B,S-1]
                    P_heads   = pi.size(-1)
                    hard_cnts = [int(((hard_sel == h) & valid).sum().item()) for h in range(P_heads)]
                    hard_total= max(1, sum(hard_cnts))
                    hard_pct  = [c / hard_total for c in hard_cnts]

                    log_pi   = pi_shift.clamp_min(1e-9).log()
                    H_gate   = (-(pi_shift * log_pi).sum(dim=2)).mean().item()
                    eff_heads= float(math.exp(H_gate))

                    # Per-head accuracy on hard assignments
                    per_head_acc = []
                    for h in range(P_heads):
                        m   = (hard_sel == h) & valid
                        tot = int(m.sum().item())
                        if tot == 0:
                            per_head_acc.append(0.0)
                        else:
                            ok = int(((preds == gold) & m).sum().item())
                            per_head_acc.append(ok / tot)

                    line = " | ".join(
                        f"h{h}: soft={soft_usage_pct[h]*100:.1f}%, hard={hard_pct[h]*100:.1f}%, acc={per_head_acc[h]*100:.2f}%"
                        for h in range(P_heads)
                    )
                    print(f"[step {self.global_step}] {line} | H={H_gate:.3f}, eff≈{eff_heads:.2f}")

            # 7) Backward + step (clip optional)
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=getattr(args, "max_grad_norm", 0.5))
            optimizer.step()
            # (If you drive a scheduler here, call scheduler.step() right after.)
            
            # Set this so your outer code can return/aggregate
            per_head_accuracy = per_head_acc if 'per_head_acc' in locals() else [0.0] * pi.size(-1)

        # --- Update LM head counters (works for soft & hard gating) ---
        with torch.no_grad():
            P = self.model.lm_head.num_perceptrons
            # ensure counters exist
            if not hasattr(self.model.lm_head, "usage_counts"):
                self.model.lm_head.usage_counts = [0 for _ in range(P)]
            if not hasattr(self.model.lm_head, "selection_counts"):
                self.model.lm_head.selection_counts = [0 for _ in range(P)]

            # gold/logp/valid come from the mixture path above
            #   gold: [B,S-1] = labels[:, 1:]
            #   logp: [B,S-1,V] = log_mix[:, :-1, :]
            #   valid: [B,S-1] = (gold != -100)
            pi_shift = pi[:, :-1, :]                 # [B,S-1,P]
            hard_sel = pi_shift.argmax(dim=-1)       # [B,S-1]

            for h in range(P):
                cnt = int(((hard_sel == h) & valid).sum().item())
                self.model.lm_head.usage_counts[h]     += cnt
                self.model.lm_head.selection_counts[h] += cnt

        # ------- Metrics from mixture -------
        with torch.no_grad():
            preds = logp.argmax(dim=-1)                                    # [B,S-1]
            denom = valid.sum().float().clamp_min(1.0)
            accuracy = ((preds == gold) & valid).sum().float() / denom

            top5_idx = logp.topk(5, dim=-1).indices                        # [B,S-1,5]
            top5_accuracy = ((top5_idx == gold.unsqueeze(-1)) &
                            valid.unsqueeze(-1)).any(dim=-1).sum().float() / denom

        # Per-head accuracy (diagnostic on hard assignments)
        per_head_accuracy = []
        with torch.no_grad():
            for h in range(P):
                m = (hard_sel == h) & valid
                tot = int(m.sum().item())
                if tot == 0:
                    per_head_accuracy.append(0.0)
                else:
                    ok = int(((preds == gold) & m).sum().item())
                    per_head_accuracy.append(ok / tot)

        return (
            float(loss.item()),
            float(accuracy.item()),
            float(top5_accuracy.item()),
            per_head_accuracy,
        )

    def validate(self, val_dataloader):
        self.model.eval()
        total_nll, total_tok, total_correct, total_top5 = 0.0, 0, 0, 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                # Use the same mixture forward as training, but NO regularizers
                mix_kwargs = self._phase_mixture_kwargs()
                fw_kwargs = dict(mix_kwargs)
                fw_kwargs["lb_coef"]  = 0.0
                fw_kwargs["div_coef"] = 0.0
                # keep anchor-KL as in mix_kwargs, or set to 0.0 if you prefer:
                # fw_kwargs["kl_to_hf_coef"] = 0.0

                _, log_mix, pi, _ = self.forward_with_mixture(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    **fw_kwargs
                )

                gold  = labels[:, 1:].contiguous()              # [B,S-1]
                logp  = log_mix[:, :-1, :].contiguous()         # [B,S-1,V]
                valid = (gold != -100)
                V = logp.size(-1)

                # Sum NLL over valid tokens
                nll = F.nll_loss(
                    logp.view(-1, V), gold.view(-1),
                    ignore_index=-100, reduction="sum"
                )
                preds = logp.argmax(dim=-1)                     # [B,S-1]
                top5  = logp.topk(5, dim=-1).indices           # [B,S-1,5]

                total_nll     += float(nll.item())
                ntok           = int(valid.sum().item())
                total_tok     += ntok
                total_correct += int(((preds == gold) & valid).sum().item())
                total_top5    += int((((top5 == gold.unsqueeze(-1)) & valid.unsqueeze(-1))
                                    .any(dim=-1)).sum().item())

        avg_nll = total_nll / max(1, total_tok)
        ppl     = math.exp(avg_nll)
        acc     = total_correct / max(1, total_tok)
        top5a   = total_top5 / max(1, total_tok)

        print(f"📊 Validation Results: NLL={avg_nll:.4f} | PPL={ppl:.2f} | Acc={acc:.2%} | Top5={top5a:.2%}")
        self.model.train()
        # Return two values to match your caller
        return ppl, acc

    def save_checkpoint(self, epoch, step, loss, optimizer, is_final=False, is_interrupted=False):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'global_step': step,
            'loss': loss,
            'timestamp': time.time(),
            'is_final': is_final,
            'is_interrupted': is_interrupted,
        }
        
        checkpoint_filename = f"checkpoint_step_{self.global_step}.pt"
        final_path = self.checkpoint_dir / checkpoint_filename
        
        print(f"\n💾 Saving checkpoint to: {final_path}")
        
        try:
            torch.save(checkpoint, final_path)
            print(f"✅ Checkpoint saved successfully at step {step}")
            return final_path
        except Exception as e:
            print(f"❌ Checkpoint save failed: {e}")
            return None

    def load_checkpoint(self, checkpoint_path, optimizer):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"📂 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Model state loaded successfully")
        except Exception as e:
            print(f"⚠️ Model state loading failed: {e}")
            return None
        
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Move optimizer states to correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            print("✅ Optimizer state loaded successfully")
            self._override_group_lrs()
        except Exception as e:
            print(f"⚠️ Optimizer state loading failed: {e}")
        
        self.global_step = checkpoint.get('global_step', 0)
        print(f"✅ Checkpoint loaded - resuming from step {self.global_step}")
        return checkpoint

    def format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m {seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"

    @torch.no_grad()
    def generate_sample(
        self,
        context_ids: torch.Tensor,
        max_len: int = 16,
        *,
        top_k: int = 1,                    # 1 = hard/fast; 2+ = soft mix of top-k heads
        tau: float = 0.3,                  # temperature for gate softmax when mixing
        compute_only_selected: bool = True,# True for fast path when top_k==1
        do_sample: bool = False,           # False = greedy; True = multinomial sampling
        temperature: float = 1.0,          # sampling/greedy temperature on token distribution
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Gated generation matching training-time routing.
        - top_k==1 & compute_only_selected=True -> hard routing (fast).
        - else -> soft mix of top-k heads via gate probabilities.
        Returns full sequence ids (prompt + generated).
        """
        import torch.nn.functional as F

        model = self.model
        dev   = self.device
        eos_id = eos_token_id if eos_token_id is not None else getattr(self.tokenizer, "eos_token_id", None)

        # [1, T] prompt
        input_ids = context_ids.unsqueeze(0).to(dev)
        attn_mask = torch.ones_like(input_ids, device=dev)

        # speed up decoding
        old_cache = getattr(model.transformer.config, "use_cache", True)
        model.transformer.config.use_cache = True
        model.eval()

        past = None
        for _ in range(max_len):
            out = model.transformer(
                input_ids=input_ids if past is None else input_ids[:, -1:],
                attention_mask=attn_mask,
                use_cache=True,
                past_key_values=past,
                return_dict=True,
            )
            past   = out.past_key_values
            hidden = out.last_hidden_state[:, -1:, :]                 # [B,1,E]
            gate_logits = model.gate(hidden)                          # [B,1,P]

            if top_k == 1 and compute_only_selected:
                # hard, fast routing: compute only the selected head
                h = gate_logits.argmax(dim=-1).squeeze(-1)            # [B]
                V = model.lm_head.perceptrons[0].weight.size(0)
                logits = hidden.new_zeros(hidden.size(0), 1, V)
                for b in range(hidden.size(0)):
                    logits[b:b+1] = model.lm_head.perceptrons[h[b].item()](hidden[b:b+1])
                token_scores = logits[:, -1, :]                       # [B,V] (logits)
            else:
                # soft mix over top-k heads
                all_logits = model.lm_head(hidden, return_all_logits=True)   # [B,1,V,P]
                log_probs_h = F.log_softmax(all_logits, dim=2)               # [B,1,V,P]
                log_pi = F.log_softmax(gate_logits / max(1e-6, tau), dim=2)  # [B,1,P]

                P = log_pi.size(-1)
                k = max(1, min(top_k, P))
                if k < P:
                    idx = gate_logits.topk(k=k, dim=-1).indices               # [B,1,k]
                    gather = idx.unsqueeze(2).expand(-1, -1, log_probs_h.size(2), -1)  # [B,1,V,k]
                    log_mix = torch.logsumexp(
                        torch.gather(log_probs_h, 3, gather) +
                        torch.gather(log_pi, 2, idx).unsqueeze(2),
                        dim=3
                    )                                                         # [B,1,V] (log-probs)
                    token_scores = log_mix[:, -1, :]                          # [B,V] (log-probs)
                else:
                    log_mix = torch.logsumexp(log_probs_h + log_pi.unsqueeze(2), dim=3)  # [B,1,V]
                    token_scores = log_mix[:, -1, :]                          # [B,V] (log-probs)

            # decode next token
            if do_sample:
                probs = F.softmax(token_scores / max(1e-6, temperature), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)          # [B,1]
            else:
                next_token = (token_scores / max(1e-6, temperature)).argmax(dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            attn_mask = torch.cat([attn_mask, torch.ones_like(next_token)], dim=1)

            if eos_id is not None and (next_token == eos_id).all():
                break

        model.transformer.config.use_cache = old_cache
        # return [T_total] tensor (same as your old API)
        return input_ids.squeeze(0).detach().cpu()

    @torch.no_grad()
    def generate_with_gate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        top_k: int = 1,
        tau: float = 0.3,
        compute_only_selected: bool = True,
        do_sample: bool = False,
        temperature: float = 1.0,
        eos_token_id: int | None = None,
    ):
        """
        Gate-routed generation that matches training-time routing.
        - top_k=1 & compute_only_selected=True => fast hard routing (Phase C style).
        - otherwise mixes top-k heads with gate softmax.
        """
        model = self.model
        tok   = self.tokenizer
        dev   = self.device
        eos_id = eos_token_id if eos_token_id is not None else tok.eos_token_id

        # tokenize
        enc = tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(dev)
        attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(dev)

        # enable cache for speed during generation
        old_cache = getattr(model.transformer.config, "use_cache", True)
        model.transformer.config.use_cache = True

        past_key_values = None
        model.eval()

        for _ in range(max_new_tokens):
            out = model.transformer(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                attention_mask=attn_mask,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
            )
            past_key_values = out.past_key_values
            hidden = out.last_hidden_state[:, -1:, :]           # [B,1,E]
            gate_logits = model.gate(hidden)                    # [B,1,P]

            if top_k == 1 and compute_only_selected:
                # hard route to one head (fast path)
                h = gate_logits.argmax(dim=-1).squeeze(-1)      # [B,1] -> [B]
                # compute only the selected head per batch element
                V = model.lm_head.perceptrons[0].weight.size(0)
                logits = hidden.new_zeros(hidden.size(0), 1, V)
                for b in range(hidden.size(0)):
                    logits[b:b+1] = model.lm_head.perceptrons[h[b].item()](hidden[b:b+1])
                mix_logits = logits[:, -1, :]                   # [B,V]
            else:
                # soft mix of top-k heads
                all_logits = model.lm_head(hidden, return_all_logits=True)     # [B,1,V,P]
                log_probs_h = F.log_softmax(all_logits, dim=2)                 # [B,1,V,P]
                log_pi = F.log_softmax(gate_logits / max(1e-6, tau), dim=2)    # [B,1,P]

                P = log_pi.size(-1)
                k = max(1, min(top_k, P))
                if k < P:
                    idx = gate_logits.topk(k=k, dim=-1).indices                # [B,1,k]
                    gather = idx.unsqueeze(2).expand(-1, -1, log_probs_h.size(2), -1)  # [B,1,V,k]
                    log_mix = torch.logsumexp(
                        torch.gather(log_probs_h, 3, gather) +
                        torch.gather(log_pi,    2, idx).unsqueeze(2), dim=3
                    )                                                          # [B,1,V]
                else:
                    log_mix = torch.logsumexp(log_probs_h + log_pi.unsqueeze(2), dim=3)  # [B,1,V]
                mix_logits = log_mix[:, -1, :]                                  # [B,V] (log-probs)

            # temperature + sampling/greedy
            if do_sample:
                probs = F.softmax(mix_logits / max(1e-6, temperature), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)            # [B,1]
            else:
                next_token = (mix_logits / max(1e-6, temperature)).argmax(dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            attn_mask = torch.cat([attn_mask, torch.ones_like(next_token)], dim=1)

            if eos_id is not None and (next_token == eos_id).all():
                break

        # restore original cache setting
        model.transformer.config.use_cache = old_cache
        return tok.decode(input_ids[0], skip_special_tokens=True)

    def train(self, train_dataloader, optimizer, scheduler, tokenizer, device, *,
            token_ranks=None, min_factor=0.2, alpha=0.8,
            num_epochs=1, val_dataloader=None, total_steps=None,
            eval_every=0, save_every=0, log_every=500, preview_every=500,
            head_only_phase=False, head_only_ids="1,2,3", head_only_epochs=0, lr=None):

        global training_interrupted

        # ---------------------------
        # Phase setup (head-only)
        # ---------------------------
        allowed_heads = None
        if head_only_phase or head_only_epochs > 0:
            allowed_heads = [int(x) for x in str(head_only_ids).split(",") if x.strip() != ""]
            # freeze trunk + non-selected heads
            self._freeze_for_head_only(allowed_heads)

            # Rebuild optimizer for head-only (do not keep frozen params in the optimizer)
            base_lr = (lr if lr is not None else (optimizer.param_groups[0]['lr'] if optimizer is not None else 2.5e-5))
            optimizer = self.build_optimizer_head_only(base_lr, allowed_heads)

            # IMPORTANT: (re)build or clear scheduler to match the new optimizer
            # If you have a factory, use it here (recommended):
            # scheduler = self.build_scheduler(optimizer, total_steps=total_steps, ...)
            # Otherwise, disable stepping to avoid optimizer mismatch:
            if (scheduler is not None) and getattr(scheduler, 'optimizer', None) is not optimizer:
                print("⚠️  Detaching old scheduler; please rebuild it for the new optimizer.")
                scheduler = None

        print(f"🚀 Starting training for {num_epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            if training_interrupted:
                print(f"\n⏸️ Training interrupted at epoch {epoch}")
                break

            # ----- Phase switch: exit head-only after N epochs -----
            if head_only_epochs > 0 and epoch == head_only_epochs + 1 and allowed_heads is not None:
                print("🔁 Switching from head-only phase to full training...")
                self._unfreeze_all()
                allowed_heads = None  # router can choose any head now
                base_lr = (lr if lr is not None else optimizer.param_groups[0]['lr'])
                optimizer = self.build_optimizer_full(base_lr)
                # Rebuild scheduler for the new optimizer if you have a factory:
                # scheduler = self.build_scheduler(optimizer, total_steps=total_steps, ...)
                if (scheduler is not None) and getattr(scheduler, 'optimizer', None) is not optimizer:
                    print("⚠️  Detaching old scheduler; please rebuild it for the new optimizer.")
                    scheduler = None

            print(f"\n🔄 Epoch {epoch}/{num_epochs}")

            total_loss = 0.0
            total_acc  = 0.0
            step_count = 0

            self.model.train()
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")

            for step, batch in enumerate(progress_bar):
                if 'done' in locals() and done:
                    break  # break out of the epoch loop

                if training_interrupted:
                    print(f"\n⏸️ Training interrupted at epoch {epoch}, step {step}")
                    current_loss = total_loss / max(step_count, 1)
                    self.save_checkpoint(epoch, self.global_step, current_loss, optimizer, is_interrupted=True)
                    return

                # -------- train step --------
                loss, accuracy, top5_acc, per_head_accuracy = self.train_step(
                    batch, optimizer,
                    token_freqs=token_ranks,
                    min_factor=min_factor,
                    alpha=alpha,
                    allowed_heads=allowed_heads,   # mask best-gold during head-only
                )

                total_loss += loss
                total_acc  += accuracy
                step_count += 1
                self.global_step += 1

                if getattr(self.args, "total_steps", None) is not None:
                    if self.global_step >= self.args.total_steps:
                        print(f"✅ Reached target steps: {self.global_step}/{self.args.total_steps}. Finishing training.")
                        done = True
                        # close progress bar cleanly if you use one
                        try:
                            progress_bar.close()
                        except Exception:
                            pass
                        break  # break out of the dataloader loop

                if scheduler is not None:
                    scheduler.step()

                # -------- lightweight preview generation --------
                if preview_every and (step % preview_every == 0) and step > 0:
                    try:
                        ctx_ids = batch["input_ids"][0][:16].detach().cpu()
                        prev_mode = self.model.training
                        self.model.eval()
                        with torch.no_grad():
                            sample_ids = self.generate_sample(ctx_ids, max_len=30,
                                                            top_k=1, compute_only_selected=True)  # fast hard routing
                            # or a softer mix of 2 heads:
                            # sample_ids = self.generate_sample(context_ids, max_len=60, top_k=2, compute_only_selected=False, tau=0.35)

                        if prev_mode: self.model.train()

                        prompt_text    = tokenizer.decode(ctx_ids,        skip_special_tokens=True)
                        generated_text = tokenizer.decode(sample_ids[16:], skip_special_tokens=True)
                        print(f"\n🎯 Prompt: '{prompt_text}' ->")
                        print(f"🎯 Generated: '{generated_text}'")
                    except Exception as e:
                        print(f"⚠️ Sample generation failed: {e}")

                # -------- progress bar info --------
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'acc': f'{accuracy:.2%}',
                    'avg_loss': f'{(total_loss/step_count):.4f}',
                    'top5_acc': top5_acc
                })

                # -------- detailed logging --------
                if log_every and (step % (log_every * 2) == 0) and step > 0:
                    if hasattr(self, "token_aware_loss") and (self.token_aware_loss is not None):
                        ce_changed = self.token_aware_loss.ce_changed
                        ce_delta   = self.token_aware_loss.ce_change_sum
                        avg_ce_delta   = (ce_delta / ce_changed) if ce_changed > 0 else 0.0
                        running_delta  = self.token_aware_loss.running_delta
                    else:
                        avg_ce_delta = 0.0
                        running_delta = 0.0
                        ce_changed    = 0

                    stats = self.model.get_lm_head_stats()
                    print(f"\n   Step {step:4d}|Loss: {loss:.4f}|Acc: {accuracy:.2%}")
                    print(f" Selection: {[f'{s:.1f}' for s in stats['selection_counts']]} | "
                        f"Accuracy: {[f'{a:.2%}' for a in per_head_accuracy]} | "
                        f"CE changed: {ce_changed}|delta: {avg_ce_delta:.5f} | "
                        f"z-avg: {running_delta:.5f} | Global: {self.global_step}")

                # -------- periodic checkpoint --------
                if save_every and (self.global_step % save_every == 0):
                    current_loss = total_loss / step_count
                    self.save_checkpoint(epoch, self.global_step, current_loss, optimizer)

            # --- Epoch-end selection summary (one line) ---
            if getattr(self, "_usage_soft", None) is not None and self._usage_soft.sum() > 0:
                pct = (self._usage_soft / self._usage_soft.sum().clamp_min(1e-9)).tolist()
                P   = len(pct)
                line = " | ".join(f"h{h}: {pct[h]*100:.1f}%" for h in range(P))
                print(f"   ├── Selection (soft %, epoch): {line}")
                # reset for next epoch
                self._usage_soft = None
                self._usage_tokens = 0

            # ----- epoch summary -----
            avg_loss = total_loss / max(step_count, 1)
            avg_acc  = total_acc  / max(step_count, 1)
            stats = self.model.get_lm_head_stats()
            elapsed_time = time.time() - start_time
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"   ├── Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.2%}")
            print(f"   ├── Usage Counts: {[f'{u:.1f}' for u in stats['usage_counts']]}")
            print(f"   ├── Selection Counts: {[f'{s:.1f}' for s in stats['selection_counts']]}")
            print(f"   ├── Selection %: {[f'{p:.1f}%' for p in stats['selection_percentages']]}")
            print(f"   └── Time: {self.format_time(elapsed_time)}")

            # ----- validation -----
            if val_dataloader is not None:
                val_loss, val_acc = self.validate(val_dataloader)
                print(f"   └── Validation | Loss: {val_loss:.4f} | Accuracy: {val_acc:.2%}")

            # checkpoint end of epoch
            if not training_interrupted:
                self.save_checkpoint(epoch, self.global_step, avg_loss, optimizer)

        # ----- final checkpoint -----
        if not training_interrupted:
            final_loss = total_loss / max(step_count, 1)
            self.save_checkpoint(num_epochs, self.global_step, final_loss, optimizer, is_final=True)

        total_time = time.time() - start_time
        print(f"\n🎉 Training completed!")
        print(f"📊 Total time: {self.format_time(total_time)}")
        print(f"📊 Total steps: {self.global_step}")

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Setup tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model if args.use_pretrained else "gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model configuration
    if args.use_pretrained:
        print(f"🔧 Using GPT2 configuration for pretrained model: {args.pretrained_model}")
        config = GPT2Config.from_pretrained(args.pretrained_model)
        config.n_positions = args.max_length
    else:
        config = GPT2Config(
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_positions=args.max_length,
            vocab_size=tokenizer.vocab_size,
        )
    
    # Add custom parameters
    config.n_lm_perceptrons = args.n_lm_perceptrons
    config.force_identical_output = False  # Changed to use individual selection

    # Create model
    model = CustomMultiHeaderGPT2Model(
        config, 
        use_pretrained=args.use_pretrained, 
        pretrained_model=args.pretrained_model
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created:")
    print(f"   ├── Total parameters: {total_params:,}")
    print(f"   ├── Trainable parameters: {trainable_params:,}")
    print(f"   ├── LM Head perceptrons: {args.n_lm_perceptrons}")
    print(f"   └── Perceptron input: FULL ({config.n_embd})")

    # Load datasets
    train_dataset = WikiTextDataset(
        data_dir=BASE_PATH / "wikitext-103",
        tokenizer=tokenizer,
        max_length=args.max_length,
        split="train",
        random_start=False,
        max_samples=args.max_samples,
        start_position=0,
        window_size=args.window_size
    )

    val_dataset = WikiTextDataset(
        data_dir=BASE_PATH / "wikitext-103",
        tokenizer=tokenizer,
        max_length=args.max_length,
        split="train",
        random_start=False,
        max_samples=50,
        start_position=0,
        window_size=args.window_size
    )

    # Create DataLoaders
    collate_fn = create_collate_fn(tokenizer)
    
    # Configure DataLoader settings based on environment
    if RUN_MODE == "colab":
        loader_config = {'num_workers': 2, 'pin_memory': True}
        print("🔧 Using Colab-optimized DataLoader configuration")
    else:
        loader_config = {'num_workers': 0, 'pin_memory': torch.cuda.is_available()}
        print("🔧 Using Windows-compatible DataLoader configuration")

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        **loader_config
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        **loader_config
    )

    print(f"📊 Training batches: {len(train_dataloader)}")
    print(f"📊 Validation batches: {len(val_dataloader)}")

    # =========================
    # Optimizer (Mixture aware)
    # =========================
    def _debug_opt(opt):
        total = 0
        for i, g in enumerate(opt.param_groups):
            n = sum(p.numel() for p in g["params"])
            total += n
            print(f"🧪 Opt group {i}: lr={g['lr']:.2e}, wd={g.get('weight_decay',0)}, params={n:,}")
        print(f"🧪 Total params in optimizer: {total:,}")

    if args.use_mixture:
        # 1) freeze backbone + head-0 (HF anchor) for head/gate-only phase
        for p in model.transformer.parameters():
            p.requires_grad = False
        if hasattr(model.lm_head, "perceptrons"):
            for p in model.lm_head.perceptrons[0].parameters():
                p.requires_grad = False

        # 2) collect params
        head_params = []
        if hasattr(model.lm_head, "perceptrons"):
            for i, head in enumerate(model.lm_head.perceptrons):
                if i == 0:
                    continue  # keep head-0 frozen
                head_params += [p for p in head.parameters() if p.requires_grad]
        else:
            # fallback: single head module
            head_params = [p for p in model.lm_head.parameters() if p.requires_grad]

        gate_params = list(model.gate.parameters()) if hasattr(model, "gate") else []

        # 3) build optimizer (AdamW)
        optimizer = torch.optim.AdamW(
            [
                {"params": head_params, "lr": args.head_lr, "weight_decay": 0.01},
                {"params": gate_params, "lr": args.gate_lr, "weight_decay": 0.0},
            ],
            betas=(0.9, 0.999),
        )
        print("🔧 Using head/gate-only training (mixture enabled):")
        _debug_opt(optimizer)

    else:
        # Hard-router path: train heads only, keep backbone frozen; freeze head-0 for stability
        for p in model.transformer.parameters():
            p.requires_grad = False
        if hasattr(model.lm_head, "perceptrons"):
            for p in model.lm_head.perceptrons[0].parameters():
                p.requires_grad = False

        head_params = []
        if hasattr(model.lm_head, "perceptrons"):
            for i, head in enumerate(model.lm_head.perceptrons):
                if i == 0:
                    continue
                head_params += [p for p in head.parameters() if p.requires_grad]
        else:
            head_params = [p for p in model.lm_head.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            [{"params": head_params, "lr": args.head_lr, "weight_decay": 0.01}],
            betas=(0.9, 0.999),
        )
        print("🔧 Using head-only training (hard router):")
        _debug_opt(optimizer)

    # --- total steps ---
    if args.total_steps is not None:
        total_steps = args.total_steps
    else:
        steps_per_epoch = len(train_dataloader)
        total_steps = steps_per_epoch * args.epochs
        if getattr(args, "auto", False):
            total_steps = min(total_steps, 60_000)

    # --- warmup ---
    warmup_steps_arg = getattr(args, "warmup_steps", None)
    if warmup_steps_arg is not None:
        warmup_steps = warmup_steps_arg
    else:
        warmup_ratio = getattr(args, "warmup_ratio", 0.1)
        use_auto = getattr(args, "auto", False)
        warmup_steps = max(1_000, int(warmup_ratio * total_steps)) if use_auto else 0

    print(f"📈 Training schedule: total_steps={total_steps}, warmup_steps={warmup_steps}")

    # --- scheduler ---
    if args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

    # Load token ranks if available (optional)
    try:
        with open(BASE_PATH / "token_counts.json", "r", encoding="utf-8") as f:
            token_data = json.load(f)

        vocab_size = model.config.vocab_size
        token_ranks = torch.ones(vocab_size)

        # Create a vocabulary string to rank mapping
        vocab_to_rank = {}
        for entry in token_data:
            vocab_str = entry["vocabulary"]
            rank = entry["rank"]
            vocab_to_rank[vocab_str] = rank

        print(f"🔍 DEBUG: Loaded {len(vocab_to_rank)} vocabulary entries")

        # Map current tokenizer's tokens to ranks using vocabulary strings
        loaded_count = 0
        for token_id in range(vocab_size):
            try:
                token_str = tokenizer.decode([token_id])
                if token_str in vocab_to_rank:
                    token_ranks[token_id] = vocab_to_rank[token_str]
                    loaded_count += 1
            except:
                continue

        print(f"✅ Loaded token ranks for {loaded_count} tokens")
        
    except FileNotFoundError:
        print("⚠️ token_counts.json not found; continuing without frequency-aware weights")

    # Create trainer
    trainer = WikiTextTrainer(args, model, tokenizer, device, checkpoint_dir)

    tace_loss = TokenAwareCrossEntropyLoss(
        token_weights=trainer.token_weights,            # None is fine too
        alpha_start=getattr(args, "token_alpha_start", args.token_alpha),
        alpha_end=getattr(args, "token_alpha_end", 1.0),   # 1.0 ⇒ pure CE by end (good for PPL)
        total_steps=total_steps,
        ignore_index=-100,
    )
   
    trainer.token_aware_loss = tace_loss

    # Load checkpoint if resuming   
    if args.resume:
        try:
            trainer.load_checkpoint(args.resume, optimizer)
            print(f"🔄 Resumed from: {args.resume}")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print("🆕 Starting fresh training")
    else:
        print("🆕 Starting fresh training")

    # Tokenizer verification
    print("\n🧪 Tokenizer Check:")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    print(f"   EOS token ID: {tokenizer.eos_token_id}")
    print(f"   PAD token ID: {tokenizer.pad_token_id}")

    # Start training
    try:
        trainer.train(
            train_dataloader, optimizer, scheduler, tokenizer, device,
            token_ranks=token_ranks if args.use_freq_lr else None,
            min_factor=args.min_factor, alpha=args.alpha,
            num_epochs=args.epochs, val_dataloader=val_dataloader,
            total_steps=total_steps,                      # ← name it
            eval_every=args.eval_every, save_every=args.save_every
        )
        
        # Save final model
        save_path = checkpoint_dir / "final_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'final_stats': model.get_lm_head_stats(),
        }, save_path)
        print(f"💾 Final model saved to {save_path}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n📁 All files saved to: {checkpoint_dir}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()