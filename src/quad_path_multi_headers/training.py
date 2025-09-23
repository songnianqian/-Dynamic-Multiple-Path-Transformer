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

# Quad Path GPT Training Script with LM Headers
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Tokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import os
from pathlib import Path
from tqdm import tqdm
import sys
import time
import signal
import argparse
import json
import math
project_root = Path(__file__).parent.parent 
src_path = project_root  
if src_path.exists():
    sys.path.insert(0, str(src_path))
    print(f"📁 Added to Python path: {src_path}")
else:
    print(f"⚠️ src folder not found at: {src_path}")

# Import the quad path model and dataset
from multi_path_model import HierarchicalMultiPathGPT2
from quad_path_model import QuadPathTrainer
from utils.dataset import WikiTextDataset  # Keep the existing dataset

from torch.serialization import add_safe_globals
try:
    # allowlist GPT2Config so weights_only=True can unpickle safely
    from transformers.models.gpt2.configuration_gpt2 import GPT2Config
    add_safe_globals([GPT2Config])
except Exception:
    pass

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

HEAD_ORDER = (0,1,2,3,4,5,6,7)

# Global interrupt flag
training_interrupted = False

def signal_handler(signum, frame):
    global training_interrupted
    print("\nTraining interruption requested...")
    print("Will save checkpoint and exit after current batch...")
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
    parser = argparse.ArgumentParser(description="Quad Path GPT Training with LM Headers")
    
    # Model and data
    parser.add_argument("--pretrained_model", type=str, default="gpt2",
                        help="HF model name or path (e.g., gpt2, gpt2-medium)")
    parser.add_argument("--split_at_layer_1", type=int, default=6,
                        help="First split layer (dual path split)")
    parser.add_argument("--split_at_layer_2", type=int, default=9,
                        help="Second split layer (quad path split)")
    parser.add_argument("--max_length", type=int, default=129,
                        help="Maximum sequence length")
    parser.add_argument("--max_samples", type=int, default=2000000,
                        help="Maximum number of training samples")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=0, help="0 = use epochs")
    
    # LM Header specific parameters (similar to multi_headers_model)
    parser.add_argument("--head_lr", type=float, default=5e-6, 
                        help="LR for trainable LM heads (heads 1,2,3)")
    parser.add_argument("--gate_lr", type=float, default=1e-5,
                        help="LR for gate networks")
    parser.add_argument("--freeze_head0_steps", type=int, default=5000,
                        help="Steps to keep head-0 frozen as anchor")
    
    # Logging and checkpointing
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Environment
    parser.add_argument("--seed", type=int, default=42)

    # Path selection strategies
    parser.add_argument(
        "--train_path_selection",
        type=str,
        default="hierarchical_gate",
        choices=["hierarchical_gate", "gate_soft", "gate_hard", "left_left_only", "max_prob"],
        help="Path routing for training."
    )

    parser.add_argument(
        "--eval_path_selection",
        type=str,
        default="hierarchical_gate",
        choices=["hierarchical_gate", "gate_hard", "left_left_only", "max_prob"],
        help="Path routing for evaluation."
    )

    # Loss coefficients
    parser.add_argument("--lb_coef", type=float, default=1e-3,
                        help="Load balance coefficient")
    parser.add_argument("--gold_aux_coef", type=float, default=1e-3,
                        help="Gold routing auxiliary loss coefficient")
    parser.add_argument("--tether_coef", type=float, default=5e-4,
                        help="Tether to baseline coefficient")
    parser.add_argument("--gate_temp", type=float, default=1.2,
                        help="Gate temperature")

    # Path freezing control
    parser.add_argument("--freeze_schedule", type=str, default=None,
                        help="JSON file with path freezing schedule")

    # hard-window: DISABLED by default (backward compatible)
    parser.add_argument("--hard_from_step", type=int, default=-1)
    parser.add_argument("--hard_to_step",   type=int, default=-1)
    parser.add_argument("--hard_from_frac", type=float, default=-1.0)
    parser.add_argument("--hard_to_frac",   type=float, default=-1.0)

    # gate temp schedule points: empty = DISABLED (use args.gate_temp like Phase A)
    parser.add_argument("--gate_temp_points", type=str, default="")

    # consistency loss: DISABLED by default for Phase A compatibility
    parser.add_argument("--consistency_lambda", type=float, default=0.0)

    # LM Header training phases (similar to multi_headers_model)
    parser.add_argument("--head_only_phase", action="store_true",
                        help="Train only LM headers, freeze everything else")
    parser.add_argument("--head_only_epochs", type=int, default=0,
                        help="Number of epochs to train only headers")
    
    # Token-aware training (optional)
    parser.add_argument("--use_token_aware", action="store_true",
                        help="Use token-aware weighting if token_weights.pt available")
    
    parser.add_argument(
        "--init_from_quad", type=str, default=None,
        help="Path to a QUAD checkpoint to load MODEL WEIGHTS ONLY (do not resume optimizer/scheduler)."
    )

    parser.add_argument(
        "--clip_grad_norm", type=float, default=1.0,
        help="Max global norm for gradient clipping; set 0 to disable."
    )

    parser.add_argument("--gate_hidden", type=int, default=256,
                    help="Hidden size for gating MLPs and per-path head gates")
    
    parser.add_argument("--n_lm_perceptrons", type=int, default=8)
    _add_bool_arg(parser, "use_head_mixture", default=False, help_text="Enable differentiable mixture-of-heads.")
    parser.add_argument("--head_allocation", type=str, default=None)

    parser.add_argument("--freeze_all_transformer", action="store_true",
                    help="Freeze all transformer blocks (layers 0..11 across all paths).")
    parser.add_argument("--freeze_split_gates", action="store_true",
                        help="Freeze gates at split-1 (layer6) and split-2 (layer9).")
    parser.add_argument("--head_topk", type=int, default=None,
                        help="Fast-k for LM-head gate (mix over top-k heads per path).")
    parser.add_argument("--head_gate_temp", type=float, default=1.0,
                        help="Temperature for LM-head gate softmax (soft gating).")

    # argparse additions
    parser.add_argument("--kd_coef", type=float, default=0.0,
        help=">0 enables KL distillation: teacher = kd_teacher_selection, student = train_path_selection.")
    parser.add_argument("--kd_teacher_selection", type=str, default="hierarchical_gate",
        help="Path selection used for the teacher pass (e.g., hierarchical_gate).")
    parser.add_argument("--kd_on", type=str, default="never", choices=["always","uncertain","never"],
        help="Where to apply KD: all valid tokens, only uncertain tokens, or never.")
    parser.add_argument("--kd_margin", type=float, default=0.10,
        help="Uncertainty margin for kd_on=uncertain (soft fallback threshold on leaf weights).")

    return parser.parse_args()

def create_collate_fn(tokenizer):
    """Create collate function for DataLoader"""
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        # Create labels for language modeling
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    return collate_fn

def analyze_path_usage(trainer, dataloader, num_batches=10):
    """Analyze which leaf path would be chosen per token (max per-token log-prob)."""
    trainer.model.eval()

    PATHS = ("left_left", "left_right", "right_left", "right_right")
    path_selections = {p: 0 for p in PATHS}
    total_tokens = 0

    with torch.inference_mode():
        for bi, batch in enumerate(dataloader):
            if bi >= num_batches:
                break

            input_ids = batch["input_ids"].to(trainer.device)
            attention_mask = batch["attention_mask"].to(trainer.device)

            # name -> [B,S,V] (could be logits OR log-probs)
            all_path_scores = trainer.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_all_paths=True
            )

            # Convert each to LOG-PROBS, then take max over vocab
            per_path_max_lp = []
            for name in PATHS:
                x = all_path_scores[name]  # [B,S,V]
                # Detect log-probs via logsumexp ≈ 0; otherwise convert
                lse = torch.logsumexp(x.float(), dim=-1)      # [B,S]
                is_logprob = lse.detach().median().abs() < 1e-3
                lp = x if is_logprob else F.log_softmax(x, dim=-1)
                max_lp, _ = lp.max(dim=-1)                    # [B,S]
                per_path_max_lp.append(max_lp)

            stacked = torch.stack(per_path_max_lp, dim=-1)    # [B,S,4]
            best_idx = stacked.argmax(dim=-1)                 # [B,S]
            valid = attention_mask.bool()

            for idx, name in enumerate(PATHS):
                path_selections[name] += ((best_idx == idx) & valid).sum().item()

            total_tokens += valid.sum().item()

    print(f"Path usage analysis (over {num_batches} batches):")
    for name in PATHS:
        cnt = path_selections[name]
        pct = (100.0 * cnt / total_tokens) if total_tokens else 0.0
        print(f"  {name}: {pct:.1f}% ({cnt:,} tokens)")
    print(f"  Total analyzed tokens: {total_tokens:,}")

def apply_freeze_schedule(trainer, step, freeze_schedule, applied_steps):
    if not freeze_schedule:
        return
    for s in sorted(int(k) for k in freeze_schedule.keys()):
        if s <= step and s not in applied_steps:
            trainer.model.set_path_freezing(freeze_schedule[str(s)])
            applied_steps.add(s)
            print(f"Applied freeze config @ step {step}: {freeze_schedule[str(s)]}")

def _parse_gate_temp_points(spec: str):
    if not spec:
        return None
    pts = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        f, t = tok.split(":")
        pts.append((float(f), float(t)))
    pts.sort(key=lambda x: x[0])
    return pts

def _interp_piecewise(points, x, default=0.5):
    # points: list of (frac, value) sorted by frac
    if not points:
        return default
    if x <= points[0][0]:
        return points[0][1]
    for i in range(1, len(points)):
        x0, y0 = points[i-1]
        x1, y1 = points[i]
        if x <= x1:
            # linear interpolate
            alpha = (x - x0) / max(1e-12, (x1 - x0))
            return y0 + alpha * (y1 - y0)
    return points[-1][1]

def _compute_hard_window(args, total_optimizer_steps):
    # Fractions (opt-in)
    if args.hard_from_frac >= 0.0 and args.hard_to_frac > 0.0:
        start = int(round(args.hard_from_frac * total_optimizer_steps))
        end   = int(round(args.hard_to_frac   * total_optimizer_steps))
        return max(0, start), max(start, end)

    # Absolute steps (opt-in)
    if args.hard_from_step >= 0 and args.hard_to_step > 0:
        return args.hard_from_step, max(args.hard_from_step, args.hard_to_step)

    # Default: DISABLED (Phase A behavior)
    return -1, -1

@torch.inference_mode()
def quick_preview_greedy(model, tokenizer, prompt_text, *, max_new_tokens=32, path_selection="hierarchical_gate"):
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    attn = torch.ones_like(input_ids)

    for _ in range(max_new_tokens):
        out = model(input_ids=input_ids, attention_mask=attn, path_selection=path_selection)
        scores = _get_log_probs_from_output(out)    
        next_id = scores[:, -1, :].argmax(dim=-1, keepdim=True)            # greedy
        input_ids = torch.cat([input_ids, next_id], dim=1)
        attn = torch.cat([attn, attn.new_ones((attn.size(0), 1))], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

@torch.inference_mode()
def eval_ppl_safe(model, dataloader, *, path_selection="hierarchical_gate"):
    device = next(model.parameters()).device
    total_logprob = 0.0
    total_tokens = 0
    total_correct = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention  = batch["attention_mask"].to(device)
        labels     = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention, path_selection=path_selection)
        logp = _get_log_probs_from_output(out)
        logp = logp[:, :-1, :]                                                 # shift
        gold = labels[:, 1:]                                                   # [B,S-1]
        mask = attention[:, 1:].bool()

        V = logp.size(-1)
        flat_lp   = logp.reshape(-1, V)
        flat_gold = gold.reshape(-1)
        flat_msk  = mask.reshape(-1)
        kept_lp   = flat_lp[flat_msk]
        kept_gold = flat_gold[flat_msk]
        tok_logp  = kept_lp.gather(1, kept_gold.view(-1,1)).squeeze(1)         # [K]

        preds = kept_lp.argmax(dim=-1)
        total_correct += int((preds == kept_gold).sum().item())
        total_logprob += float(tok_logp.sum().item())
        total_tokens  += int(kept_gold.numel())

    ppl = math.exp(-total_logprob / max(1, total_tokens))
    acc = total_correct / max(1, total_tokens)
    return ppl, acc

def _lm_map(train_indices, n_heads=8, freeze_anchor0=True, as_str=True):
    m = {}
    for i in range(n_heads):
        # head-0 stays frozen as anchor by default
        is_frozen = (i == 0 and freeze_anchor0) or (i not in train_indices)
        key = str(i) if as_str else i
        m[key] = is_frozen
    return m

def _ensure_log_probs(t: torch.Tensor, dim: int = -1, atol: float = 1e-3) -> torch.Tensor:
    """
    Return log-probs with the same shape as `t` (handles logits or already log-probs).
    Heuristic: if logsumexp over vocab ≈ 0, treat as log-probs; else apply log_softmax.
    """
    # Scalar edge case (rare)
    if t.dim() == 0:
        return F.log_softmax(t.unsqueeze(0), dim=0).squeeze(0)

    x = t.float()  # safe for fp16/bf16
    # Build a representative vector along the vocab dimension
    if x.dim() == 1:
        sample = x
    else:
        # move `dim` to last, then take the first row across the collapsed batch/time
        vdim = dim if dim >= 0 else (x.dim() + dim)
        sample = x.transpose(vdim, -1).reshape(-1, x.size(vdim))[0]  # [V]

    lse = torch.logsumexp(sample, dim=-1)  # scalar
    is_logprob = torch.isfinite(lse) and torch.allclose(
        lse, torch.tensor(0.0, device=sample.device), atol=atol
    )
    return t if is_logprob else F.log_softmax(t, dim=dim)

def _get_log_probs_from_output(out):
    """Return [B,S,V] log-probs regardless of what the model returned."""
    # Prefer native log-probs
    if isinstance(out, dict) and (out.get("log_probs", None) is not None):
        lp = out["log_probs"]
        if lp.dim() != 3:
            raise ValueError(f"log_probs must be [B,S,V], got {tuple(lp.shape)}")
        return lp

    # Fallback to logits → normalize if needed
    logits = out["logits"] if isinstance(out, dict) else out.logits
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B,S,V], got {tuple(logits.shape)}")
    return _ensure_log_probs(logits, dim=-1)

def migrate_lm_heads_from_quad_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    source_state = ckpt.get("model_state_dict", ckpt)

    # Old quad keys
    quad_keys = {
        "left_left":  "lm_head_left_left.weight",
        "left_right": "lm_head_left_right.weight",
        "right_left": "lm_head_right_left.weight",
        "right_right":"lm_head_right_right.weight",
    }

    # 1) Load all compatible (non-LM-head) weights first, as before
    model_state = model.state_dict()
    compatible = {k: v for k, v in source_state.items()
                  if k in model_state and v.shape == model_state[k].shape}

    # 2) Copy each quad head into the *first* allocated multi-head, then
    #    duplicate to the rest of that path's allocation.
    with torch.no_grad():
        for path_name, old_key in quad_keys.items():
            if old_key not in source_state:
                continue
            srcW = source_state[old_key]
            head_ids = model.head_allocation.get(path_name, [])
            if not head_ids:
                continue
            anchor = head_ids[0]
            # copy to anchor
            model.lm_head.perceptrons[anchor].weight.copy_(srcW)
            # duplicate to others (optionally add tiny noise to break symmetry)
            for hid in head_ids[1:]:
                model.lm_head.perceptrons[hid].weight.copy_(srcW)
                model.lm_head.perceptrons[hid].weight.add_(0.001 * torch.randn_like(srcW))

    missing = [k for k in model_state.keys() if k not in compatible]
    print(f"Migrated quad→multi heads per head_allocation={model.head_allocation}. "
          f"Copied quad heads into indices by path; non-LM compatible={len(compatible)}.")
    return compatible

def set_head_gate_hparams(self, fast_k=None, temp=1.0):
    self.head_fast_k = fast_k
    self.head_gate_temp = temp

def log_lm_head_stats(trainer, batch, step, label="train", path_selection=None):
    model = trainer.model
    sel = path_selection or getattr(trainer, "train_path_selection", None) or "hierarchical_gate"
    model.eval()
    with torch.inference_mode():
        input_ids = batch["input_ids"].to(trainer.device)
        attention_mask = batch["attention_mask"].to(trainer.device)
        labels = batch.get("labels", input_ids).to(trainer.device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            path_selection=sel,          # same routing as train/eval
            return_head_indices=True
        )

        if not isinstance(out, dict) or out.get("head_top_idx") is None:
            print(f"[{label} @ step {step}] head-usage/acc: (skipped — head mixture disabled or unavailable)")
            model.train(); return
        if out.get("gate") is None or "final_weights" not in out["gate"]:
            print(f"[{label} @ step {step}] head-usage/acc: (skipped — no gates in this mode)")
            model.train(); return

        hti = out["head_top_idx"]
        need = ("left_left","left_right","right_left","right_right")
        if not all(k in hti for k in need):
            print(f"[{label} @ step {step}] head-usage/acc: (skipped — missing head_top_idx keys)")
            model.train(); return

        # --- pick the SAME leaf per token the model used ---
        fw = out["gate"]["final_weights"]  # dict of [B,S,1]
        w = torch.stack([fw["left_left"], fw["left_right"], fw["right_left"], fw["right_right"]], dim=-1).squeeze(-2)  # [B,S,4]
        best_path = w.argmax(dim=-1)[..., :-1]  # [B,S-1]

        # quick sanity: how many tokens went to each leaf?
        path_counts = best_path.flatten().bincount(minlength=4).tolist()
        print(f"[{label} @ step {step}] path-tokens  LL={path_counts[0]} LR={path_counts[1]} RL={path_counts[2]} RR={path_counts[3]}")

        # --- merge per-path head ids using that chosen path ---
        names = ("left_left","left_right","right_left","right_right")
        chosen_head = torch.full_like(best_path, -1, dtype=torch.long)  # sentinel
        for idx, name in enumerate(names):
            per_path = hti[name][..., :-1].to(torch.long)
            mask = (best_path == idx)
            chosen_head = torch.where(mask, per_path, chosen_head)

        if (chosen_head == -1).any():
            miss = int((chosen_head == -1).sum())
            print(f"[warn] {miss} positions not assigned a head id (check masks/paths).")

        # --- optional: verify mapping matches head_allocation ---
        if getattr(model, "head_allocation", None):
            alloc = model.head_allocation
            ok_ranges = {
                "left_left":  set(alloc.get("left_left", [])),
                "left_right": set(alloc.get("left_right", [])),
                "right_left": set(alloc.get("right_left", [])),
                "right_right":set(alloc.get("right_right", [])),
            }
            for idx, name in enumerate(names):
                used = chosen_head[best_path == idx]
                if used.numel():
                    uniq = set(used.unique().tolist())
                    if not uniq.issubset(ok_ranges[name]):
                        print(f"[warn] {name} heads out of range: {sorted(uniq)} not in {sorted(ok_ranges[name])}")

        # --- compute usage/acc ---
        lp = out.get("log_probs")
        if lp is None:
            logits = out["logits"]
            lp = torch.log_softmax(logits, dim=-1)
        preds = lp[..., :-1, :].argmax(dim=-1)  # [B,S-1]
        gold  = labels[..., 1:]                 # [B,S-1]
        valid = attention_mask[..., 1:].bool()
        correct = (preds == gold) & valid

        n_heads = getattr(model.lm_head, "num_perceptrons", 8)
        HEAD_ORDER = list(range(n_heads))
        sel_cnt = {h: 0 for h in HEAD_ORDER}
        cor_cnt = {h: 0 for h in HEAD_ORDER}
        tot = int(valid.sum().item())

        for h in HEAD_ORDER:
            m = valid & (chosen_head == h)
            sel_cnt[h] = int(m.sum().item())
            cor_cnt[h] = int((m & correct).sum().item())

        usage = {h: (100.0 * sel_cnt[h] / tot if tot else 0.0) for h in HEAD_ORDER}
        acc   = {h: (100.0 * cor_cnt[h] / sel_cnt[h] if sel_cnt[h] else float("nan")) for h in HEAD_ORDER}

    usage_str = " ".join([f"H{h}:{usage[h]:4.1f}%" for h in HEAD_ORDER])
    acc_str   = " ".join([f"H{h}:{(acc[h] if sel_cnt[h] else ' n/a'):>5}" for h in HEAD_ORDER])
    print(f"[{label} @ step {step}] head-usage  {usage_str}")
    print(f"[{label} @ step {step}] head-acc    {acc_str}")
    model.train()

def debug_head_gates(trainer, batch):
    """Debug head gate behavior - shows gate outputs and head selections"""
    model = trainer.model
    model.eval()
    with torch.no_grad():
        input_ids = batch["input_ids"].to(trainer.device)
        attention_mask = batch["attention_mask"].to(trainer.device)
        
        print(f"Batch shape: {input_ids.shape}")
        
        # Get embeddings and shared layers
        hidden_states = model.get_embeddings(input_ids, attention_mask)
        shared_output = model.forward_shared_layers(hidden_states, attention_mask)
        
        # Get intermediate outputs
        left_intermediate = model.forward_intermediate_layers(shared_output.clone(), attention_mask, "left")
        right_intermediate = model.forward_intermediate_layers(shared_output.clone(), attention_mask, "right")
        
        print(f"Left intermediate shape: {left_intermediate.shape}")
        print(f"Right intermediate shape: {right_intermediate.shape}")
        
        # Forward through final transformer layers to get the ACTUAL hidden states used by head gates
        path_final_hiddens = {}
        for path_name in ["left_left", "left_right", "right_left", "right_right"]:
            if path_name.startswith("left"):
                x_input = left_intermediate.clone()
            else:
                x_input = right_intermediate.clone()
            
            # Forward through the final transformer layers for this path
            final_hidden = model.forward_final_path(x_input, attention_mask, path_name)
            path_final_hiddens[path_name] = final_hidden
            print(f"{path_name} final hidden shape: {final_hidden.shape}")
        
        print("\n=== Head Gate Analysis ===")
        
        # Check head gate outputs for each path using FINAL hidden states
        for path_name in ["left_left", "left_right", "right_left", "right_right"]:
            if path_name in model._heads_gate_by_path:
                gate = model._heads_gate_by_path[path_name]
                final_hidden = path_final_hiddens[path_name]
                
                # Get gate output using final hidden states (what the model actually uses)
                gate_output = gate(final_hidden)
                gate_probs = F.softmax(gate_output, dim=-1)
                
                print(f"\n{path_name}:")
                print(f"  Gate output shape: {gate_output.shape}")
                print(f"  Expected heads: {model.head_allocation[path_name]}")
                print(f"  Gate mean probs: {gate_probs.mean(dim=(0,1)).cpu().numpy()}")
                print(f"  Gate std probs: {gate_probs.std(dim=(0,1)).cpu().numpy()}")
                
                # Show which heads are actually selected (argmax)
                selected_heads = gate_probs.argmax(dim=-1)  # [B, S]
                unique_selections, counts = selected_heads.unique(return_counts=True)
                
                print(f"  Local head selections:")
                for local_idx, count in zip(unique_selections, counts):
                    global_idx = model.head_allocation[path_name][local_idx.item()]
                    pct = count.item() / selected_heads.numel() * 100
                    print(f"    Local {local_idx.item()} -> Global {global_idx}: {count.item()} tokens ({pct:.1f}%)")
                
                # Check if gate parameters are trainable
                gate_trainable = any(p.requires_grad for p in gate.parameters())
                print(f"  Gate trainable: {gate_trainable}")
                
                # Show gate parameter statistics
                for name, param in gate.named_parameters():
                    print(f"  {name}: requires_grad={param.requires_grad}, "
                          f"mean={param.data.mean().item():.6f}, "
                          f"std={param.data.std().item():.6f}")
        
        print("\n=== Path Routing Check ===")
        
        # Run a forward pass to see actual path routing
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask, 
                       path_selection="hierarchical_gate", return_head_indices=True)
            
            if "gate" in out and "final_weights" in out["gate"]:
                fw = out["gate"]["final_weights"]
                weights = torch.stack([fw["left_left"], fw["left_right"], 
                                     fw["right_left"], fw["right_right"]], dim=-1).squeeze(-2)
                path_selections = weights.argmax(dim=-1)
                
                path_names = ["left_left", "left_right", "right_left", "right_right"]
                valid_tokens = attention_mask.sum().item()
                
                print(f"Valid tokens: {valid_tokens}")
                for i, pname in enumerate(path_names):
                    count = (path_selections == i).sum().item()
                    pct = count / valid_tokens * 100 if valid_tokens > 0 else 0
                    print(f"  {pname}: {count} tokens ({pct:.1f}%)")
            
            # Check head indices if available
            if "head_top_idx_combined" in out and out["head_top_idx_combined"] is not None:
                head_indices = out["head_top_idx_combined"]
                print(f"\nActual head usage in this batch:")
                for head_id in range(model.lm_head.num_perceptrons):
                    count = (head_indices == head_id).sum().item()
                    pct = count / attention_mask.sum().item() * 100 if attention_mask.sum().item() > 0 else 0
                    print(f"  Head {head_id}: {count} tokens ({pct:.1f}%)")
            else:
                print("\nNo head_top_idx_combined in output")
                
        except Exception as e:
            print(f"Forward pass failed: {e}")

        model.train()
        
        print("\n=== Gate Temperature Settings ===")
        print(f"Head gate temp: {getattr(model, 'head_gate_temp', 'Not set')}")
        print(f"Head fast k: {getattr(model, 'head_fast_k', 'Not set')}")
        print(f"Use head mixture: {getattr(model, 'use_head_mixture', False)}")

def debug_head_gate_gradients(model, batch, loss):
    """Debug whether head gates are receiving gradients"""
    print("\n=== Head Gate Gradient Check ===")
    
    # Check if gradients are flowing to head gates
    for path_name, gate in model._heads_gate_by_path.items():
        print(f"\n{path_name} gate gradients:")
        for name, param in gate.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"  {name}: grad_norm={grad_norm:.8f}")
            else:
                print(f"  {name}: No gradient")

def simple_head_usage_stats(trainer, batch):
    model = trainer.model
    model.eval()
    with torch.no_grad():
        input_ids = batch["input_ids"].to(trainer.device)
        attention_mask = batch["attention_mask"].to(trainer.device)
        
        # Forward pass with return_all_paths to get individual path outputs
        path_outputs = model(input_ids=input_ids, attention_mask=attention_mask, 
                           return_all_paths=True, return_head_indices=True)
        
        head_usage = {i: 0 for i in range(model.lm_head.num_perceptrons)}
        total_tokens = 0
        
        for path_name in ["left_left", "left_right", "right_left", "right_right"]:
            if path_name in path_outputs.get("head_top_idx", {}):
                head_indices = path_outputs["head_top_idx"][path_name]
                mask = attention_mask.bool()
                if head_indices.size(0) == mask.size(0) and head_indices.size(1) == mask.size(1):
                    for i in range(model.lm_head.num_perceptrons):
                        count = ((head_indices == i) & mask).sum().item()
                        head_usage[i] += count
                    total_tokens += mask.sum().item()
        
        print("Simple head usage:")
        for i in range(model.lm_head.num_perceptrons):
            pct = (head_usage[i] / total_tokens * 100) if total_tokens > 0 else 0
            print(f"  Head {i}: {pct:.1f}%")

    model.train()

def _valid_mask_from(labels, attn_mask):
    gold = labels[..., 1:].contiguous()
    if attn_mask is not None:
        valid = attn_mask[..., 1:].to(torch.float32)
    else:
        valid = torch.ones_like(gold, dtype=torch.float32)
    if (gold == -100).any():
        valid = valid * (gold != -100).to(valid.dtype)
    return valid  # [B,S-1] float

def _logp_targets(out_dict):
    # your model returns normalized log-probs
    return out_dict["log_probs"][..., :-1, :]  # [B,S-1,V]

def _kl_soft_to_hard(hard_logp, soft_logp, valid_mask):
    # KL(soft || hard), mean over valid tokens
    kd_tok = F.kl_div(hard_logp, soft_logp.exp(), reduction="none", log_target=False).sum(-1)
    return (kd_tok * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

def _uncertain_mask_from_gate(gate_info, margin):
    """Return [B,S-1] bool mask where leaf decision is 'uncertain' (w1 - w2 < margin)."""
    if gate_info is None or "final_weights" not in gate_info:
        return None
    fw = gate_info["final_weights"]  # dict of 4 tensors [B,S,1]
    w_ll, w_lr, w_rl, w_rr = (fw["left_left"], fw["left_right"], fw["right_left"], fw["right_right"])
    w = torch.stack([w_ll, w_lr, w_rl, w_rr], dim=-1).squeeze(-2)  # [B,S,4]
    top, idx = w.topk(2, dim=-1)  # [B,S,2]
    margin_gap = top[..., 0] - top[..., 1]  # [B,S]
    m = (margin_gap < margin)  # [B,S] bool
    return m[..., 1:]  # align to targets [B,S-1]


def main():
    args = get_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Environment setup
    RUN_MODE = "colab" if "COLAB_GPU" in os.environ else "local"
    
    if RUN_MODE == "colab":
        BASE_PATH = Path("/content/drive/My Drive/Project1")
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except ImportError:
            pass
    else:
        BASE_PATH = Path("C:/Machine Learning/Project1")
    
    BASE_PATH.mkdir(parents=True, exist_ok=True)
    
    # Setup checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = BASE_PATH / "quad_path_headers_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running in {RUN_MODE} mode")
    print(f"Base path: {BASE_PATH}")
    print(f"Checkpoints: {checkpoint_dir}")
    
    # Setup tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model configuration
    config = GPT2Config.from_pretrained(args.pretrained_model)
    config.n_positions = args.max_length
    config.n_lm_perceptrons = args.n_lm_perceptrons  # <- add this

    # ---- head allocation loading (safe) ----
    head_alloc = None
    if args.use_head_mixture:
        if not args.head_allocation:
            raise FileNotFoundError(
                "You enabled --use_head_mixture but did not pass --head_allocation "
                "(e.g., /.../head_allocation_3_1_2_2.json). "
                "Either supply the file or remove --use_head_mixture."
            )
        with open(args.head_allocation, "r") as f:
            head_alloc = json.load(f)

    # Load freeze schedule if provided
    freeze_schedule = None
    if args.freeze_schedule:
        with open(args.freeze_schedule, 'r') as f:
            freeze_schedule = json.load(f)
        print(f"Loaded freeze schedule: {freeze_schedule}")
    
    # Create model
    print(f"Creating quad path model with LM headers:")
    print(f"  - Splits at layers {args.split_at_layer_1} and {args.split_at_layer_2}")
   
    # Create model (use the head_alloc you just loaded)
    model = HierarchicalMultiPathGPT2(
        config,
        pretrained_model=args.pretrained_model,
        split_at_layer_1=args.split_at_layer_1,
        split_at_layer_2=args.split_at_layer_2,
        head_allocation=head_alloc,
        gate_hidden=args.gate_hidden,
    )
    model.use_head_mixture = bool(args.use_head_mixture)

    model.head_gate_temp = args.head_gate_temp
    model.head_fast_k = args.head_topk if args.head_topk is not None and args.head_topk > 0 else None

    model.freeze_split_gates(args.freeze_split_gates)

    '''
    if model.use_head_mixture :
        assert len(model.lm_head.perceptrons) == 8, "Expected 8 LM heads"
        assert model.head_allocation is not None, "Head allocation JSON not loaded"
        exp = {"left_left":[0,1,2], "left_right":[3], "right_left":[4,5], "right_right":[6,7]}
        assert model.head_allocation == exp, f"Unexpected allocation: {model.head_allocation}"
        print("Per-path head allocation =", model.head_allocation)
        print("Head mixture ON? ", getattr(model, "use_head_mixture", False))


    # Add this to your training script after model creation
    print("Head allocation verification:")
    for path_name, head_ids in model.head_allocation.items():
        print(f"{path_name}: heads {head_ids}")
        if path_name in model._heads_gate_by_path:
            gate = model._heads_gate_by_path[path_name]
            # Check the output dimension of the gate
            if hasattr(gate, 'net') and isinstance(gate.net, nn.Sequential):
                # Get the last linear layer in the sequential
                last_layer = None
                for layer in reversed(gate.net):
                    if isinstance(layer, nn.Linear):
                        last_layer = layer
                        break
                if last_layer:
                    print(f"  Gate output dimension: {last_layer.out_features} (should be {len(head_ids)})")
                else:
                    print(f"  Gate structure: {gate.net}")
            else:
                print(f"  Gate structure: {gate}")
    '''
    
    # Check head gate parameter status
    print("Head gate parameter status:")
    for path_name, gate in model._heads_gate_by_path.items():
        for name, param in gate.named_parameters():
            print(f"{path_name}.{name}: requires_grad={param.requires_grad}")

    # Create trainer
    trainer = QuadPathTrainer(
        model, tokenizer, device, checkpoint_dir,
        lb_coef=args.lb_coef, 
        gold_aux_coef=args.gold_aux_coef, 
        tether_coef=args.tether_coef,
        gate_temp=args.gate_temp,
        clip_grad=args.clip_grad_norm  # Map your arg to the expected parameter name
    )
    trainer.consistency_lambda = args.consistency_lambda
    trainer.kd_coef = args.kd_coef
    trainer.kd_teacher_selection = args.kd_teacher_selection
    trainer.kd_on = args.kd_on
    trainer.kd_margin = args.kd_margin

    # Create optimizer with separate parameter groups
    try:
        optimizer = trainer.create_optimizer(
            lr=args.lr, head_lr=args.head_lr, gate_lr=args.gate_lr, weight_decay=args.weight_decay,
            foreach=False,   
            fused=False      
        )
    except TypeError:
        optimizer = trainer.create_optimizer(lr=args.lr, weight_decay=args.weight_decay)

    # ---- Sanity checks: gates group ----
    gate_names = [n for n, p in trainer.model.named_parameters()
                if p.requires_grad and n.startswith("_heads_gate_by_path")]

    found_gates_group = False
    for g in optimizer.param_groups:
        if g.get("name") == "gates":
            found_gates_group = True
            print("[check] gates lr =", g["lr"])
            owned = {id(p) for p in g["params"]}
            # sample a few to show they’re inside the group
            sample = [n for n, p in trainer.model.named_parameters()
                    if id(p) in owned and n.startswith("_heads_gate_by_path")]
            print("[check] sample gate params:", sample[:4])

            # ensure all expected gate params are actually in this group
            gates_in_group = set(n for n, p in trainer.model.named_parameters() if id(p) in owned)
            missing_in_group = [n for n in gate_names if n not in gates_in_group]
            if missing_in_group:
                print("[warn] some _heads_gate_by_path params not in gates group:", missing_in_group[:8])
            break

    if not found_gates_group:
        if gate_names:
            print("[warn] 'gates' param group not found but gate params require_grad=True:",
                gate_names[:8])
        else:
            print("[info] no trainable _heads_gate_by_path params (likely frozen)")

    # ---- Optional: sanity check LM heads group ----
    lm_head_names = [n for n, p in trainer.model.named_parameters()
                    if p.requires_grad and n.startswith("lm_head.perceptrons")]
    for g in optimizer.param_groups:
        if g.get("name") == "lm_heads":
            print("[check] lm_heads lr =", g["lr"])
            owned = {id(p) for p in g["params"]}
            sample = [n for n, p in trainer.model.named_parameters()
                    if id(p) in owned and n.startswith("lm_head.perceptrons")]
            print("[check] sample lm_head params:", sample[:4])
            break

    # Load datasets
    print("Loading datasets...")
    train_dataset = WikiTextDataset(
        data_dir=BASE_PATH / "wikitext-103",
        tokenizer=tokenizer,
        max_length=args.max_length,
        split="train",
        max_samples=args.max_samples
    )
    
    val_dataset = WikiTextDataset(
        data_dir=BASE_PATH / "wikitext-103",
        tokenizer=tokenizer,
        max_length=args.max_length,
        split="valid",
        max_samples=1000
    )
    
    # Create DataLoaders
    collate_fn = create_collate_fn(tokenizer)
    
    if RUN_MODE == "colab":
        loader_config = {'num_workers': 2, 'pin_memory': True}
    else:
        loader_config = {'num_workers': 0, 'pin_memory': torch.cuda.is_available()}
    
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
    
    print(f"Training batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")
    
    # Setup scheduler
    batches_per_epoch = len(train_dataloader)
    epoch_steps = math.ceil(batches_per_epoch * args.epochs / args.grad_accum_steps)
    total_optimizer_steps = args.max_steps if args.max_steps > 0 else epoch_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_optimizer_steps
    )

    print("Training setup:")
    print(f"  Total steps: {total_optimizer_steps:,}")
    print(f"  Warmup steps: {args.warmup_steps:,}")
    print(f"  Gradient accumulation: {args.grad_accum_steps}")
    print(f"  Train path selection: {args.train_path_selection}")
    print(f"  Eval path selection: {args.eval_path_selection}")
    print(f"  Head LR: {args.head_lr:.2e}, Gate LR: {args.gate_lr:.2e}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        ckpt = None
        try:
            ckpt = torch.load(args.resume, map_location=device, weights_only=True)
            print(f"Resumed (weights_only=True) from: {args.resume}")
        except Exception as e1:
            print(f"Failed weights_only=True: {e1}")
            try:
                ckpt = torch.load(args.resume, map_location=device, weights_only=False)
                print(f"[warn] Resumed (weights_only=False) from: {args.resume}")
            except Exception as e2:
                print(f"Failed to load checkpoint: {e2}\nStarting fresh training")
                ckpt = None

        if ckpt is not None:
            # model state
            state = None
            if isinstance(ckpt, dict):
                state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
                if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                    state = ckpt
            if state is None:
                raise RuntimeError("Checkpoint missing model state dict")

            incompat = model.load_state_dict(state, strict=False)
            missing, unexpected = incompat.missing_keys, incompat.unexpected_keys
            print(f"Loaded model state. missing={len(missing)}, unexpected={len(unexpected)}")

            # optimizer
            opt_sd = ckpt.get("optimizer_state_dict")
            if opt_sd is not None:
                try:
                    optimizer.load_state_dict(opt_sd)
                    print("Loaded optimizer state.")
                except Exception as e:
                    print(f"[warn] Could not load optimizer state: {e}")

            # epoch
            start_epoch = int(ckpt.get("epoch", 0))
            print(f"Resumed epoch = {start_epoch}")

    # Optional: initialize weights from a prior QUAD checkpoint (one-time)
    if args.init_from_quad:
        print(f"Migrating LM heads from quad checkpoint: {args.init_from_quad}")
        model = migrate_lm_heads_from_quad_checkpoint(model, args.init_from_quad)
    
    # Training loop
    print("\nStarting training...")
    global training_interrupted

    total_train_loss = 0.0
    num_train_steps = 0

    trainer.save_checkpoint(0, optimizer, 0)
    
    acc_steps = args.grad_accum_steps
    trainer.zero_grad(optimizer)

    current_epoch = 0
    training_interrupted = False
    total_step = 0

    # after loading checkpoint
    current_epoch = start_epoch
    applied_steps = set()

    try:
        for epoch in range(start_epoch, args.epochs):
            current_epoch = epoch
            epoch_start_time = time.time()

            # ---- Phase management for head-only training ----
            if args.freeze_all_transformer:
                # Apply split-gate freeze according to the flag (layer-6 / layer-9 gates)
                # Do this BEFORE creating the optimizer so frozen params aren't added.
                try:
                    model_ref = trainer.model  # if you're inside a Trainer wrapper
                except NameError:
                    model_ref = model

                model_ref.freeze_split_gates(args.freeze_split_gates)

                split_status = "frozen" if args.freeze_split_gates else "TRAINABLE"
                print(
                    f"\nGLOBAL FREEZE ACTIVE: transformer 0..11 frozen; "
                    f"split gates {split_status}; training only LM heads + head-gates."
                )

            else:
                if args.head_only_epochs > 0 and epoch < args.head_only_epochs:
                    print(f"\nEpoch {epoch+1}: HEAD-ONLY PHASE (train LM heads [+ gates if desired])")
                    trainer.model.set_path_freezing({
                        "shared": True,
                        "left_intermediate": True,
                        "right_intermediate": True,
                        "left_right": True,
                        "right_left": True,
                        "right_right": True,
                        "lm_headers": _lm_map(train_indices=range(1, 8), n_heads=8, freeze_anchor0=True, as_str=True),
                    })
                elif args.head_only_epochs > 0 and epoch == args.head_only_epochs:
                    print(f"\nEpoch {epoch+1}: SWITCHING TO FULL TRAINING")
                    trainer.model.set_path_freezing({
                        "shared": False,
                        "left_intermediate": False,
                        "right_intermediate": False,
                        "left_right": False,
                        "right_left": False,
                        "right_right": False,
                        "lm_headers": _lm_map(train_indices=range(1, 8), n_heads=8, freeze_anchor0=True, as_str=True),
                    })
                    try:
                        optimizer = trainer.create_optimizer(
                            lr=args.lr, head_lr=args.head_lr, gate_lr=args.gate_lr, weight_decay=args.weight_decay
                        )
                    except TypeError:
                        optimizer = trainer.create_optimizer(lr=args.lr, weight_decay=args.weight_decay)
                else:
                    print(f"\nEpoch {epoch+1}: FULL TRAINING PHASE")

            model.backbone_is_frozen = bool(args.freeze_all_transformer)

            for i, batch in enumerate(train_dataloader):
                if args.max_steps and total_step >= args.max_steps:
                    print(f"Reached max_steps={args.max_steps}, stopping training.")
                    training_interrupted = True
                    break

                # Apply freeze schedule if provided
                apply_freeze_schedule(trainer, total_step, freeze_schedule, applied_steps)

                if not hasattr(args, "_hard_window"):
                    args._hard_window = _compute_hard_window(args, total_optimizer_steps)
                hard_start, hard_end = args._hard_window

                # Use trainer's live mode in backward_only
                if hard_start >= 0 and hard_start <= total_step < hard_end:
                    trainer.train_path_selection = "gate_hard"
                else:
                    trainer.train_path_selection = args.train_path_selection

                metrics = trainer.backward_only(
                    batch,
                    path_selection=trainer.train_path_selection,   
                    loss_scale=1.0 / acc_steps
                )
                
                total_train_loss += metrics['loss']
                num_train_steps += 1

                # Optimizer step every acc_steps micro-batches
                if ((i + 1) % acc_steps) == 0:
                    # Step optim + scheduler first
                    trainer.optimizer_step(optimizer)
                    scheduler.step()
                    total_step += 1
                    
                    if training_interrupted:
                        break

                    # Parse gate temp points once
                    pts = getattr(args, "_gate_temp_points_parsed", None)
                    if pts is None:
                        args._gate_temp_points_parsed = _parse_gate_temp_points(args.gate_temp_points)
                        pts = args._gate_temp_points_parsed

                    progress = total_step / float(total_optimizer_steps + 1e-8)

                    # Default: keep Phase A fixed temp
                    if pts:
                        trainer.gate_temp = _interp_piecewise(pts, progress, default=args.gate_temp)
                    else:
                        trainer.gate_temp = args.gate_temp

                    # Logging on optimizer steps
                    if args.log_every and (total_step % args.log_every == 0):
                        log_str = (
                            f"[step {total_step}/{total_optimizer_steps}] "
                            f"loss={metrics['loss']:.4f} ce={metrics['ce']:.4f} "
                            f"acc={metrics['accuracy']:.3f} "
                            f"gold_aux={metrics['gold_aux']:.4f} "
                            f"lb_loss={metrics['lb_loss']:.4f} "
                            f"tether={metrics['tether']:.4f}"
                        )
                        
                        # Add gate stats if available
                        if 'gate1_right_pct' in metrics:
                            log_str += f" g1_right%={metrics['gate1_right_pct']:.3f}"
                        if 'gate1_entropy' in metrics:
                            log_str += f" g1_H={metrics['gate1_entropy']:.3f}"
                        if 'left_left_usage' in metrics:
                            log_str += (f" usage: LL={metrics.get('left_left_usage', 0):.3f} "
                                      f"LR={metrics.get('left_right_usage', 0):.3f} "
                                      f"RL={metrics.get('right_left_usage', 0):.3f} "
                                      f"RR={metrics.get('right_right_usage', 0):.3f}")
                        
                        print(log_str)

                        # Add LM head stats
                        log_lmstr = ""
                        lm_stats = trainer.model.lm_head.get_perceptron_stats()
                        if lm_stats['total_usage'] > 0:
                            usage_str = " | ".join(f"h{i}:{pct:.1f}%" for i, pct in enumerate(lm_stats['usage_percentages']))
                            log_lmstr += f" | LM_heads: {usage_str}"  # Use += to append
                        else:
                            log_lmstr += f" | LM_heads: no usage data"  # Use += to append

                        print(log_lmstr)

                        # Add head gate debugging
                        #debug_head_gates(trainer, batch)
    
                    # -------- lightweight preview generation --------
                    if args.eval_every and (total_step % args.eval_every == 0):
                        prev_mode = trainer.model.training
                        trainer.model.eval()
                        try:
                            ctx_ids = batch["input_ids"][0][:32].detach().cpu()
                            prompt_text = tokenizer.decode(ctx_ids, skip_special_tokens=True)

                            hard_text = quick_preview_greedy(trainer.model, tokenizer, prompt_text,
                                                            max_new_tokens=32, path_selection="gate_hard")
                            soft_text = quick_preview_greedy(trainer.model, tokenizer, prompt_text,
                                                            max_new_tokens=32, path_selection="hierarchical_gate")

                            print(f"\n🎯 Prompt: '{prompt_text}'")
                            print(f"🎯 Generated (gate_hard): '{hard_text[len(prompt_text):].strip()}'")
                            print(f"🎯 Generated (gate_soft): '{soft_text[len(prompt_text):].strip()}'")
                        except Exception as e:
                            print(f"⚠️ Sample generation failed: {e}")

                        # — regular eval —
                        val = trainer.evaluate(val_dataloader, path_selection=args.eval_path_selection)
                        eval_str = (
                            f"[eval @ step {total_step}] "
                            f"loss={val['loss']:.4f} PPL={val['perplexity']:.3f} "
                            f"acc={val['accuracy']:.3f}"
                        )

                        # — safe PPL (ground truth) —
                        ppl_safe, acc_safe = eval_ppl_safe(trainer.model, val_dataloader,
                                                        path_selection=args.eval_path_selection)
                        print(f"[eval-safe @ step {total_step}] PPL={ppl_safe:.3f} acc={acc_safe:.3f}")

                        # gate stats (optional)
                        if 'gate1_right_pct' in val:
                            eval_str += f" g1_right%={val['gate1_right_pct']:.3f}"
                        if 'left_left_usage' in val:
                            eval_str += (f" usage: LL={val.get('left_left_usage', 0):.3f} "
                                        f"LR={val.get('left_right_usage', 0):.3f} "
                                        f"RL={val.get('right_left_usage', 0):.3f} "
                                        f"RR={val.get('right_right_usage', 0):.3f}")
                        print(eval_str)

                        # restore original mode once
                        if prev_mode:
                            trainer.model.train()

                    # Periodic checkpoint
                    if args.save_every and (total_step % args.save_every == 0):
                        trainer.save_checkpoint(
                            current_epoch, optimizer, total_train_loss / max(num_train_steps, 1)
                        )

            # Flush remainder micro-batches at end of epoch
            if not training_interrupted:
                remainder = (len(train_dataloader) % acc_steps)
                if remainder != 0:
                    trainer.optimizer_step(optimizer)
                    scheduler.step()
                    total_step += 1

            # End-of-epoch evaluation and statistics
            val = trainer.evaluate(val_dataloader, path_selection=args.eval_path_selection)
            lm_stats = trainer.model.get_lm_head_stats()
            epoch_time = time.time() - epoch_start_time
            
            epoch_str = (
                f"Epoch {current_epoch+1}: "
                f"val loss={val['loss']:.4f} | PPL={val['perplexity']:.3f} | "
                f"acc={val['accuracy']:.3f} | time={epoch_time:.1f}s"
            )
            
            # Add gate stats if available
            if 'gate1_right_pct' in val:
                epoch_str += f" | g1_right%={val['gate1_right_pct']:.3f}"
            if 'left_left_usage' in val:
                epoch_str += (f" | usage: LL={val.get('left_left_usage', 0):.3f} "
                            f"LR={val.get('left_right_usage', 0):.3f} "
                            f"RL={val.get('right_left_usage', 0):.3f} "
                            f"RR={val.get('right_right_usage', 0):.3f}")
            
            print(epoch_str)

            if total_step < args.freeze_head0_steps:
                trainer.model.set_path_freezing({"lm_headers": {"0": True}})
            elif total_step == args.freeze_head0_steps:
                trainer.model.set_path_freezing({"lm_headers": {"0": False}})
            
            # Print LM head statistics
            if lm_stats['total_usage'] > 0:
                usage_str = " | ".join(f"head_{i}: {pct:.1f}%" for i, pct in enumerate(lm_stats['usage_percentages']))
                print(f"  LM Head Usage: {usage_str}")

            log_lm_head_stats(trainer, batch, total_step, label="train",
                            path_selection=args.train_path_selection)
            try:
                val_batch = next(iter(val_dataloader))
                log_lm_head_stats(trainer, val_batch, total_step, label="val",
                                path_selection=args.eval_path_selection)
            except StopIteration:
                pass

            if training_interrupted:
                break

        print(f"Done. Optimizer steps taken: {total_step} (expected ~{total_optimizer_steps})")

        # Final evaluation and analysis
        if not training_interrupted:
            print("\nTraining completed!")
            
            # Final evaluation on all paths and selection strategies
            print("\nFinal evaluation with different path selection strategies:")
            
            # Test main selection strategies
            selection_strategies = ["left_left_only", "hierarchical_gate", "max_prob"]
            results = {}
            
            for path_sel in selection_strategies:
                try:
                    res = trainer.evaluate(val_dataloader, path_selection=path_sel)
                    results[path_sel] = res
                    print(f"  {path_sel}: Loss={res['loss']:.4f}, Acc={res['accuracy']:.2%}, PPL={res['perplexity']:.2f}")
                except Exception as e:
                    print(f"  {path_sel}: Error - {e}")

            # Summary comparison
            if results:
                baseline_ppl = results.get("left_left_only", {}).get("perplexity", float('inf'))
                best_ppl = min(res.get("perplexity", float('inf')) for res in results.values())
                
                print(f"\nSummary:")
                print(f"  Baseline (left_left_only): {baseline_ppl:.2f} PPL")
                print(f"  Best performance: {best_ppl:.2f} PPL")
                if baseline_ppl != float('inf') and best_ppl != float('inf'):
                    improvement = baseline_ppl - best_ppl
                    print(f"  Improvement: {improvement:.2f} PPL ({improvement/baseline_ppl*100:.1f}%)")

            # Path usage analysis
            print("\nFinal path usage analysis:")
            analyze_path_usage(trainer, val_dataloader, num_batches=20)
            
            # LM Head statistics
            print("\nFinal LM Head Statistics:")
            final_lm_stats = trainer.model.get_lm_head_stats()
            for i, (u, s) in enumerate(zip(final_lm_stats['usage_percentages'],
                                        final_lm_stats['selection_percentages'])):
                print(f"  Head {i}: Usage={u:.1f}%, Selections={s:.1f}%")

            # Save final model
            final_path = trainer.save_checkpoint(
                current_epoch, optimizer, total_train_loss / max(num_train_steps, 1), is_final=True
            )
            print(f"\nFinal model saved to: {final_path}")
            
            # Sample generation
            print("\nFinal generation samples:")
            test_prompts = [
                "The future of artificial intelligence is",
                "In a world where technology advances rapidly,",
                "The most important aspect of machine learning"
            ]
            
            for prompt in test_prompts:
                print(f"\nPrompt: '{prompt}'")
                for path_sel in ["left_left_only", "hierarchical_gate", "max_prob"]:
                    try:
                        generated = trainer.generate_sample(
                            prompt,
                            max_length=40,
                            path_selection=path_sel,
                            temperature=0.7
                        )
                        print(f"  {path_sel}: {generated[len(prompt):].strip()}")
                    except Exception as e:
                        print(f"  {path_sel}: Error - {e}")
        
        else:
            print("\nTraining interrupted. Saving final checkpoint...")
            trainer.save_checkpoint(
                current_epoch, optimizer, total_train_loss / max(num_train_steps, 1)
            )
 
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\nAll files saved to: {checkpoint_dir}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()