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

# Dual Path GPT Training Script with Multi LM Headers
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
import re

project_root = Path(__file__).parent.parent 
src_path = project_root  
if src_path.exists():
    sys.path.insert(0, str(src_path))
    print(f"📁 Added to Python path: {src_path}")
else:
    print(f"⚠️ src folder not found at: {src_path}")

# Import the dual path model with multi headers and dataset
from dual_path_multi_headers_model import DualPathMultiHeadersGPT2
from trainer import DualPathMultiHeadersTrainer
from utils.dataset import WikiTextDataset  # Keep the existing dataset

from torch.serialization import add_safe_globals
try:
    # allowlist GPT2Config so weights_only=True can unpickle safely
    from transformers.models.gpt2.configuration_gpt2 import GPT2Config
    add_safe_globals([GPT2Config])
except Exception:
    pass

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

# Global interrupt flag
training_interrupted = False

def signal_handler(signum, frame):
    global training_interrupted
    print("\nTraining interruption requested...")
    print("Will save checkpoint and exit after current batch...")
    training_interrupted = True

signal.signal(signal.SIGINT, signal_handler)

def _add_bool_arg(parser, name, default=True, help_text=""):
    # adds --name / --no-name 
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})

def get_args():
    parser = argparse.ArgumentParser(description="Dual Path GPT Training with Multi LM Headers")
    
    # Model and data
    parser.add_argument("--pretrained_model", type=str, default="gpt2",
                        help="HF model name or path (e.g., gpt2, gpt2-medium)")
    parser.add_argument("--split_at_layer", type=int, default=6,
                        help="Layer at which to split into left/right paths")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--max_samples", type=int, default=2000000,
                        help="Maximum number of training samples")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=0, help="0 = use epochs")
    
    # LM Header specific parameters
    parser.add_argument("--head_lr", type=float, default=5e-6, 
                        help="LR for trainable LM heads")
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
        default="gate_soft",
        choices=["gate_soft", "gate_hard", "left_only", "right_only", "max_prob"],
        help="Path routing for training."
    )

    parser.add_argument(
        "--eval_path_selection",
        type=str,
        default="gate_soft",
        choices=["gate_soft", "gate_hard", "left_only", "right_only", "max_prob"],
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

    # LM Header training phases
    parser.add_argument("--head_only_phase", action="store_true",
                        help="Train only LM headers, freeze everything else")
    parser.add_argument("--head_only_epochs", type=int, default=0,
                        help="Number of epochs to train only headers")
    
    # Token-aware training (optional)
    parser.add_argument("--use_token_aware", action="store_true",
                        help="Use token-aware weighting if token_weights.pt available")
    
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                        help="Max global norm for gradient clipping; set 0 to disable.")

    parser.add_argument("--gate_hidden", type=int, default=256,
                        help="Hidden size for gating MLPs and per-path head gates")
    
    parser.add_argument("--n_lm_perceptrons", type=int, default=4)
    _add_bool_arg(parser, "use_head_mixture", default=False, 
                  help_text="Enable differentiable mixture-of-heads.")
    parser.add_argument("--head_allocation", type=str, default=None,
                        help="JSON file with head allocation per path")

    parser.add_argument("--freeze_all_transformer", action="store_true",
                        help="Freeze all transformer blocks.")
    parser.add_argument("--freeze_split_gates", action="store_true",
                        help="Freeze path selection gate.")
    parser.add_argument("--head_topk", type=int, default=None,
                        help="Fast-k for LM-head gate (mix over top-k heads per path).")
    parser.add_argument("--head_gate_temp", type=float, default=1.0,
                        help="Temperature for LM-head gate softmax.")
    
    parser.add_argument(
        "--init_from_dual", type=str, default=None,
        help="Path to a dual checkpoint to load MODEL WEIGHTS ONLY (do not resume optimizer/scheduler)."
    )

    parser.add_argument("--head_lb_coef", type=float, default=0.0,
                        help="Per-path LM-head load-balancing (KL to uniform)")
    parser.add_argument("--head_entropy_coef", type=float, default=0.0,
                        help="Per-path LM-head entropy target (logK - H(p))")

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
    """Analyze which path is selected more often"""
    trainer.model.eval()
    
    left_selections = 0
    right_selections = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            input_ids = batch['input_ids'].to(trainer.device)
            attention_mask = batch['attention_mask'].to(trainer.device)
            
            # Get both paths
            left_logits, right_logits = trainer.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_both_paths=True
            )
            
            # Calculate which path would be selected
            left_probs = F.softmax(left_logits, dim=-1)
            right_probs = F.softmax(right_logits, dim=-1)
            
            left_max_probs, _ = left_probs.max(dim=-1)
            right_max_probs, _ = right_probs.max(dim=-1)
            
            # Count selections
            valid_mask = attention_mask.bool()
            left_wins = (left_max_probs > right_max_probs) & valid_mask
            right_wins = (right_max_probs > left_max_probs) & valid_mask
            
            left_selections += left_wins.sum().item()
            right_selections += right_wins.sum().item()
            total_tokens += valid_mask.sum().item()
    
    left_pct = (left_selections / total_tokens) * 100 if total_tokens > 0 else 0
    right_pct = (right_selections / total_tokens) * 100 if total_tokens > 0 else 0
    
    print(f"Path usage analysis (over {num_batches} batches):")
    print(f"  Left path selected: {left_pct:.1f}% ({left_selections:,} tokens)")
    print(f"  Right path selected: {right_pct:.1f}% ({right_selections:,} tokens)")
    print(f"  Total analyzed tokens: {total_tokens:,}")

def apply_freeze_schedule(trainer, step, freeze_schedule, applied_steps):
    if not freeze_schedule:
        return
    for s in sorted(int(k) for k in freeze_schedule.keys()):
        if s <= step and s not in applied_steps:
            trainer.model.set_path_freezing(freeze_schedule[str(s)])
            applied_steps.add(s)
            print(f"Applied freeze config @ step {step}: {freeze_schedule[str(s)]}")

def _lm_map(train_indices, n_heads=4, freeze_anchor0=True, as_str=True):
    """Create LM head freeze mapping"""
    m = {}
    for i in range(n_heads):
        # head-0 stays frozen as anchor by default
        is_frozen = (i == 0 and freeze_anchor0) or (i not in train_indices)
        key = str(i) if as_str else i
        m[key] = is_frozen
    return m

def log_lm_head_stats(trainer, batch, step, label="train", path_selection=None):
    """Log LM head usage statistics"""
    model = trainer.model
    sel = path_selection or getattr(trainer, "train_path_selection", None) or "gate_soft"
    model.eval()
    
    with torch.inference_mode():
        input_ids = batch["input_ids"].to(trainer.device)
        attention_mask = batch["attention_mask"].to(trainer.device)
        labels = batch.get("labels", input_ids).to(trainer.device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            path_selection=sel,
            return_head_indices=True
        )

        if not isinstance(out, dict) or out.get("head_top_idx") is None:
            print(f"[{label} @ step {step}] head-usage: (skipped - head mixture disabled)")
            model.train()
            return

        if out.get("gate") is None or "final_weights" not in out["gate"]:
            print(f"[{label} @ step {step}] head-usage: (skipped - no gates)")
            model.train()
            return

        hti = out["head_top_idx"]
        if not all(k in hti for k in ["left", "right"]):
            print(f"[{label} @ step {step}] head-usage: (skipped - missing head_top_idx keys)")
            model.train()
            return

        # Pick the same path per token the model used
        fw = out["gate"]["final_weights"]
        w = torch.stack([fw["left"], fw["right"]], dim=-1).squeeze(-2)  # [B,S,2]
        best_path = w.argmax(dim=-1)[..., :-1]  # [B,S-1]

        # Path token counts
        path_counts = best_path.flatten().bincount(minlength=2).tolist()
        print(f"[{label} @ step {step}] path-tokens Left={path_counts[0]} Right={path_counts[1]}")

        # Merge per-path head ids using chosen path
        names = ("left", "right")
        chosen_head = torch.full_like(best_path, -1, dtype=torch.long)
        for idx, name in enumerate(names):
            per_path = hti[name][..., :-1].to(torch.long)
            mask = (best_path == idx)
            chosen_head = torch.where(mask, per_path, chosen_head)

        # Compute usage/accuracy
        lp = out.get("log_probs")
        if lp is None:
            logits = out["logits"]
            lp = torch.log_softmax(logits, dim=-1)
        
        preds = lp[..., :-1, :].argmax(dim=-1)  # [B,S-1]
        gold  = labels[..., 1:]                 # [B,S-1]
        valid = attention_mask[..., 1:].bool()
        correct = (preds == gold) & valid

        n_heads = getattr(model.lm_head, "num_perceptrons", 4)
        sel_cnt = {h: 0 for h in range(n_heads)}
        cor_cnt = {h: 0 for h in range(n_heads)}
        tot = int(valid.sum().item())

        for h in range(n_heads):
            m = valid & (chosen_head == h)
            sel_cnt[h] = int(m.sum().item())
            cor_cnt[h] = int((m & correct).sum().item())

        usage = {h: (100.0 * sel_cnt[h] / tot if tot else 0.0) for h in range(n_heads)}
        acc   = {h: (100.0 * cor_cnt[h] / sel_cnt[h] if sel_cnt[h] else float("nan")) for h in range(n_heads)}

    usage_str = " ".join([f"H{h}:{usage[h]:4.1f}%" for h in range(n_heads)])
    acc_str   = " ".join([f"H{h}:{(acc[h] if sel_cnt[h] else ' n/a'):>5}" for h in range(n_heads)])
    print(f"[{label} @ step {step}] head-usage  {usage_str}")
    print(f"[{label} @ step {step}] head-acc    {acc_str}")
    model.train()

@torch.inference_mode()
def quick_preview_greedy(model, tokenizer, prompt_text, *, max_new_tokens=32, path_selection="gate_soft"):
    """Quick greedy generation for preview"""
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    attn = torch.ones_like(input_ids)

    for _ in range(max_new_tokens):
        out = model(input_ids=input_ids, attention_mask=attn, path_selection=path_selection)
        scores = out.get("log_probs", out["logits"])
        next_id = scores[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        attn = torch.cat([attn, attn.new_ones((attn.size(0), 1))], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

@torch.inference_mode()
def eval_ppl_safe(model, dataloader, *, path_selection="gate_soft"):
    """Safe perplexity evaluation"""
    device = next(model.parameters()).device
    total_logprob = 0.0
    total_tokens = 0
    total_correct = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention  = batch["attention_mask"].to(device)
        labels     = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention, path_selection=path_selection)
        logp = out.get("log_probs", torch.log_softmax(out["logits"], dim=-1))
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

def _extract_state_dict(raw):
    if isinstance(raw, dict):
        for k in ("model_state_dict", "state_dict", "ema_state_dict"):
            if k in raw and isinstance(raw[k], dict):
                return raw[k]
        if all(isinstance(v, torch.Tensor) for v in raw.values()):
            return raw
    return raw

def _maybe_transpose_copy(dst: torch.nn.Parameter, src: torch.Tensor):
    if dst.shape == src.shape:
        dst.copy_(src)
        return "as-is"
    if dst.shape == src.t().shape:
        dst.copy_(src.t())
        return "transposed"
    raise ValueError(f"Shape mismatch: dst {tuple(dst.shape)} vs src {tuple(src.shape)}")

def _target_shapes_from_model(model):
    # First perceptron defines target (V,E) for all heads
    W = model.lm_head.perceptrons[0].weight
    V, E = W.shape
    return (V, E), (E, V)  # direct or transposed

def _collect_weight_candidates(sd, target_shapes):
    """Return list of (key, tensor, score) for keys that look like LM-head weights."""
    keys = list(sd.keys())
    (V,E), (E,Vt) = target_shapes
    cands = []

    def _shape_ok(t):
        s = tuple(t.shape)
        return (s == (V,E)) or (s == (E,V))

    def _score(name):
        s = 0
        low = name.lower()
        if "lm_head" in low: s += 100
        if "perceptron" in low or "heads." in low: s += 30
        if ".weight" in low: s += 10
        if "left" in low: s += 20
        if "right" in low: s += 20
        # prefer non-embedding weights over wte
        if "wte" in low: s -= 50
        return s

    for k in keys:
        if not k.endswith(".weight"):
            continue
        t = sd[k]
        if not isinstance(t, torch.Tensor):
            continue
        if _shape_ok(t):
            cands.append((k, t, _score(k)))

    # If we found nothing, allow wte as a last-resort candidate
    if not cands:
        for pref in ("", "model.", "module.", "module.model."):
            k = f"{pref}transformer.wte.weight"
            if k in sd and isinstance(sd[k], torch.Tensor):
                t = sd[k]
                if tuple(t.shape) == (V,E):
                    cands.append((k, t, 1))  # very low score, but usable

    # Sort by score desc, then by name for determinism
    cands.sort(key=lambda x: (x[2], x[0]), reverse=True)
    return cands

def warmstart_lm_heads_from_dual_ckpt(model, ckpt_path, head_alloc, left_src=0, right_src=1, diversify_std=0.0):
    """
    Robust dual->multi warmstart:
      - Finds 0/1 (left/right) sources by name hints; otherwise picks best two candidates.
      - If only one candidate exists, uses it for both seeds.
      - Falls back to wte.weight; if nothing found, skips cloning gracefully.
    """
    raw = torch.load(ckpt_path, map_location="cpu")
    sd = _extract_state_dict(raw)
    if not isinstance(sd, dict):
        print(f"[warmstart] WARNING: {ckpt_path} has no usable state_dict; skip cloning.")
        return

    # Load rest with strict=False first (so shapes don't crash)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[warmstart] load_state_dict(strict=False): missing={len(missing)} unexpected={len(unexpected)}")

    target_shapes = _target_shapes_from_model(model)
    cands = _collect_weight_candidates(sd, target_shapes)

    if not cands:
        print("[warmstart] WARNING: no LM-head-like weights found (by shape). Skipping cloning.")
        return

    # Try to pick left/right by name
    left_key = next((k for k,_,_ in cands if "left" in k.lower()), None)
    right_key = next((k for k,_,_ in cands if "right" in k.lower()), None)

    # If not found, try indexed forms (…/0.weight, …/1.weight)
    if left_key is None:
        left_key = next((k for k,_,_ in cands if re.search(r"\.0\.weight$", k)), None)
    if right_key is None:
        right_key = next((k for k,_,_ in cands if re.search(r"\.1\.weight$", k)), None)

    # Otherwise take top-2 highest-scoring candidates
    if left_key is None and right_key is None:
        if len(cands) >= 2:
            left_key, right_key = cands[0][0], cands[1][0]
        else:
            left_key = cands[0][0]
            right_key = cands[0][0]  # single-head fallback

    # If only one of them is found, mirror it
    if left_key is None and right_key is not None:
        left_key = right_key
    if right_key is None and left_key is not None:
        right_key = left_key

    # Final guard
    if left_key is None or right_key is None:
        print("[warmstart] WARNING: could not determine left/right seeds. Skipping cloning.")
        return

    lW = sd[left_key]
    rW = sd[right_key]

    # Optional biases (match by sibling name)
    def _bias_of(k):
        bkey = k.replace(".weight", ".bias")
        return sd.get(bkey, None)

    lB = _bias_of(left_key)
    rB = _bias_of(right_key)

    left_ids  = list(head_alloc.get("left",  []))
    right_ids = list(head_alloc.get("right", []))

    print(f"[warmstart] using left_src='{left_key}', right_src='{right_key}'")
    print(f"[warmstart] cloning to left={left_ids}, right={right_ids}")

    def _copy_into(hid, srcW, srcB, tag):
        head = model.lm_head.perceptrons[hid]
        with torch.no_grad():
            howW = _maybe_transpose_copy(head.weight, srcW)
            if hasattr(head, "bias") and head.bias is not None and srcB is not None:
                try:
                    howB = _maybe_transpose_copy(head.bias, srcB)
                except ValueError:
                    # bias shape mismatch -> skip silently
                    howB = "skipped"
            else:
                howB = "n/a"
            if diversify_std and diversify_std > 0:
                head.weight.add_(torch.randn_like(head.weight) * diversify_std)
                if hasattr(head, "bias") and head.bias is not None:
                    head.bias.add_(torch.randn_like(head.bias) * (diversify_std * 0.1))
        print(f"[warmstart] {tag} -> head[{hid}]  (W {howW}, B {howB})")

    for hid in left_ids:
        _copy_into(hid, lW, lB, "left_src")

    for hid in right_ids:
        _copy_into(hid, rW, rB, "right_src")

    print("[warmstart] cloning complete.")


def main():
    args = get_args()

    # Repro
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Env/paths
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

    # Checkpoints dir
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (BASE_PATH / "dual_path_multi_headers_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running in {RUN_MODE} mode")
    print(f"Base path: {BASE_PATH}")
    print(f"Checkpoints: {checkpoint_dir}")

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Config
    config = GPT2Config.from_pretrained(args.pretrained_model)
    config.n_positions = args.max_length
    config.n_lm_perceptrons = args.n_lm_perceptrons

    # Head allocation (required when mixture enabled)
    head_alloc = None
    if args.use_head_mixture:
        if not args.head_allocation:
            raise FileNotFoundError(
                "You enabled --use_head_mixture but did not pass --head_allocation "
                "(e.g., head_allocation.json)."
            )
        with open(args.head_allocation, "r") as f:
            head_alloc = json.load(f)

    # Optional freeze schedule
    freeze_schedule = None
    if args.freeze_schedule:
        with open(args.freeze_schedule, 'r') as f:
            freeze_schedule = json.load(f)
        print(f"Loaded freeze schedule: {freeze_schedule}")

    # ---- Build model (shape must match flags before loading ckpt) ----
    print(f"Creating dual path model with multi LM headers:")
    print(f"  - Split at layer {args.split_at_layer}")

    model = DualPathMultiHeadersGPT2(
        config,
        pretrained_model=args.pretrained_model,
        split_at_layer=args.split_at_layer,
        head_allocation=head_alloc,
        gate_hidden=args.gate_hidden,
    )
    model.use_head_mixture = bool(args.use_head_mixture)
    model.head_gate_temp   = args.head_gate_temp
    model.head_fast_k      = args.head_topk if args.head_topk is not None and args.head_topk > 0 else None
    model.freeze_split_gates(args.freeze_split_gates)

    if args.freeze_all_transformer:
        model.backbone_is_frozen = True
        print("GLOBAL FREEZE: All transformer layers frozen")

    # Trainer
    trainer = DualPathMultiHeadersTrainer(
        model, tokenizer, device, checkpoint_dir,
        lb_coef=args.lb_coef,
        gold_aux_coef=args.gold_aux_coef,
        tether_coef=args.tether_coef,
        gate_temp=args.gate_temp,
        clip_grad=args.clip_grad_norm
    )
    trainer.head_lb_coef      = args.head_lb_coef
    trainer.head_entropy_coef = args.head_entropy_coef

    # ---- Resume: load MODEL first (safe), then build optimizer/scheduler ----
    ck = None
    start_epoch = 0
    if args.resume:
        try:
            # Model-only resume: do not restore optimizer/scheduler for this phase
            ck = trainer.load_checkpoint(
                args.resume,
                optimizer=None,
                scheduler=None,
                load_optimizer=False,
                load_scheduler=False,
                strict_model=False,
                drop_mismatched_head_counters=True,
            )
            if ck:
                start_epoch = ck.get("epoch", 0)
                if "global_step" in ck:
                    trainer.global_step = int(ck["global_step"])
                # Adopt head_allocation from ckpt if not provided via CLI
                if "head_allocation" in ck["raw"] and getattr(model, "head_allocation", None) is None:
                    model.head_allocation = ck["raw"]["head_allocation"]
                    print(f"[ckpt] head_allocation restored from ckpt: {model.head_allocation}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting fresh (model)")

    # ---- Optimizer (fresh param groups for THIS run) ----
    try:
        optimizer = trainer.create_optimizer(
            lr=args.lr, head_lr=args.head_lr, gate_lr=args.gate_lr,
            weight_decay=args.weight_decay, foreach=False, fused=False
        )
    except TypeError:
        optimizer = trainer.create_optimizer(lr=args.lr, weight_decay=args.weight_decay)

    # Datasets / loaders
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
    collate_fn = create_collate_fn(tokenizer)
    loader_config = {'num_workers': 2, 'pin_memory': True} if RUN_MODE == "colab" else {'num_workers': 0, 'pin_memory': torch.cuda.is_available()}

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn, **loader_config)
    val_dataloader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, **loader_config)

    print(f"Training batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")

    # ---- Scheduler ----
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

    # Warm-start from older dual-path ONLY if not resuming a full ckpt
    if not args.resume and args.init_from_dual and (model.head_allocation is not None):
        warmstart_lm_heads_from_dual_ckpt(
            model,
            ckpt_path=args.init_from_dual,
            head_alloc=model.head_allocation,
            left_src=0,
            right_src=1,
            diversify_std=0.0,
        )

    # Allocation sanity
    if model.head_allocation:
        all_ids = [*(model.head_allocation.get("left", [])), *(model.head_allocation.get("right", []))]
        if all_ids:
            max_id = max(all_ids)
            if max_id >= args.n_lm_perceptrons:
                raise ValueError(f"--n_lm_perceptrons must be > {max_id} (got {args.n_lm_perceptrons})")

   
    # Training loop
    print("\nStarting training...")
    global training_interrupted

    total_train_loss = 0.0
    num_train_steps = 0

    if not args.resume:
        trainer.save_checkpoint(0, optimizer, 0)
    
    acc_steps = args.grad_accum_steps
    trainer.zero_grad(optimizer)

    current_epoch = start_epoch
    training_interrupted = False
    total_step = 0
    applied_steps = set()

    try:
        for epoch in range(start_epoch, args.epochs):
            current_epoch = epoch
            epoch_start_time = time.time()

            # Head-only training phase
            if args.head_only_epochs > 0 and epoch < args.head_only_epochs:
                print(f"\nEpoch {epoch+1}: HEAD-ONLY PHASE")
                trainer.model.set_path_freezing({
                    "shared": True,
                    "left_path": True,
                    "right_path": True,
                    "lm_headers": _lm_map(train_indices=range(1, args.n_lm_perceptrons), 
                                        n_heads=args.n_lm_perceptrons, freeze_anchor0=True, as_str=True),
                })
            elif args.head_only_epochs > 0 and epoch == args.head_only_epochs:
                print(f"\nEpoch {epoch+1}: SWITCHING TO FULL TRAINING")
                trainer.model.set_path_freezing({
                    "shared": False,
                    "left_path": False,
                    "right_path": False,
                    "lm_headers": _lm_map(train_indices=range(1, args.n_lm_perceptrons), 
                                        n_heads=args.n_lm_perceptrons, freeze_anchor0=True, as_str=True),
                })
            else:
                print(f"\nEpoch {epoch+1}: FULL TRAINING PHASE")

            for i, batch in enumerate(train_dataloader):
                if args.max_steps and total_step >= args.max_steps:
                    print(f"Reached max_steps={args.max_steps}, stopping training.")
                    training_interrupted = True
                    break

                # Apply freeze schedule if provided
                apply_freeze_schedule(trainer, total_step, freeze_schedule, applied_steps)

                metrics = trainer.backward_only(
                    batch,
                    path_selection=args.train_path_selection,   
                    loss_scale=1.0 / acc_steps
                )
                
                total_train_loss += metrics['loss']
                num_train_steps += 1

                # Optimizer step every acc_steps micro-batches
                if ((i + 1) % acc_steps) == 0:
                    trainer.optimizer_step(optimizer)
                    scheduler.step()
                    total_step += 1
                    
                    if training_interrupted:
                        break

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
                        if 'left_pct' in metrics:
                            log_str += f" left%={metrics['left_pct']:.3f}"
                        if 'right_pct' in metrics:
                            log_str += f" right%={metrics['right_pct']:.3f}"
                        if 'gate_entropy' in metrics:
                            log_str += f" gate_H={metrics['gate_entropy']:.3f}"
                        
                        print(log_str)

                        # Add LM head stats
                        lm_stats = trainer.model.lm_head.get_perceptron_stats()
                        if lm_stats['total_usage'] > 0:
                            usage_str = " | ".join(f"h{i}:{pct:.1f}%" for i, pct in enumerate(lm_stats['usage_percentages']))
                            print(f" | LM_heads: {usage_str}")

                    # Lightweight preview generation
                    if args.eval_every and (total_step % args.eval_every == 0):
                        prev_mode = trainer.model.training
                        trainer.model.eval()
                        try:
                            ctx_ids = batch["input_ids"][0][:32].detach().cpu()
                            prompt_text = tokenizer.decode(ctx_ids, skip_special_tokens=True)

                            hard_text = quick_preview_greedy(trainer.model, tokenizer, prompt_text,
                                                            max_new_tokens=32, path_selection="gate_hard")
                            soft_text = quick_preview_greedy(trainer.model, tokenizer, prompt_text,
                                                            max_new_tokens=32, path_selection="gate_soft")

                            print(f"\n🎯 Prompt: '{prompt_text}'")
                            print(f"🎯 Generated (gate_hard): '{hard_text[len(prompt_text):].strip()}'")
                            print(f"🎯 Generated (gate_soft): '{soft_text[len(prompt_text):].strip()}'")
                        except Exception as e:
                            print(f"⚠️ Sample generation failed: {e}")

                        # Regular eval
                        val = trainer.evaluate(val_dataloader, path_selection=args.eval_path_selection)
                        eval_str = (
                            f"[eval @ step {total_step}] "
                            f"loss={val['loss']:.4f} PPL={val['perplexity']:.3f} "
                            f"acc={val['accuracy']:.3f}"
                        )

                        # Safe PPL (ground truth)
                        ppl_safe, acc_safe = eval_ppl_safe(trainer.model, val_dataloader,
                                                        path_selection=args.eval_path_selection)
                        print(f"[eval-safe @ step {total_step}] PPL={ppl_safe:.3f} acc={acc_safe:.3f}")

                        # Gate stats
                        if 'left_pct' in val:
                            eval_str += f" left%={val['left_pct']:.3f}"
                        if 'right_pct' in val:
                            eval_str += f" right%={val['right_pct']:.3f}"
                        if 'gate_entropy' in val:
                            eval_str += f" gate_H={val['gate_entropy']:.3f}"
                        print(eval_str)

                        # Restore training mode
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
            if 'left_pct' in val:
                epoch_str += f" | left%={val['left_pct']:.3f}"
            if 'right_pct' in val:
                epoch_str += f" | right%={val['right_pct']:.3f}"
            if 'gate_entropy' in val:
                epoch_str += f" | gate_H={val['gate_entropy']:.3f}"
            
            print(epoch_str)

            # Head freezing schedule
            if total_step < args.freeze_head0_steps:
                trainer.model.set_path_freezing({"lm_headers": {"0": True}})
            elif total_step == args.freeze_head0_steps:
                trainer.model.set_path_freezing({"lm_headers": {"0": False}})
                print(f"Unfroze head-0 at step {total_step}")
            
            # Print LM head statistics
            if lm_stats['total_usage'] > 0:
                usage_str = " | ".join(f"head_{i}: {pct:.1f}%" for i, pct in enumerate(lm_stats['usage_percentages']))
                print(f"  LM Head Usage: {usage_str}")

            # Detailed head stats
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
            selection_strategies = ["left_only", "right_only", "gate_soft", "max_prob"]
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
                baseline_ppl = results.get("left_only", {}).get("perplexity", float('inf'))
                best_ppl = min(res.get("perplexity", float('inf')) for res in results.values())
                
                print(f"\nSummary:")
                print(f"  Baseline (left_only): {baseline_ppl:.2f} PPL")
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
                for path_sel in ["left_only", "right_only", "gate_soft", "max_prob"]:
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