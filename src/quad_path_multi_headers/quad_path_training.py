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

# Quad Path GPT Training Script
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
from quad_path_model import HierarchicalQuadPathGPT2, QuadPathTrainer
from utils.dataset import WikiTextDataset  # Keep the existing dataset

from torch.serialization import add_safe_globals
try:
    # allowlist GPT2Config so weights_only=True can unpickle safely
    from transformers.models.gpt2.configuration_gpt2 import GPT2Config
    add_safe_globals([GPT2Config])
except Exception:
    pass

# Global interrupt flag
training_interrupted = False

def signal_handler(signum, frame):
    global training_interrupted
    print("\nTraining interruption requested...")
    print("Will save checkpoint and exit after current batch...")
    training_interrupted = True

signal.signal(signal.SIGINT, signal_handler)

def get_args():
    parser = argparse.ArgumentParser(description="Quad Path GPT Training")
    
    # Model and data
    parser.add_argument("--pretrained_model", type=str, default="gpt2",
                        help="HF model name or path (e.g., gpt2, gpt2-medium)")
    parser.add_argument("--split_at_layer_1", type=int, default=6,
                        help="First split layer (dual path split)")
    parser.add_argument("--split_at_layer_2", type=int, default=9,
                        help="Second split layer (quad path split)")
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

    parser.add_argument(
        "--dual_path_checkpoint",
        type=str,
        default=None,
        help="(Optional) Path to a trained dual-path checkpoint; expand into quad-path init"
    )

    # hard-window: DISABLED by default (backward compatible)
    parser.add_argument("--hard_from_step", type=int, default=-1)
    parser.add_argument("--hard_to_step",   type=int, default=-1)
    parser.add_argument("--hard_from_frac", type=float, default=-1.0)
    parser.add_argument("--hard_to_frac",   type=float, default=-1.0)

    # gate temp schedule points: empty = DISABLED (use args.gate_temp like Phase A)
    parser.add_argument("--gate_temp_points", type=str, default="")

    # consistency loss: DISABLED by default for Phase A compatibility
    parser.add_argument("--consistency_lambda", type=float, default=0.0)

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
    """Analyze which paths are selected most often"""
    trainer.model.eval()
    
    path_selections = {"left_left": 0, "left_right": 0, "right_left": 0, "right_right": 0}
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            input_ids = batch['input_ids'].to(trainer.device)
            attention_mask = batch['attention_mask'].to(trainer.device)
            
            # Get all paths
            all_path_logits = trainer.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_all_paths=True
            )
            
            # Calculate which path would be selected at each position
            path_probs = {}
            path_max_probs = {}
            for path_name, logits in all_path_logits.items():
                probs = F.softmax(logits, dim=-1)
                max_probs, _ = probs.max(dim=-1)
                path_probs[path_name] = probs
                path_max_probs[path_name] = max_probs
            
            # Find best path for each position
            valid_mask = attention_mask.bool()
            stacked_max = torch.stack(list(path_max_probs.values()), dim=-1)  # [B,S,4]
            best_path_idx = torch.argmax(stacked_max, dim=-1)  # [B,S]
            
            # Count selections
            path_names = list(path_selections.keys())
            for i, path_name in enumerate(path_names):
                selections = ((best_path_idx == i) & valid_mask).sum().item()
                path_selections[path_name] += selections
            
            total_tokens += valid_mask.sum().item()
    
    print(f"Path usage analysis (over {num_batches} batches):")
    for path_name, count in path_selections.items():
        pct = (count / total_tokens) * 100 if total_tokens > 0 else 0
        print(f"  {path_name}: {pct:.1f}% ({count:,} tokens)")
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
        checkpoint_dir = BASE_PATH / "quad_path_checkpoints"
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
    
    # Load freeze schedule if provided
    freeze_schedule = None
    if args.freeze_schedule:
        with open(args.freeze_schedule, 'r') as f:
            freeze_schedule = json.load(f)
        print(f"Loaded freeze schedule: {freeze_schedule}")
    
    # Create model
    print(f"Creating quad path model with splits at layers {args.split_at_layer_1} and {args.split_at_layer_2}")
    
    if args.dual_path_checkpoint:
        print(f"Will initialize from dual path checkpoint: {args.dual_path_checkpoint}")
    
    model = HierarchicalQuadPathGPT2(
        config=config,
        pretrained_model=args.pretrained_model,
        split_at_layer_1=args.split_at_layer_1,
        split_at_layer_2=args.split_at_layer_2,
        dual_path_checkpoint=args.dual_path_checkpoint
    )
    
    # Create trainer
    trainer = QuadPathTrainer(
        model, tokenizer, device, checkpoint_dir,
        lb_coef=args.lb_coef, gold_aux_coef=args.gold_aux_coef, tether_coef=args.tether_coef,
        gate_temp=args.gate_temp, clip_grad=args.max_grad_norm
    )
    trainer.consistency_lambda = args.consistency_lambda

    optimizer = trainer.create_optimizer(lr=args.lr, weight_decay=args.weight_decay)
    
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

                # Use trainer’s live mode in backward_only
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

                    # Parse once
                    pts = getattr(args, "_gate_temp_points_parsed", None)
                    if pts is None:
                        def _parse_gate_temp_points(spec):
                            if not spec: return None
                            pts = []
                            for tok in spec.split(","):
                                if not tok.strip(): continue
                                f, t = tok.split(":")
                                pts.append((float(f), float(t)))
                            return sorted(pts)
                        args._gate_temp_points_parsed = _parse_gate_temp_points(args.gate_temp_points)
                        pts = args._gate_temp_points_parsed

                    progress = total_step / float(total_optimizer_steps + 1e-8)

                    # Default: keep Phase A fixed temp
                    if pts:
                        # piecewise linear interpolation
                        def _interp(points, x, default):
                            if x <= points[0][0]: return points[0][1]
                            for i in range(1, len(points)):
                                x0,y0 = points[i-1]; x1,y1 = points[i]
                                if x <= x1:
                                    a = (x - x0) / max(1e-12, (x1 - x0))
                                    return y0 + a * (y1 - y0)
                            return points[-1][1]
                        trainer.gate_temp = _interp(pts, progress, default=args.gate_temp)
                    else:
                        trainer.gate_temp = args.gate_temp

                    # ---- gate temp: from CLI piecewise points or fallback 1.2->0.7->0.5 ----
                    pts = getattr(args, "_gate_temp_points_parsed", None)
                    if pts is None:
                        pts = _parse_gate_temp_points(args.gate_temp_points)
                        args._gate_temp_points_parsed = pts

                    if pts:
                        trainer.gate_temp = _interp_piecewise(pts, progress, default=0.5)
                    else:
                        # fallback schedule
                        if progress < 0.30:
                            trainer.gate_temp = 1.2 - 0.5 * (progress / 0.30)   # 1.2 -> 0.7
                        elif progress < 0.60:
                            trainer.gate_temp = 0.7 - 0.2 * ((progress - 0.30) / 0.30)  # 0.7 -> 0.5
                        else:
                            trainer.gate_temp = 0.5

                    # ---- compute hard window once; cache it on args ----
                    if not hasattr(args, "_hard_window"):
                        args._hard_window = _compute_hard_window(args, total_optimizer_steps, freeze_schedule)

                    hard_start, hard_end = args._hard_window

                    # ---- select routing mode for NEXT step (or apply before backward_only if you prefer immediate effect) ----
                    if hard_start >= 0 and hard_start <= total_step < hard_end:
                        trainer.train_path_selection = "gate_hard"
                    else:
                        trainer.train_path_selection = args.train_path_selection

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

                    # Periodic evaluation
                    if args.eval_every and (total_step % args.eval_every == 0):
                        val = trainer.evaluate(val_dataloader, path_selection=args.eval_path_selection)
                        eval_str = (
                            f"[eval @ step {total_step}] "
                            f"loss={val['loss']:.4f} PPL={val['perplexity']:.3f} "
                            f"acc={val['accuracy']:.3f}"
                        )
                        
                        # Add gate stats if available
                        if 'gate1_right_pct' in val:
                            eval_str += f" g1_right%={val['gate1_right_pct']:.3f}"
                        if 'left_left_usage' in val:
                            eval_str += (f" usage: LL={val.get('left_left_usage', 0):.3f} "
                                       f"LR={val.get('left_right_usage', 0):.3f} "
                                       f"RL={val.get('right_left_usage', 0):.3f} "
                                       f"RR={val.get('right_right_usage', 0):.3f}")
                        
                        print(eval_str)

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

            # End-of-epoch evaluation
            val = trainer.evaluate(val_dataloader, path_selection=args.eval_path_selection)
            epoch_str = (
                f"Epoch {current_epoch+1}: "
                f"val loss={val['loss']:.4f} | PPL={val['perplexity']:.3f} | "
                f"acc={val['accuracy']:.3f}"
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

            if training_interrupted:
                break

        print(f"Done. Optimizer steps taken: {total_step} (expected ~{total_optimizer_steps})")

        # Final evaluation and analysis
        if not training_interrupted:
            print("\nTraining completed!")
            
            # Final evaluation on all paths and selection strategies
            print("\nFinal evaluation with different path selection strategies:")
            
            # Test individual paths first
            print("\nIndividual path performance:")
            for path_name in ["left_left", "left_right", "right_left", "right_right"]:
                # We need to modify the model to support individual path selection
                # For now, let's evaluate the main strategies
                pass
            
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