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

# Dual Path GPT Training Script
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

# Import the dual path model and dataset (assumes they're in the same directory)
from dual_path_model import IndependentDualPathGPT2, DualPathTrainer
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
    parser = argparse.ArgumentParser(description="Dual Path GPT Training")
    
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
        default="gate_soft",  # ← use the learnable mixture
        choices=["gate_soft", "gate_hard", "left_only", "right_only", "max_prob", "soft_weighted"],
        help="Path selection during training",
    )
    parser.add_argument(
        "--eval_path_selection",
        type=str,
        default="gate_soft",  # ← evaluate the same mixture you optimize
        choices=["gate_soft", "gate_hard", "left_only", "right_only", "max_prob", "soft_weighted"],
        help="Path selection during evaluation",
    )


    #Phase B
    parser.add_argument("--phase_b_start", type=int, default=0,
                help="Step to start Phase B gate hardening")
    parser.add_argument("--phase_b_ramp1", type=int, default=500,
                    help="Steps to ramp λ 0→0.2")
    parser.add_argument("--phase_b_ramp2", type=int, default=1500,
                    help="Steps to ramp λ 0.2→0.5")
    parser.add_argument("--phase_b_max", type=float, default=0.5,
                    help="Maximum hard_blend_lambda value")
    parser.add_argument("--gate_temp_target", type=float, default=1.0,
                help="Target gate temperature during Phase B")

    parser.add_argument("--gate_margin_coef", type=float, default=0.0,
                        help="Weight for gate margin (confidence) loss; 0 disables.")
    parser.add_argument("--gate_margin", type=float, default=0.25,
                        help="Target margin for (top1 - top2) on the 2-way gate logits.")


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
        checkpoint_dir = BASE_PATH / "dual_path_checkpoints"
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
    
    # Create model
    print(f"Creating dual path model with split at layer {args.split_at_layer}")
    model = IndependentDualPathGPT2(
        config=config,
        pretrained_model=args.pretrained_model,
        split_at_layer=args.split_at_layer
    )

    model.gate_margin_coef = args.gate_margin_coef
    model.gate_margin      = args.gate_margin

    # Create trainer
    trainer = DualPathTrainer(
        model,
        tokenizer,           # ← use the actual tokenizer var
        device,
        checkpoint_dir,      # ← use the actual checkpoint_dir var
        lb_coef=1e-3,
        gold_aux_coef=1e-3,
        tether_coef=5e-4,
        gate_temp=1.2,
    )
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
        max_samples=1000  # Smaller validation set
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
    total_optimizer_steps = math.ceil(batches_per_epoch * args.epochs / args.grad_accum_steps)
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
                ckpt = torch.load(args.resume, map_location=device, weights_only=False)  # trusted file
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
                    state = ckpt  # direct state_dict case
            if state is None:
                raise RuntimeError("Checkpoint missing model state dict")

            missing, unexpected = model.load_state_dict(state, strict=False)
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

    total_optimizer_steps = math.ceil(len(train_dataloader) * args.epochs / acc_steps)  # for scheduler sanity

    current_epoch = 0
    training_interrupted = False
    total_step = 0           

    try:

        for epoch in range(args.epochs):
            for i, batch in enumerate(train_dataloader):

                if args.max_steps and total_step >= args.max_steps:
                    print(f"Reached max_steps={args.max_steps}, stopping training.")
                    training_interrupted = True
                    break

                metrics = trainer.backward_only(
                    batch,
                    path_selection=args.train_path_selection,
                    loss_scale=1.0 / acc_steps
                )
                total_train_loss += metrics['loss']
                num_train_steps  += 1

                # do an optimizer step every acc_steps micro-batches
                if ((i + 1) % acc_steps) == 0:
                    trainer.optimizer_step(optimizer)
                    scheduler.step()
                    total_step += 1      

                    s = max(0, total_step - args.phase_b_start)
                    if s <= args.phase_b_ramp1:
                        lam = 0.20 * (s / args.phase_b_ramp1)
                    elif s <= args.phase_b_ramp2:
                        lam = 0.20 + (args.phase_b_max - 0.20) * ((s - args.phase_b_ramp1) /
                                                                (args.phase_b_ramp2 - args.phase_b_ramp1))
                    else:
                        lam = args.phase_b_max

                    trainer.hard_blend_lambda = float(lam)
                    trainer.gate_temp = args.gate_temp_target

                    if training_interrupted:
                        break

                    # metrics from backward_only()
                    usage_kl = float(metrics.get("usage_kl", 0.0))
                    right_pct = float(metrics.get("right_pct", 0.5))

                    # thresholds
                    IMBALANCE = (usage_kl > 0.05) or (right_pct < 0.30) or (right_pct > 0.70)

                    # simple hysteresis: loosen the condition to resume normal ramp
                    RECOVERED = (usage_kl < 0.02) and (0.40 <= right_pct <= 0.60)

                    # persistent cap across steps
                    if not hasattr(trainer, "_lam_cap"):
                        trainer._lam_cap = args.phase_b_max

                    # if imbalanced, cap λ for a while and slightly raise temp
                    if IMBALANCE:
                        trainer._lam_cap = min(trainer._lam_cap, 0.30)   # cap harder blending
                        trainer.gate_temp = max(args.gate_temp_target, 1.05)
                    elif RECOVERED:
                        trainer._lam_cap = args.phase_b_max              # restore normal cap
                        trainer.gate_temp = args.gate_temp_target

                    # apply cap after schedule
                    trainer.hard_blend_lambda = float(min(lam, trainer._lam_cap))

                    # optionally strengthen load-balance when imbalanced
                    if IMBALANCE and hasattr(trainer, "lb_coef"):
                        trainer.lb_coef = max(trainer.lb_coef, 1e-3)     # temporary bump

                    # ---- logging on optimizer steps ----
                    if args.log_every and (total_step % args.log_every == 0):
                        print(
                            f"[step {total_step}/{total_optimizer_steps}] "
                            f"loss={metrics['loss']:.4f} ce={metrics['ce']:.4f} "
                            f"acc={metrics['accuracy']:.3f} "
                            f"gold_aux={metrics['gold_aux']:.4f} "
                            f"lbKL={metrics['lbkl']:.4f} "
                            f"gateH={metrics['gate_entropy']:.3f} "
                            f"usageKL={metrics['usage_kl']:.4f} "
                            f"right%={metrics['right_pct']:.3f} "
                            f"{metrics['mode']}"
                        )

                    # ---- periodic eval on optimizer steps ----
                    if args.eval_every and (total_step % args.eval_every == 0):
                        val = trainer.evaluate(val_dataloader, path_selection=args.eval_path_selection)
                        print(
                            f"[eval @ step {total_step}] "
                            f"loss={val['loss']:.4f} PPL={val['perplexity']:.3f} acc={val['accuracy']:.3f} "
                            f"right%={val.get('right_pct', float('nan')):.3f} "
                            f"gateH={val.get('gate_entropy', float('nan')):.3f} "
                            f"usageKL={val.get('usage_kl', float('nan')):.4f} "
                            f"{val.get('mode','')}"
                        )

                        prev_mode = trainer.model.training
                        trainer.model.eval()
                        try:
                            ctx_ids = batch["input_ids"][0][:32].detach().cpu()
                            prompt_text = tokenizer.decode(ctx_ids, skip_special_tokens=True)
                            hard_text = trainer.generate_sample(
                                        prompt_text, 
                                        max_length=30, 
                                        path_selection="gate_hard",
                                        temperature=0.8,
                                        do_sample=True
                                    )
                            print(f"\n🎯 Prompt: '{prompt_text}'")
                            print(f"🎯 Generated (gate_hard): '{hard_text[len(prompt_text):].strip()}'")
                        except Exception as e:
                            print(f"⚠️ Sample generation failed: {e}")
                        # restore original mode once
                        if prev_mode:
                            trainer.model.train()

                    # ---- periodic checkpoint on optimizer steps ----
                    if args.save_every and (total_step % args.save_every == 0):
                        trainer.save_checkpoint(
                            current_epoch, optimizer, total_train_loss / max(num_train_steps, 1)
                        )

            # flush remainder micro-batches (if the epoch length not divisible by acc_steps)
            if not training_interrupted:
                remainder = (len(train_dataloader) % acc_steps)
                if remainder != 0:
                    trainer.optimizer_step(optimizer)
                    scheduler.step()
                    total_step += 1

            # end-of-epoch eval (optional if you also do eval_every)
            val = trainer.evaluate(val_dataloader, path_selection=args.eval_path_selection)
            print(
                f"Epoch {current_epoch+1}: "
                f"val loss={val['loss']:.4f} | PPL={val['perplexity']:.3f} | acc={val['accuracy']:.3f} | "
                f"right%={val.get('right_pct', float('nan')):.3f} | "
                f"gateH={val.get('gate_entropy', float('nan')):.3f} | "
                f"usageKL={val.get('usage_kl', float('nan')):.4f} | "
                f"{val.get('mode','')}"
            )

            if training_interrupted:
                break
            current_epoch += 1

        print(f"Done. Optimizer steps taken: {total_step} (expected ~{total_optimizer_steps})")

        # Final steps
        if not training_interrupted:
            print("\nTraining completed!")
            
            # Final evaluation on both path selection strategies
            print("\nFinal evaluation with different path selection strategies:")
            for path_sel in ["left_only", "right_only", "max_prob", "soft_weighted", "gate_soft"]:
                res = trainer.evaluate(val_dataloader, path_selection=path_sel)
                print(f"  {path_sel}: Loss={res['loss']:.4f}, Acc={res['accuracy']:.2%}, PPL={res['perplexity']:.2f}")

            ppl_left  = trainer.evaluate(val_dataloader, "left_only")["perplexity"]
            ppl_right = trainer.evaluate(val_dataloader, "right_only")["perplexity"]
            ppl_mix   = trainer.evaluate(val_dataloader, "gate_soft")["perplexity"]
            print(f"\nSummary PPL -> Left: {ppl_left:.2f} | Right: {ppl_right:.2f} | Mixture(gate_soft): {ppl_mix:.2f}")


            # Final path usage analysis
            print("\nFinal path usage analysis:")
            analyze_path_usage(trainer, val_dataloader, num_batches=20)
            
            # Save final model
            final_path = trainer.save_checkpoint(
                current_epoch, optimizer, total_train_loss / max(num_train_steps, 1), is_final=True
            )

            print(f"\nFinal model saved to: {final_path}")
            
            # Final sample generation
            print("\nFinal generation samples:")
            test_prompts = [
                "The future of machine learning is",
                "In a world where technology",
                "The most fascinating aspect of"
            ]
            
            for prompt in test_prompts:
                print(f"\nPrompt: '{prompt}'")
                for path_sel in ["left_only", "right_only", "max_prob", "gate_soft"]:
                    generated = trainer.generate_sample(
                        prompt,
                        max_length=40,
                        path_selection=path_sel,
                        temperature=0.7
                    )
                    print(f"  {path_sel}: {generated[len(prompt):].strip()}")
        
        else:
            print("\nTraining interrupted. Saving final checkpoint...")
            trainer.save_checkpoint(
                current_epoch, optimizer, total_train_loss / max(num_train_steps, 1)
            )
 
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
    