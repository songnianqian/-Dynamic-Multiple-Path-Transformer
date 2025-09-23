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

# Dual Path Multi Headers Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from pathlib import Path
import json
import time
from tqdm import tqdm
import math


class DualPathMultiHeadersTrainer:
    """Trainer for dual path model with multiple LM headers"""
    
    def __init__(self, model, tokenizer, device, checkpoint_dir,
                 lb_coef=1e-3, gold_aux_coef=1e-3, tether_coef=5e-4,
                 gate_temp=1.2, clip_grad=1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss coefficients
        self.lb_coef = lb_coef
        self.gold_aux_coef = gold_aux_coef
        self.tether_coef = tether_coef
        self.gate_temp = gate_temp
        self.clip_grad = clip_grad
        
        # Training state
        self.global_step = 0
        self.train_path_selection = "gate_soft"
        
        # Additional loss coefficients for compatibility
        self.consistency_lambda = 0.0
        self.kd_coef = 0.0
        self.kd_teacher_selection = "gate_soft"
        self.kd_on = "never"
        self.kd_margin = 0.10

        self.head_lb_coef      = 0
        self.head_entropy_coef = 0
        
        # Move model to device
        self.model.to(device)
        
        print(f"DualPath Multi Headers Trainer initialized")
        print(f"  Device: {device}")
        print(f"  LB coef: {lb_coef}, Gold aux: {gold_aux_coef}, Tether: {tether_coef}")
        print(f"  Gate temp: {gate_temp}, Clip grad: {clip_grad}")

    def create_optimizer(self, lr=2e-5, head_lr=5e-6, gate_lr=1e-5, weight_decay=0.01, **kwargs):
        """Create optimizer with separate learning rates for different components"""
        
        # Group parameters by type
        param_groups = []
        
        # 1. Main model parameters (transformer layers, etc.)
        main_params = []
        for name, param in self.model.named_parameters():
            if (param.requires_grad and 
                not name.startswith("lm_head.perceptrons") and 
                not name.startswith("_heads_gate_by_path") and
                not name.startswith("gate.")):
                main_params.append(param)
        
        if main_params:
            param_groups.append({
                "params": main_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "main"
            })

        
        # 2. LM heads (excluding head 0 which should be frozen)
        lm_head_params = []
        for i, head in enumerate(self.model.lm_head.perceptrons):
            for param in head.parameters():
                if param.requires_grad:
                    lm_head_params.append(param)
        
        if lm_head_params:
            param_groups.append({
                "params": lm_head_params,
                "lr": head_lr,
                "weight_decay": weight_decay * 0.1,  # Lower weight decay for heads
                "name": "lm_heads"
            })
        
        # 3. Head gates (per-path mixture gates)
        gate_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and name.startswith("_heads_gate_by_path"):
                gate_params.append(param)
        
        if gate_params:
            param_groups.append({
                "params": gate_params,
                "lr": gate_lr,
                "weight_decay": weight_decay * 0.1,
                "name": "gates"
            })
        
        # 4. Split gate (path selection gate)
        split_gate_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and name.startswith("gate."):
                split_gate_params.append(param)
        
        if split_gate_params:
            param_groups.append({
                "params": split_gate_params,
                "lr": gate_lr,
                "weight_decay": weight_decay * 0.1,
                "name": "split_gate"
            })
        
        print(f"Created optimizer with {len(param_groups)} parameter groups:")
        for group in param_groups:
            print(f"  {group['name']}: {len(group['params'])} params, lr={group['lr']:.2e}")
        
        return AdamW(param_groups, **kwargs)

    def zero_grad(self, optimizer):
        """Zero gradients"""
        optimizer.zero_grad()

    def optimizer_step(self, optimizer):
        """Take optimizer step with gradient clipping"""
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        optimizer.step()
        optimizer.zero_grad()

    def _gate_is_trainable(self) -> bool:
            """
            Returns True if the split gate at the path split has any trainable params.
            When you call model.freeze_split_gates(True), all gate params have
            requires_grad=False and this will return False.
            """
            try:
                return any(p.requires_grad for p in self.model.gate.parameters())
            except AttributeError:
                return False

    def backward_only(self, batch, path_selection="gate_soft", loss_scale=1.0):
        """Backward pass only (for gradient accumulation)"""
        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels         = batch["labels"].to(self.device)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            path_selection=path_selection,
        )

        # Main CE loss
        ce_loss    = outputs["loss"]
        total_loss = ce_loss

        # Additional losses (init)
        lb_loss        = 0.0  # path-level load-balance (left vs right)
        gold_aux_loss  = 0.0  # auxiliary "pick better path" term
        tether_loss    = 0.0  # KL tether between right/left path distributions
        head_lb_reg    = 0.0  # per-path LM-head load-balance
        head_ent_reg   = 0.0  # per-path LM-head entropy target

        # === Path load-balancing (left/right). Skip if gate is frozen or coef==0.
        if (getattr(self, "lb_coef", 0.0) > 0.0
            and self._gate_is_trainable()
            and outputs.get("gate") is not None):
            gate_info = outputs["gate"]
            if "final_weights" in gate_info:
                weights = gate_info["final_weights"]
                if "left" in weights and "right" in weights:
                    w_left  = weights["left"].squeeze(-1)   # [B,S]
                    w_right = weights["right"].squeeze(-1)  # [B,S]
                    valid_mask = attention_mask.bool()
                    if valid_mask.any():
                        vmf = valid_mask.float()
                        denom = vmf.sum().clamp_min(1.0)
                        left_avg  = (w_left  * vmf).sum() / denom
                        right_avg = (w_right * vmf).sum() / denom
                        target = w_left.new_tensor(0.5)
                        lb_loss = (left_avg - target).pow(2) + (right_avg - target).pow(2)
                        total_loss = total_loss + self.lb_coef * lb_loss

        # === Gold auxiliary (prefer lower-loss path). Skip if gate is frozen or coef==0.
        if (getattr(self, "gold_aux_coef", 0.0) > 0.0
            and self._gate_is_trainable()
            and outputs.get("path_logits") is not None):
            path_logits = outputs["path_logits"]
            if "left" in path_logits and "right" in path_logits:
                left_logits  = path_logits["left"][...,  :-1, :]  # [B,S-1,V]
                right_logits = path_logits["right"][..., :-1, :]  # [B,S-1,V]
                gold = labels[..., 1:].contiguous()               # [B,S-1]

                left_nll = F.cross_entropy(
                    left_logits.reshape(-1, left_logits.size(-1)),
                    gold.reshape(-1),
                    reduction="none"
                ).reshape(gold.shape)
                right_nll = F.cross_entropy(
                    right_logits.reshape(-1, right_logits.size(-1)),
                    gold.reshape(-1),
                    reduction="none"
                ).reshape(gold.shape)

                better_left  = (left_nll  < right_nll).float()
                better_right = 1.0 - better_left

                gate_info = outputs.get("gate")
                if gate_info is not None and "final_weights" in gate_info:
                    w_left  = gate_info["final_weights"]["left"][...,  :-1, :].squeeze(-1)  # [B,S-1]
                    w_right = gate_info["final_weights"]["right"][..., :-1, :].squeeze(-1)  # [B,S-1]
                    valid_mask = attention_mask[..., 1:].bool()
                    if valid_mask.any():
                        vmf = valid_mask.float()
                        # clamp for numerical safety
                        wl = w_left.clamp_min(1e-8).log()
                        wr = w_right.clamp_min(1e-8).log()
                        gold_aux_loss = -(((better_left * wl) + (better_right * wr)) * vmf).sum() / vmf.sum().clamp_min(1.0)
                        total_loss = total_loss + self.gold_aux_coef * gold_aux_loss

        # === Tether loss (keep right close to left). Leave active even if gate frozen.
        if getattr(self, "tether_coef", 0.0) > 0.0 and outputs.get("path_logits") is not None:
            path_logits = outputs["path_logits"]
            if "left" in path_logits and "right" in path_logits:
                left_probs  = F.softmax(path_logits["left"],  dim=-1)
                right_probs = F.softmax(path_logits["right"], dim=-1)
                # KL(R || L)
                tether_loss = F.kl_div(right_probs.log(), left_probs, reduction="batchmean", log_target=False)
                total_loss = total_loss + self.tether_coef * tether_loss

        # === Per-path LM-head balance regularizers (tiny, safe)
        head_lb_coef      = float(getattr(self, "head_lb_coef", 0.0))      # e.g., 1e-3
        head_entropy_coef = float(getattr(self, "head_entropy_coef", 0.0)) # e.g., 5e-4
        if (head_lb_coef > 0.0 or head_entropy_coef > 0.0):
            hgm = outputs.get("head_gate_means", None)  # {"left": vec[num_heads], "right": vec[num_heads]}
            if hgm:
                reg_lb  = ce_loss.new_tensor(0.0)
                reg_ent = ce_loss.new_tensor(0.0)
                for _, p_global in hgm.items():
                    if p_global is None:
                        continue
                    idx = (p_global > 0)
                    if not torch.any(idx):
                        continue
                    p = p_global[idx]
                    p = p / p.sum().clamp_min(1e-8)  # normalize within this path
                    K = p.numel()

                    if head_lb_coef > 0.0:
                        # KL(p || uniform) = sum p * (log p - log u)
                        u = torch.full_like(p, 1.0 / K)
                        reg_lb = reg_lb + torch.sum(p * (p.clamp_min(1e-8).log() - u.log()))
                    if head_entropy_coef > 0.0:
                        # Encourage high entropy: (log K - H(p))
                        H = -torch.sum(p * p.clamp_min(1e-8).log())
                        reg_ent = reg_ent + (torch.log(torch.tensor(float(K), device=p.device)) - H)

                if head_lb_coef > 0.0:
                    head_lb_reg = reg_lb
                    total_loss = total_loss + head_lb_coef * reg_lb
                if head_entropy_coef > 0.0:
                    head_ent_reg = reg_ent
                    total_loss = total_loss + head_entropy_coef * reg_ent

        # Backward pass with loss scaling
        (total_loss * loss_scale).backward()

        # === Accuracy (masked, next-token)
        with torch.no_grad():
            logits = outputs["log_probs"] if "log_probs" in outputs else outputs["logits"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            valid_mask = attention_mask[..., 1:].bool()
            if (shift_labels == -100).any():
                valid_mask = valid_mask & (shift_labels != -100)
            if valid_mask.any():
                predictions = shift_logits.argmax(dim=-1)
                accuracy = ((predictions == shift_labels) & valid_mask).float().sum() / valid_mask.sum()
                accuracy = float(accuracy.item())
            else:
                accuracy = 0.0

        # === Gate statistics
        gate_stats = {}
        if outputs.get("gate") is not None:
            gate_info = outputs["gate"]

            # Path selection percentages
            if "final_weights" in gate_info:
                weights = gate_info["final_weights"]
                if "left" in weights and "right" in weights:
                    valid_mask_full = attention_mask.bool()
                    if valid_mask_full.any():
                        vmf = valid_mask_full.float()
                        w_left  = weights["left"].squeeze(-1)
                        w_right = weights["right"].squeeze(-1)
                        left_pct  = (w_left  * vmf).sum() / vmf.sum().clamp_min(1.0)
                        right_pct = (w_right * vmf).sum() / vmf.sum().clamp_min(1.0)
                        gate_stats.update({
                            "left_pct":  float(left_pct.item()),
                            "right_pct": float(right_pct.item()),
                        })

            # Gate entropy
            if "gate_logits" in gate_info and gate_info["gate_logits"] is not None:
                gate_logits = gate_info["gate_logits"]
                gate_probs  = F.softmax(gate_logits, dim=-1)
                gate_entropy = -(gate_probs * gate_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
                gate_stats["gate_entropy"] = float(gate_entropy.item())

        self.global_step += 1

        return {
            "loss":     float(ce_loss.item()),
            "ce":       float(ce_loss.item()),
            "lb_loss":  float(lb_loss.item())       if isinstance(lb_loss, torch.Tensor) else lb_loss,
            "gold_aux": float(gold_aux_loss.item()) if isinstance(gold_aux_loss, torch.Tensor) else gold_aux_loss,
            "tether":   float(tether_loss.item())   if isinstance(tether_loss, torch.Tensor) else tether_loss,
            "head_lb":  float(head_lb_reg.item())   if isinstance(head_lb_reg, torch.Tensor) else head_lb_reg,
            "head_ent": float(head_ent_reg.item())  if isinstance(head_ent_reg, torch.Tensor) else head_ent_reg,
            "accuracy": accuracy,
            **gate_stats,
        }

    def train_step(self, batch, optimizer, path_selection="gate_soft"):
        """Complete training step (backward + optimizer step)"""
        metrics = self.backward_only(batch, path_selection=path_selection, loss_scale=1.0)
        self.optimizer_step(optimizer)
        return metrics

    def evaluate(self, dataloader, path_selection="gate_soft", max_batches=None):
        """Evaluate model on validation set"""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        gate_stats_sum = {}
        num_batches = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches and i >= max_batches:
                    break
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    path_selection=path_selection
                )
                
                if outputs["loss"] is not None:
                    total_loss += outputs["loss"].item()
                
                # Calculate accuracy
                if "log_probs" in outputs:
                    logits = outputs["log_probs"]
                else:
                    logits = outputs["logits"]
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                valid_mask = attention_mask[..., 1:].bool()
                if (shift_labels == -100).any():
                    valid_mask = valid_mask & (shift_labels != -100)
                
                if valid_mask.any():
                    predictions = shift_logits.argmax(dim=-1)
                    correct = ((predictions == shift_labels) & valid_mask).sum().item()
                    tokens = valid_mask.sum().item()
                    
                    total_correct += correct
                    total_tokens += tokens
                
                # Accumulate gate statistics
                if outputs.get("gate") is not None:
                    gate_info = outputs["gate"]
                    
                    if "final_weights" in gate_info:
                        weights = gate_info["final_weights"]
                        if "left" in weights and "right" in weights:
                            valid_mask_gate = attention_mask.bool()
                            if valid_mask_gate.any():
                                w_left = weights["left"].squeeze(-1)
                                w_right = weights["right"].squeeze(-1)
                                
                                left_pct = (w_left * valid_mask_gate.float()).sum() / valid_mask_gate.sum()
                                right_pct = (w_right * valid_mask_gate.float()).sum() / valid_mask_gate.sum()
                                
                                gate_stats_sum.setdefault("left_pct", 0.0)
                                gate_stats_sum.setdefault("right_pct", 0.0)
                                gate_stats_sum["left_pct"] += left_pct.item()
                                gate_stats_sum["right_pct"] += right_pct.item()
                    
                    if "gate_logits" in gate_info:
                        gate_logits = gate_info["gate_logits"]
                        gate_probs = F.softmax(gate_logits, dim=-1)
                        gate_entropy = -(gate_probs * gate_probs.log()).sum(dim=-1).mean()
                        gate_stats_sum.setdefault("gate_entropy", 0.0)
                        gate_stats_sum["gate_entropy"] += gate_entropy.item()
                
                num_batches += 1
        
        # Calculate averages
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = total_correct / max(total_tokens, 1)
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        # Average gate statistics
        avg_gate_stats = {}
        for key, value in gate_stats_sum.items():
            avg_gate_stats[key] = value / max(num_batches, 1)
        
        self.model.train()
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "perplexity": perplexity,
            **avg_gate_stats
        }

    def generate_sample(self, prompt, max_length=50, path_selection="gate_soft", 
                       temperature=1.0, do_sample=True):
        """Generate text sample"""
        self.model.eval()
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    path_selection=path_selection
                )
                
                if "log_probs" in outputs:
                    logits = outputs["log_probs"]
                else:
                    logits = outputs["logits"]
                
                next_token_logits = logits[0, -1, :] / temperature
                
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=-1)
                
                # Stop at EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.model.train()
        
        return generated_text

    def save_checkpoint(self, epoch, optimizer, loss, is_final=False):
        """Save training checkpoint"""
        if is_final:
            filename = "final_model.pt"
        else:
            filename = f"checkpoint_epoch_{epoch}_step_{self.global_step}.pt"
        
        filepath = self.checkpoint_dir / filename
        
        # Get model stats for logging
        param_count = self.model.get_parameter_count()
        lm_head_stats = self.model.get_lm_head_stats()
        
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "model_config": {
                "split_at_layer": self.model.split_at_layer,
                "gate_temp": self.gate_temp,
                "use_head_mixture": getattr(self.model, "use_head_mixture", False),
                "head_allocation": getattr(self.model, "head_allocation", None),
            },
            "trainer_config": {
                "lb_coef": self.lb_coef,
                "gold_aux_coef": self.gold_aux_coef,
                "tether_coef": self.tether_coef,
                "clip_grad": self.clip_grad,
            },
            "stats": {
                "parameter_count": param_count,
                "lm_head_stats": lm_head_stats,
            }
        }
        
        torch.save(checkpoint, filepath)
        
        print(f"Checkpoint saved: {filepath}")
        print(f"  Parameters: {param_count['trainable']:,} trainable / {param_count['total']:,} total")
        if lm_head_stats["total_usage"] > 0:
            usage_str = " | ".join(f"h{i}:{pct:.1f}%" for i, pct in enumerate(lm_head_stats['usage_percentages']))
            print(f"  LM Head Usage: {usage_str}")
        
        return filepath

    def load_checkpoint(self, path, optimizer=None, scheduler=None,
                        load_optimizer=True, load_scheduler=True,
                        strict_model=False, drop_mismatched_head_counters=True):
        import torch
        ckpt = torch.load(path, map_location="cpu")

        # --- robust extraction of model weights ---
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
                sd = ckpt["model_state_dict"]
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                sd = ckpt["state_dict"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                sd = ckpt["model"]
            else:
                sd = ckpt  # maybe it's already a raw state_dict
        else:
            sd = ckpt

        # Drop old head usage buffers if shape mismatches current head count
        if drop_mismatched_head_counters and isinstance(sd, dict):
            try:
                n = int(getattr(self.model.lm_head, "num_perceptrons", None))
            except Exception:
                n = None
            if n is not None:
                for k in ("lm_head.usage_counts", "lm_head.selection_counts"):
                    if k in sd and isinstance(sd[k], torch.Tensor) and sd[k].numel() != n:
                        print(f"[ckpt] dropping {k}: ckpt={sd[k].numel()} vs model={n}")
                        sd.pop(k)

        missing, unexpected = self.model.load_state_dict(sd, strict=strict_model)
        print(f"[ckpt] model loaded (strict={strict_model}): missing={len(missing)} unexpected={len(unexpected)}")

        # (Keep your safe/optional optimizer & scheduler restore here)

        epoch = int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0
        global_step = int(ckpt.get("global_step", 0)) if isinstance(ckpt, dict) else 0
        return {"epoch": epoch, "global_step": global_step, "raw": ckpt}
