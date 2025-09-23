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

# Independent Dual Path GPT Model with Completely Separate Processing
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel
import math


class IndependentDualPathGPT2(nn.Module):
    """
    GPT-2 model with completely independent left and right paths after block 6.
    Left path: frozen baseline (HF weights)
    Right path: trainable specialization
    """

    def __init__(self, config, pretrained_model="gpt2", split_at_layer=6, gate_hidden=256, gate_temp=1.0):
        super().__init__()
        self.config = config
        self.split_at_layer = split_at_layer
        self.gate_temp = gate_temp
        
        # Load pretrained model for initialization
        print(f"Loading pretrained model: {pretrained_model}")
        hf_model = GPT2LMHeadModel.from_pretrained(pretrained_model)
        
        # Shared layers (0 to split_at_layer-1)
        self.shared_layers = nn.ModuleList()
        for i in range(split_at_layer):
            self.shared_layers.append(hf_model.transformer.h[i])
        
        # Copy other transformer components
        self.wte = hf_model.transformer.wte  # token embeddings
        self.wpe = hf_model.transformer.wpe  # position embeddings
        self.ln_f_left = hf_model.transformer.ln_f   # final layer norm for left
        self.ln_f_right = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)  # separate for right
        
        # Initialize right path layer norm from left
        self.ln_f_right.load_state_dict(self.ln_f_left.state_dict())
        
        # Left path (frozen) - remaining layers from pretrained
        self.left_path = nn.ModuleList()
        for i in range(split_at_layer, len(hf_model.transformer.h)):
            self.left_path.append(hf_model.transformer.h[i])
        
        # Right path (trainable) - copy from pretrained but will be trained
        self.right_path = nn.ModuleList()
        for i in range(split_at_layer, len(hf_model.transformer.h)):
            # Create new layer and copy weights
            right_layer = type(hf_model.transformer.h[i])(config)
            right_layer.load_state_dict(hf_model.transformer.h[i].state_dict())
            self.right_path.append(right_layer)
        
        # LM heads - completely independent
        self.left_lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.right_lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize both heads with pretrained weights
        self.left_lm_head.weight.data.copy_(hf_model.lm_head.weight.data)
        self.right_lm_head.weight.data.copy_(hf_model.lm_head.weight.data)
        
        # Freeze left/baseline & assert
        self._freeze_left_path()
        self._assert_left_frozen()

        # trainable per-token gate, computed at the split (layer-6 output) ---
        # Gate takes shared hidden [B,S,E] and outputs logits [B,S,2] for {left, right}
        self.gate = nn.Sequential(
            nn.Linear(config.n_embd, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, 2)
        )
        # Bias gate slightly toward left/baseline at init (optional/stable start)
        with torch.no_grad():
            last = self.gate[-1]
            last.bias.fill_(0.25)  # positive toward index 0 if we treat 0=left
        
        try:
            attn_impl = getattr(self.config, "_attn_implementation", None)
        except AttributeError:
            attn_impl = None
        if attn_impl is None:
            self.config._attn_implementation = "eager"  # safest default

        # Make 100% sure every block references a config with the same setting
        for blk in list(self.shared_layers) + list(self.left_path) + list(self.right_path):
            # Some HF modules store their own .config; align it.
            if hasattr(blk, "config"):
                blk.config._attn_implementation = self.config._attn_implementation
            # Also align nested attention module if present
            if hasattr(blk, "attn") and hasattr(blk.attn, "config"):
                blk.attn.config._attn_implementation = self.config._attn_implementation

        # Clean up
        del hf_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"Model initialized with split at layer {split_at_layer}")
        print(f"Shared layers: {len(self.shared_layers)}")
        print(f"Left path layers: {len(self.left_path)} (frozen)")
        print(f"Right path layers: {len(self.right_path)} (trainable)")
    
    def _expand_attn_mask(self, attention_mask, dtype, tgt_len=None):
        if attention_mask is None:
            return None
        bsz, src_len = attention_mask.shape
        if tgt_len is None:
            tgt_len = src_len
        mask = attention_mask[:, None, None, :].to(dtype=dtype)  # [B,1,1,S]
        mask = (1.0 - mask) * -1e4  # 0 keep, -1e4 mask
        return mask

    def forward_shared_layers(self, hidden_states, attention_mask):
        attn = self._expand_attn_mask(attention_mask, hidden_states.dtype, tgt_len=hidden_states.size(1))
        for layer in self.shared_layers:
            hidden_states = layer(hidden_states, attention_mask=attn, use_cache=False)[0]
        return hidden_states

    def forward_left_path(self, hidden_states, attention_mask):
        attn = self._expand_attn_mask(attention_mask, hidden_states.dtype, tgt_len=hidden_states.size(1))
        for layer in self.left_path:
            hidden_states = layer(hidden_states, attention_mask=attn, use_cache=False)[0]
        hidden_states = self.ln_f_left(hidden_states)
        logits = self.left_lm_head(hidden_states)
        return logits

    def forward_right_path(self, hidden_states, attention_mask):
        attn = self._expand_attn_mask(attention_mask, hidden_states.dtype, tgt_len=hidden_states.size(1))
        for layer in self.right_path:
            hidden_states = layer(hidden_states, attention_mask=attn, use_cache=False)[0]
        hidden_states = self.ln_f_right(hidden_states)
        logits = self.right_lm_head(hidden_states)
        return logits

    def _freeze_left_path(self):
        """Freeze all left path parameters"""
        # Freeze shared layers (will be used by both paths but not trained)
        for param in self.shared_layers.parameters():
            param.requires_grad = False
        
        # Freeze embeddings
        for param in self.wte.parameters():
            param.requires_grad = False
        for param in self.wpe.parameters():
            param.requires_grad = False
            
        # Freeze left-specific components
        for param in self.left_path.parameters():
            param.requires_grad = False
        for param in self.ln_f_left.parameters():
            param.requires_grad = False
        for param in self.left_lm_head.parameters():
            param.requires_grad = False
            
        print("Left path frozen (including shared layers)")
    
    def get_embeddings(self, input_ids, attention_mask=None):
        """Get initial embeddings (shared)"""
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeddings = self.wte(input_ids)
        
        # Position embeddings
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.wpe(position_ids)
        
        return token_embeddings + position_embeddings
   
    def get_parameter_count(self):
        """Get parameter counts for analysis"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'trainable_ratio': trainable_params / total_params
        }

    def _assert_left_frozen(self):
        def _all_frozen(mod):
            return all(not p.requires_grad for p in mod.parameters())
        # shared layers are frozen in your code (by design)
        assert _all_frozen(self.shared_layers), "Shared layers must be frozen!"
        assert _all_frozen(self.left_path), "Left (baseline) path must be frozen!"
        assert _all_frozen(self.ln_f_left), "Left LN must be frozen!"
        assert _all_frozen(self.left_lm_head), "Left LM head must be frozen!"
        assert _all_frozen(self.wte) and _all_frozen(self.wpe), "Embeddings must be frozen!"

    def _gate_weights(self, shared_output, hard=False):
        # shared_output: [B,S,E] at the split (layer-6 output)
        logits = self.gate(shared_output)  # [B,S,2], index 0=left, 1=right
        if hard:
            # Straight-through: one-hot in forward, softmax used for backward
            soft = F.softmax(logits / max(self.gate_temp, 1e-6), dim=-1)
            idx = torch.argmax(soft, dim=-1)                            # [B,S]
            hard_onehot = F.one_hot(idx, num_classes=2).to(soft.dtype)  # [B,S,2]
            gate = hard_onehot + (soft - soft.detach())                 # STE
            return gate, logits
        else:
            gate = F.softmax(logits / max(self.gate_temp, 1e-6), dim=-1)
            return gate, logits

    def _gate_margin_loss(self, logits, margin: float = 0.25):
        """
        Encourage confident gate choices: max_logit - second_logit >= margin.
        logits: [B,S,2] (left,right)
        """
        # works for K>=2, but we use K==2
        top2 = logits.topk(2, dim=-1).values     # [B,S,2]
        gap  = top2[..., 0] - top2[..., 1]       # [B,S]
        return torch.relu(margin - gap).mean()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        return_both_paths: bool = False,
        path_selection: str = "max_prob",
    ):
        """
        Dual-path forward with optional gate-margin hardening.

        path_selection:
        - "left_only"  : use baseline (left) path only
        - "right_only" : use trainable (right) path only
        - "gate_soft"  : soft gate over [left,right]
        - "gate_hard"  : STE hard gate over [left,right]
        - "max_prob"   : pick path by higher per-token max prob (left/right)
        - "soft_weighted": weight paths by their per-token max prob
        """
        import torch.nn.functional as F

        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # 1) Embeddings
        hidden_states = self.get_embeddings(input_ids, attention_mask)

        # 2) Shared (0..split-1)
        shared_output = self.forward_shared_layers(hidden_states, attention_mask)

        # 3) Split at layer-6
        left_hidden  = shared_output.clone()
        right_hidden = shared_output.clone()

        # 4) Paths
        #    Baseline (left) is frozen: no grad
        with torch.no_grad():
            left_logits = self.forward_left_path(left_hidden, attention_mask)   # [B,S,V]
        right_logits = self.forward_right_path(right_hidden, attention_mask)    # [B,S,V]

        if return_both_paths:
            return left_logits, right_logits

        # Bookkeeping for gating & aux loss
        gate_info   = None
        gate_logits = None                    # [B,S,2] when using a gate
        aux_m       = 0.0                     # margin aux-loss accumulator

        # 5) Selection / gating
        if path_selection == "left_only":
            final_logits = left_logits

        elif path_selection == "right_only":
            final_logits = right_logits

        elif path_selection == "gate_soft":
            gate, gate_logits = self._gate_weights(shared_output, hard=False)   # gate: [B,S,2]
            gw_left  = gate[..., 0].unsqueeze(-1)                               # [B,S,1]
            gw_right = gate[..., 1].unsqueeze(-1)
            final_logits = gw_left * left_logits + gw_right * right_logits
            gate_info = {"gate_soft": gate, "gate_logits": gate_logits}

        elif path_selection == "gate_hard":
            gate, gate_logits = self._gate_weights(shared_output, hard=True)    # STE one-hot-ish
            gw_left  = gate[..., 0].unsqueeze(-1)
            gw_right = gate[..., 1].unsqueeze(-1)
            final_logits = gw_left * left_logits + gw_right * right_logits
            gate_info = {"gate_hard": gate, "gate_logits": gate_logits}

        elif path_selection == "max_prob":
            # Compare per-token confidence of each path (max over vocab)
            left_probs  = F.softmax(left_logits,  dim=-1)
            right_probs = F.softmax(right_logits, dim=-1)
            left_max,  _ = left_probs.max(dim=-1)    # [B,S]
            right_max, _ = right_probs.max(dim=-1)   # [B,S]
            use_right = (right_max > left_max).unsqueeze(-1)  # [B,S,1]
            final_logits = torch.where(use_right, right_logits, left_logits)

        elif path_selection == "soft_weighted":
            # Soft weights from per-token max probs
            left_probs  = F.softmax(left_logits,  dim=-1)
            right_probs = F.softmax(right_logits, dim=-1)
            left_max,  _ = left_probs.max(dim=-1)    # [B,S]
            right_max, _ = right_probs.max(dim=-1)   # [B,S]
            total = left_max + right_max + 1e-8
            lw = (left_max / total).unsqueeze(-1)    # [B,S,1]
            rw = (right_max / total).unsqueeze(-1)
            final_logits = lw * left_logits + rw * right_logits

        else:
            raise ValueError(f"Unknown path_selection: {path_selection}")

        # 5b) Gate-margin hardening (only if gate was used AND we're training)
        #     Encourages (top1 - top2) on gate logits to exceed `gate_margin`.
        if self.training and gate_logits is not None:
            coef   = float(getattr(self, "gate_margin_coef", 0.0))
            margin = float(getattr(self, "gate_margin", 0.25))
            if coef > 0.0:
                # inline margin loss (avoid extra helpers)
                top2  = gate_logits.topk(2, dim=-1).values        # [B,S,2]
                gaps  = top2[..., 0] - top2[..., 1]               # [B,S]
                m_gate = torch.relu(margin - gaps).mean()
                aux_m += coef * m_gate

        # 6) Loss (CE + optional gate-margin aux)
        loss = None
        if labels is not None:
            shift_logits = final_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss = ce_loss
            if self.training and aux_m != 0.0:
                loss = loss + aux_m

        return {
            "loss": loss,
            "logits": final_logits,
            "left_logits": left_logits,
            "right_logits": right_logits,
            "gate": gate_info,
        }

    def get_trainable_parameters(self):
        # include gate + right path + right LN + right head (left is frozen)
        params = []
        params += list(self.gate.parameters())              # NEW
        params += list(self.right_path.parameters())
        params += list(self.ln_f_right.parameters())
        params += list(self.right_lm_head.parameters())
        return [p for p in params if p.requires_grad]

    def _expand_mask(self, mask):
        # mask [B,S] -> [B,1,1,S] with 0/-inf like HF
        if mask is None: return None
        return (1.0 - mask[:, None, None, :].to(dtype=torch.float32)) * -1e4

# =========================
# DualPathTrainer 
# =========================
class DualPathTrainer:
    """Trainer for the dual path model with log-prob mixture & routing losses"""

    def __init__(
        self,
        model,
        tokenizer,
        device,
        checkpoint_dir,
        *,
        # loss weights (safe defaults)
        lb_coef=1e-3,       # load-balance KL
        gold_aux_coef=1e-3, # gold-routing aux CE on gate
        tether_coef=5e-4,   # KL(mix || left) early; anneal during training
        gate_temp=1.0,      # temperature for gate softmax
        clip_grad=1.0
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.global_step = 0

        # loss knobs
        self.lb_coef = lb_coef
        self.gold_aux_coef = gold_aux_coef
        self.tether_coef = tether_coef
        self.gate_temp = gate_temp
        self.clip_grad = clip_grad

        self.hard_blend_lambda = 0.0  # 0=soft only, 1=STE hard; can be ramped during Phase B


        # report params
        info = self.model.get_parameter_count()
        print(f"Model parameters:\n  Total: {info['total']:,}\n  Trainable: {info['trainable']:,}\n  Frozen: {info['frozen']:,}\n  Trainable ratio: {info['trainable_ratio']:.2%}")  # :contentReference[oaicite:4]{index=4}

    def create_optimizer(self, lr=2e-5, weight_decay=0.01):
        trainable_params = self.model.get_trainable_parameters()  # gate + right path + ln_f_right + right head :contentReference[oaicite:5]{index=5}
        opt = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
        print(f"Optimizer created with {len(trainable_params)} parameter groups")
        return opt

    def zero_grad(self, optimizer):
        optimizer.zero_grad()

    def optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), max_norm=self.clip_grad)
        optimizer.step()
        optimizer.zero_grad()
        self.global_step += 1  # count *optimizer* steps, not micro-batches

    def backward_only(self, batch, *, path_selection="gate_soft", loss_scale=1.0):
        """
        One micro-batch: forward + loss + (loss_scale * backward).
        No optimizer.step() here. Returns dict of lightweight metrics.
        """
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        token_weights = batch.get("token_weights", None)
        if token_weights is not None:
            token_weights = token_weights.to(self.device)

        # forward
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=None, path_selection=path_selection)
        left_logits  = outputs["left_logits"]
        right_logits = outputs["right_logits"]
        gate_logits  = outputs["gate"]["gate_logits"]

        # log-space mixture + losses (reusing your existing helpers)
        log_mix, logp_left, logp_right, logpi = self._log_mix(left_logits, right_logits, gate_logits)
        ce_loss = self._token_ce(log_mix, labels, token_weights)

        # gold-routing aux (shifted) - FIXED VERSION
        gold = labels[:, 1:]  # [B, S-1]
        
        # Ensure gold tokens are valid (not -100 padding)
        valid_mask = (gold != -100)
        if not valid_mask.any():
            # No valid tokens, use dummy loss
            gold_aux = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            # Clamp gold tokens to valid range to prevent index errors
            gold_clamped = torch.clamp(gold, 0, logp_left.size(-1) - 1)
            
            # Only compute for valid positions
            l_gold_all = logp_left[:, :-1, :].gather(-1, gold_clamped.unsqueeze(-1)).squeeze(-1)
            r_gold_all = logp_right[:, :-1, :].gather(-1, gold_clamped.unsqueeze(-1)).squeeze(-1)
            
            # Apply valid mask
            l_gold = torch.where(valid_mask, l_gold_all, torch.zeros_like(l_gold_all))
            r_gold = torch.where(valid_mask, r_gold_all, torch.zeros_like(r_gold_all))
            
            # Create target (0=left better, 1=right better)
            target_all = (r_gold > l_gold).long()
            target = torch.where(valid_mask, target_all, torch.zeros_like(target_all))
            
            # Compute cross entropy only on valid positions
            gate_logits_shift = gate_logits[:, :-1, :]  # [B,S-1,2]
            
            # Flatten and select only valid positions
            gate_flat = gate_logits_shift.reshape(-1, 2)  # [B*(S-1), 2]
            target_flat = target.reshape(-1)  # [B*(S-1)]
            valid_flat = valid_mask.reshape(-1)  # [B*(S-1)]
            
            if valid_flat.any():
                gold_aux = F.cross_entropy(gate_flat[valid_flat], target_flat[valid_flat])
            else:
                gold_aux = torch.tensor(0.0, device=self.device, requires_grad=True)

        # load-balance
        gate_soft = torch.softmax(gate_logits, dim=-1)
        usage = gate_soft.mean(dim=(0,1))
        u = usage + 1e-8
        lb_kl = (u * (u.log() - math.log(0.5))).sum()

        # tether
        tether = self._kl_mean(log_mix, logp_left)

        loss = ce_loss + self.gold_aux_coef*gold_aux + self.lb_coef*lb_kl + self.tether_coef*tether

        # scale for accumulation
        (loss * loss_scale).backward()

        # cheap accuracy from mixed distribution
        with torch.no_grad():
            preds = log_mix.argmax(dim=-1)
            shift_labels = labels[:, 1:]
            shift_preds  = preds[:, :-1]
            valid = (shift_labels != -100)
            acc = ((shift_preds == shift_labels) & valid).sum().float() / (valid.sum().float() + 1e-8)

        gstats = self._gate_usage_stats(gate_logits)

        return {
            "loss": float(loss.detach()),
            "ce": float(ce_loss.detach()),
            "gold_aux": float(gold_aux.detach()),
            "lbkl": float(lb_kl.detach()),
            "tether": float(tether.detach()),
            "accuracy": float(acc.detach()),
            "right_pct": gstats["right_pct"],          # mean prob to path-1
            "gate_entropy": gstats["gate_entropy"],    # nats
            "usage_kl": gstats["usage_kl"],            # KL(usage || uniform)
            "mode": self._mode_string(),
        }

    def _token_ce(self, log_probs, labels, weights=None, ignore_index: int = -100):
        """
        Cross-entropy over log_probs vs labels with optional per-token weights.
        Accepts log_probs of shape [B,S,V] or [S,V]; labels [B,S] or [S].
        Safely masks ignore_index before gather (GPU-safe).
        """
        # ---- Normalize shapes to [B,S,V] and [B,S] ----
        if log_probs.dim() == 2:         # [S,V] -> [1,S,V]
            log_probs = log_probs.unsqueeze(0)
        elif log_probs.dim() != 3:
            raise ValueError(f"log_probs must be 2D or 3D, got {log_probs.shape}")

        if labels.dim() == 1:            # [S] -> [1,S]
            labels = labels.unsqueeze(0)
        elif labels.dim() != 2:
            raise ValueError(f"labels must be 1D or 2D, got {labels.shape}")

        if weights is not None:
            if weights.dim() == 1:       # [S] -> [1,S]
                weights = weights.unsqueeze(0)
            elif weights.dim() != 2:
                raise ValueError(f"weights must be 1D or 2D, got {weights.shape}")

        B, S, V = log_probs.shape

        # ---- Standard next-token shift ----
        shift_logp = log_probs[:, :-1, :]     # [B,S-1,V]
        shift_y    = labels[:, 1:]            # [B,S-1]

        # ---- Valid mask (ignore_index) BEFORE gather ----
        valid_mask = (shift_y != ignore_index)         # [B,S-1] bool
        if not valid_mask.any():
            # no valid targets in this micro-batch
            return shift_logp.new_zeros(())

        safe_y = shift_y.masked_fill(~valid_mask, 0)   # any valid index

        # ---- Gather gold log-probs (GPU-safe) ----
        gold_logp = shift_logp.gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)  # [B,S-1]
        nll = -gold_logp

        # ---- Optional token weights ----
        if weights is not None:
            w = weights[:, 1:]                                    # align shift
            w = torch.where(valid_mask, w, torch.zeros_like(w))   # zero ignored
            # normalize to mean≈1 over valid tokens (stable LR)
            denom = (w.sum() + 1e-8)
            if denom > 0:
                w = w * (w.numel() / denom)
                nll = nll * w

        # ---- Mean over valid tokens only ----
        loss = nll[valid_mask].mean()
        return loss

    @staticmethod
    def _kl_mean(p_log, q_log):
        """Mean KL( p || q ) where both are log-probs [B,S,V]."""
        p = p_log.exp()
        return (p * (p_log - q_log)).mean()

    def _log_mix(self, left_logits, right_logits, gate_logits):
        logp_left  = torch.log_softmax(left_logits,  dim=-1)
        logp_right = torch.log_softmax(right_logits, dim=-1)

        # soft gate at temperature T (used for losses/regularizers)
        g      = gate_logits / max(self.gate_temp, 1e-6)
        soft   = torch.softmax(g, dim=-1)                         # [B,S,2]

        # hard one-hot with straight-through estimator
        idx    = torch.argmax(soft, dim=-1)                       # [B,S]
        hard   = torch.nn.functional.one_hot(idx, num_classes=2).to(soft.dtype)  # [B,S,2]
        ste    = hard + (soft - soft.detach())

        lam    = float(getattr(self, "hard_blend_lambda", 0.0))
        w      = (1.0 - lam) * soft + lam * ste                   # [B,S,2]  (forward weights)

        # stack path log-probs: [B,S,2,V]
        lp = torch.stack([logp_left, logp_right], dim=-2)
        log_mix = torch.logsumexp((w.clamp_min(1e-8)).log().unsqueeze(-1) + lp, dim=-2)
        # return soft log-pi too so gold-aux/balance can use it if needed
        logpi_soft = torch.log_softmax(g, dim=-1)
        return log_mix, logp_left, logp_right, logpi_soft

    def train_step(self, batch, optimizer, path_selection="gate_soft"):
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        token_weights = batch.get("token_weights", None)
        if token_weights is not None:
            token_weights = token_weights.to(self.device)

        optimizer.zero_grad()

        # forward with learnable gate (soft)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # we compute CE on mixed log-probs below
            path_selection=path_selection  # "gate_soft" recommended for training
        )
        left_logits  = outputs["left_logits"]
        right_logits = outputs["right_logits"]

        gate_info = outputs.get("gate", None)
        if gate_info is None or ("gate_logits" not in gate_info):
            raise RuntimeError("Model must return gate_logits when using gate_* modes.")
        gate_logits = gate_info.get("gate_logits")  # [B,S,2]

        # ---- log-space mixture (core) ----
        log_mix, logp_left, logp_right, logpi = self._log_mix(left_logits, right_logits, gate_logits)

        # ---- main CE loss on mixture (token-aware optional) ----
        ce_loss = self._token_ce(log_mix, labels, token_weights)

        # ---- gold-routing aux on the gate - FIXED VERSION ----
        gold = labels[:, 1:]  # [B, S-1] - align with CE shift
        
        # Ensure gold tokens are valid (not -100 padding)
        valid_mask = (gold != -100)
        if not valid_mask.any():
            # No valid tokens, use dummy loss
            gold_aux = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            # Clamp gold tokens to valid range to prevent index errors
            gold_clamped = torch.clamp(gold, 0, logp_left.size(-1) - 1)
            
            # Only compute for valid positions
            l_gold_all = logp_left[:, :-1, :].gather(-1, gold_clamped.unsqueeze(-1)).squeeze(-1)  # [B,S-1]
            r_gold_all = logp_right[:, :-1, :].gather(-1, gold_clamped.unsqueeze(-1)).squeeze(-1)  # [B,S-1]
            
            # Apply valid mask
            l_gold = torch.where(valid_mask, l_gold_all, torch.zeros_like(l_gold_all))
            r_gold = torch.where(valid_mask, r_gold_all, torch.zeros_like(r_gold_all))
            
            # target 0 if left better, 1 if right better (ties go to left)
            target_all = (r_gold > l_gold).long()
            target = torch.where(valid_mask, target_all, torch.zeros_like(target_all))
            
            gate_logits_shift = gate_logits[:, :-1, :]  # [B,S-1,2]
            
            # Flatten and select only valid positions
            gate_flat = gate_logits_shift.reshape(-1, 2)  # [B*(S-1), 2]
            target_flat = target.reshape(-1)  # [B*(S-1)]
            valid_flat = valid_mask.reshape(-1)  # [B*(S-1)]
            
            if valid_flat.any():
                gold_aux = F.cross_entropy(gate_flat[valid_flat], target_flat[valid_flat])
            else:
                gold_aux = torch.tensor(0.0, device=self.device, requires_grad=True)

        # ---- load-balance KL to uniform over {left,right} ----
        gate_soft = torch.softmax(gate_logits, dim=-1)             # [B,S,2]
        usage = gate_soft.mean(dim=(0, 1))                         # [2]
        u = usage + 1e-8
        lb_kl = (u * (u.log() - math.log(0.5))).sum()  # KL(u || [0.5,0.5])

        # ---- optional tether to baseline (anneal this during training) ----
        # KL( mix || left ) keeps the mixture near baseline early on
        tether = self._kl_mean(log_mix, logp_left)

        # ---- total loss ----
        loss = ce_loss \
            + self.gold_aux_coef * gold_aux \
            + self.lb_coef * lb_kl \
            + self.tether_coef * tether

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), max_norm=self.clip_grad)
        optimizer.step()

        # ---- accuracy for dashboard ----
        with torch.no_grad():
            preds = log_mix.argmax(dim=-1)
            shift_labels = labels[:, 1:]
            shift_preds  = preds[:, :-1]
            valid = (shift_labels != -100)
            acc = ((shift_preds == shift_labels) & valid).sum().float() / (valid.sum().float() + 1e-8)

        self.global_step += 1

        # ---- routing diagnostics (optional) ----
        if self.global_step % 50 == 0:
            right_mean = gate_soft[..., 1].mean().item()
            print(f"[step {self.global_step}] loss={loss.item():.4f} | CE={ce_loss.item():.4f} | gold_aux={gold_aux.item():.4f} | lbKL={lb_kl.item():.4f} | tether={tether.item():.4f} | right%={right_mean:.3f}")

        return {
            "loss": loss.item(),
            "accuracy": acc.item()
        }

    def evaluate(self, dataloader, path_selection="gate_soft"):
        self.model.eval()
        total_ce = 0.0
        total_acc = 0.0
        right_accum = 0.0
        ent_accum = 0.0
        kl_accum = 0.0
        n = 0
        m = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                    path_selection=path_selection,  # evaluate the same mixture you train
                )
                left_logits  = outputs["left_logits"]
                right_logits = outputs["right_logits"]
                gate_logits  = outputs["gate"]["gate_logits"] if outputs.get("gate") else None
                if gate_logits is None:
                    raise RuntimeError("Need gate_logits at eval for mixture PPL.")

                log_mix, _, _, _ = self._log_mix(left_logits, right_logits, gate_logits)

                # CE (once)
                ce = self._token_ce(log_mix, labels)
                total_ce += ce.item()

                # Accuracy (once)
                preds = log_mix.argmax(dim=-1)
                shift_labels = labels[:, 1:]
                shift_preds  = preds[:, :-1]
                valid = (shift_labels != -100)
                acc = ((shift_preds == shift_labels) & valid).sum().float() / (valid.sum().float() + 1e-8)
                total_acc += acc.item()
                n += 1

                # Gate stats (accumulate means)
                gstats = self._gate_usage_stats(gate_logits)
                right_accum += gstats["right_pct"]
                ent_accum   += gstats["gate_entropy"]
                kl_accum    += gstats["usage_kl"]
                m += 1

        avg_ce  = total_ce / max(n, 1)
        ppl     = math.exp(avg_ce)
        avg_acc = total_acc / max(n, 1)
        gate_stats = {}
        if m > 0:
            gate_stats = {
                "right_pct":   right_accum / m,
                "gate_entropy": ent_accum / m,
                "usage_kl":     kl_accum / m,
            }
        return {"loss": avg_ce, "perplexity": ppl, "accuracy": avg_acc, **gate_stats, "mode": self._mode_string()}

    def save_checkpoint(self, epoch, optimizer, loss, is_final=False):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
            'loss': loss,
            "model_config": self.model.config.to_dict(),   
            'is_final': is_final
        }

        if is_final:
            filename = "final_checkpoint.pt"
        else:
            filename = f"checkpoint_step_{self.global_step}.pt"
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
        return filepath
    
    def load_checkpoint(self, checkpoint_path, optimizer=None):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state loaded")
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded")
        
        self.global_step = checkpoint.get('global_step', 0)
        epoch = checkpoint.get('epoch', 0)
        
        print(f"Checkpoint loaded - resuming from step {self.global_step}, epoch {epoch}")
        return checkpoint
    
    def _mixture_next_logits(self, generated_ids, attention_mask):
        out = self.model(input_ids=generated_ids, attention_mask=attention_mask, labels=None, path_selection="gate_soft")
        left, right = out["left_logits"], out["right_logits"]
        gate_logits  = out["gate"]["gate_logits"]
        log_mix, _, _, _ = self._log_mix(left, right, gate_logits)
        # return the last-step logits consistent with training (they’re log-probs)
        return log_mix[:, -1, :]  # [B,V]

    @staticmethod
    def _gate_usage_stats(gate_logits: torch.Tensor):
        # gate_logits: [B,S,2]
        with torch.no_grad():  # metrics shouldn't require grad
            soft = torch.softmax(gate_logits, dim=-1)           # [B,S,2]
            mean_right = soft[..., 1].mean()
            ent = -(soft * (soft.clamp_min(1e-8)).log()).sum(dim=-1).mean()
            usage = soft.mean(dim=(0,1))                        # [2]
            u = usage.clamp_min(1e-8)
            lbkl = (u * (u.log() - math.log(0.5))).sum()
            return {
                "right_pct": float(mean_right.item()),
                "gate_entropy": float(ent.item()),
                "usage_kl": float(lbkl.item()),
            }

    def _mode_string(self):
        # Report gate mode succinctly
        # If you added hard-blend, this attribute may exist; default to soft
        lam = getattr(self, "hard_blend_lambda", 0.0)
        return f"mode={'hard-blend' if lam > 0 else 'soft'}(λ={lam:.2f},T={self.gate_temp:.2f})"

    def generate_sample(self, prompt, max_length=50, path_selection="max_prob", 
                       temperature=1.0, do_sample=False):
        """Generate text sample"""
        self.model.eval()
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids)).to(self.device)
        
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    path_selection=path_selection
                )
                
                # Get next token logits
                next_token_logits = self._mixture_next_logits(generated_ids, attention_mask) / temperature
                
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
                
                # Stop at EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text
    

    