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


class TokenAwareCrossEntropyLoss:
    """
    CE with optional per-token weighting (token_weights) and an optional
    auxiliary 'Z-distance' term that can be annealed out (alpha schedule).

    Backward-compatible: if constructed with `alpha=...`, it behaves like before.
    Prefer using (token_weights, alpha_start, alpha_end, total_steps).
    """
    def __init__(
        self,
        token_weights=None,      # torch.Tensor [V] or None
        alpha_start=None,        # float | None
        alpha_end=None,          # float | None
        total_steps=None,        # int | None
        ignore_index=-100,
        epsilon=1e-6,
        alpha=None               # legacy param (will map to start=end=alpha)
    ):
        # Back-compat mapping
        if alpha is not None and (alpha_start is None and alpha_end is None):
            alpha_start = alpha_end = float(alpha)

        self.token_weights = token_weights
        self.alpha_start = float(1.0 if alpha_start is None else alpha_start)
        self.alpha_end   = float(1.0 if alpha_end   is None else alpha_end)
        self.total_steps = total_steps
        self.ignore_index = ignore_index
        self.eps = float(epsilon)
        self.prev_ce = None
        self.ce_changed = 0
        self.ce_change_sum = 0.0
        self.running_delta = 0.0

    def _alpha(self, step):
        if self.total_steps is None or step is None:
            return self.alpha_end
        t = max(0.0, min(1.0, step / max(1, self.total_steps)))
        return self.alpha_start + (self.alpha_end - self.alpha_start) * t

    def __call__(self, logits, targets, step=None):
        # logits: [N, V], targets: [N] with ignore_index for pads
        valid = (targets != self.ignore_index)
        if valid.sum() == 0:
            return logits.new_zeros(())

        logits_v  = logits[valid]           # [M, V]
        targets_v = targets[valid]          # [M]

        # base CE (per-token)
        ce = F.cross_entropy(logits_v, targets_v, reduction='none')  # [M]

        # per-token weights (built from CLEANED counts -> token_weights.pt)
        if self.token_weights is not None:
            # Build weights aligned to targets (safe for -100 because we masked)
            w = self.token_weights[targets_v].to(logits_v.device)    # [M]
            # normalize per batch to keep mean ≈ 1
            w = w / w.mean().clamp_min(1e-8)
            ce = ce * w

        ce_loss = ce.mean()

        # --- bookkeeping for trainer logs ---
        # Lazily add fields so older checkpoints/classes don't crash.
        if not hasattr(self, "prev_ce"):
            self.prev_ce = None
            self.ce_changed = 0
            self.ce_change_sum = 0.0
            self.running_delta = 0.0

        cur = float(ce_loss.detach().item())
        if self.prev_ce is not None:
            delta = abs(cur - self.prev_ce)
            self.ce_change_sum += delta
            self.ce_changed += 1
            self.running_delta = self.ce_change_sum / max(1, self.ce_changed)
        self.prev_ce = cur
        # --- end bookkeeping ---

        # auxiliary Z-distance (only if alpha < 1)
        a = self._alpha(step)
        if a < 0.999:
            probs = logits_v.softmax(dim=-1)                          # [M, V]
            vocab = torch.arange(probs.size(-1), device=probs.device, dtype=probs.dtype)
            mu = (probs * vocab).sum(dim=-1)
            var = (probs * (vocab - mu.unsqueeze(1))**2).sum(dim=-1)
            sigma = (var + self.eps).sqrt()
            z = (targets_v.to(probs.dtype) - mu) / (sigma + self.eps)
            z = z.clamp_(-6.0, 6.0)
            aux = 0.5 * (z ** 2).mean()
            return a * ce_loss + (1.0 - a) * aux

        return ce_loss


class GatingMLP(nn.Module):
    """Gate over P heads; input is last hidden state x [B,S,E]."""
    def __init__(self, hidden_size: int, num_heads: int, gate_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, num_heads)
        )

    def forward(self, x):  # x: [B,S,E]
        return self.net(x)  # [B,S,P] (logits over heads)

# 🧠 Multi-Perceptron LM Head with FULL Input (Fixed Version)
class MultiPerceptronLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, config, pretrained_weights=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_perceptrons = getattr(config, "n_lm_perceptrons", 4)
        
        # ✅ FIXED: Each perceptron now takes FULL hidden_size input
        # Create multiple linear perceptrons with FULL input size
        self.perceptrons = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False) for _ in range(self.num_perceptrons)
        ])
        
        # Initialize with pretrained weights if provided
        if pretrained_weights is not None:
            # ✅ FIXED: Initialize ALL perceptrons with the SAME pretrained weights
            for i, p in enumerate(self.perceptrons):
                p.weight.data.copy_(pretrained_weights)
                if i > 0:
                    # Flatten the distribution by scaling weights toward zero
                    p.weight.data *= 0.99  # Slightly reduce all weights
                    # or
                    p.weight.data *= (1.0 - 0.01 * i)  # Progressive flattening
            print(f"✅ All {self.num_perceptrons} perceptrons initialized with full pretrained weights")
        else:
            for p in self.perceptrons:
                nn.init.normal_(p.weight, mean=0.0, std=0.02)

        self.force_identical_output = getattr(config, "force_identical_output", False)
        self.quantizer = None
        
        # Tracking for monitoring
        self.register_buffer("usage_counts", torch.zeros(self.num_perceptrons, dtype=torch.float32))
        self.register_buffer("selection_counts", torch.zeros(self.num_perceptrons, dtype=torch.float32))

    def get_perceptron_stats(self):
        """Get statistics for monitoring"""
        total_usage = self.usage_counts.sum().item()
        total_selections = self.selection_counts.sum().item()
        
        if total_usage > 0:
            usage_percentages = (self.usage_counts / total_usage * 100).tolist()
        else:
            usage_percentages = [0.0] * self.num_perceptrons
            
        if total_selections > 0:
            selection_percentages = (self.selection_counts / total_selections * 100).tolist()
        else:
            selection_percentages = [0.0] * self.num_perceptrons
        
        return {
            'usage_counts': self.usage_counts.tolist(),
            'usage_percentages': usage_percentages,
            'selection_counts': self.selection_counts.tolist(),
            'selection_percentages': selection_percentages,
            'total_usage': total_usage,
            'total_selections': total_selections,
            'num_perceptrons': self.num_perceptrons,
            'hidden_size': self.hidden_size,  # ✅ FIXED: Now reports full size
            'force_identical_output': self.force_identical_output,
        }

    def forward(self, x, return_all_logits: bool = False, selected_indices: torch.Tensor | None = None):
        """
        x: [B, S, E] hidden states
        return_all_logits: if True, return [B, S, V, P] without selecting a head
        selected_indices: [B, S] head ids per (batch, pos); if given, we gather those heads
        """
        B, S, E = x.shape

        # ----- Optional "identical output" mode: average all heads -----
        if getattr(self, "force_identical_output", False):
            outs = [p(x) for p in self.perceptrons]           # each: [B, S, V]
            all_logits = torch.stack(outs, dim=-1)            # [B, S, V, P]
            avg_logits = all_logits.mean(dim=-1)              # [B, S, V]

            # track usage uniformly (every position counted for every head)
            if self.training:
                with torch.no_grad():
                    valid_positions = B * S
                    for i in range(self.num_perceptrons):
                        self.usage_counts[i] += valid_positions
            return avg_logits

        # ----- Compute all heads -----
        outs = []
        for p in self.perceptrons:
            if self.training or not hasattr(p, "quantized_weight"):
                out = p(x)                                    # [B, S, V]
            else:
                # inference-time quantized weight, restore afterward
                float_w = p.weight
                try:
                    p.weight = torch.nn.Parameter(p.quantized_weight, requires_grad=False)
                    out = p(x)
                finally:
                    p.weight = float_w
            outs.append(out)

        all_logits = torch.stack(outs, dim=-1)                # [B, S, V, P]

        # ----- Give trainer access to all heads when requested -----
        if return_all_logits:
            return all_logits

        # ----- Training second pass: respect trainer's chosen indices -----
        if selected_indices is not None:
            if selected_indices.dtype != torch.long:
                selected_indices = selected_indices.long()
            # gather chosen head per (b, s)
            gather_idx = selected_indices.unsqueeze(-1).unsqueeze(-1)          # [B, S, 1, 1]
            gather_idx = gather_idx.expand(-1, -1, all_logits.size(2), 1)      # [B, S, V, 1]
            selected_logits = torch.gather(all_logits, dim=3, index=gather_idx).squeeze(3)  # [B, S, V]

            # usage tracking
            if self.training:
                with torch.no_grad():
                    for i in range(self.num_perceptrons):
                        cnt = (selected_indices == i).sum().item()
                        self.usage_counts[i] += cnt
                        self.selection_counts[i] += cnt
            return selected_logits

        # ----- 🆕 Inference: Raw logits selection (NO z-score) -----
        # Direct softmax on raw logits (no z-score normalization)
        probs = F.softmax(all_logits.float(), dim=2)           # [B, S, V, P]
        conf, _ = probs.max(dim=2)                             # [B, S, P] - max prob per head
        best_idx = conf.argmax(dim=-1)                         # [B, S] - head with highest max prob

        # Gather selected logits from chosen heads
        gather_idx = best_idx.unsqueeze(2).unsqueeze(2).expand(-1, -1, all_logits.size(2), 1)
        selected_logits = torch.gather(all_logits, dim=3, index=gather_idx).squeeze(3)  # [B,S,V]
        return selected_logits
    
    def select_heads_like_training(self, all_logits: torch.Tensor) -> torch.Tensor:
        # all_logits: [B,T,V,P] or [B,V,P]
        import torch.nn.functional as F
        if all_logits.dim() == 3:
            all_logits = all_logits.unsqueeze(1)
        probs = F.softmax(all_logits.float(), dim=2)
        head_conf, _ = probs.max(dim=2)
        return head_conf.argmax(dim=-1)  # [B,T]

# 🧠 Custom GPT2 Model
class CustomMultiHeaderGPT2Model(nn.Module):
    def __init__(self, config, use_pretrained=True, pretrained_model="gpt2"):
        super().__init__()
        self.config = config
        
        if use_pretrained:
            print(f"🔄 Loading pretrained GPT2 model: {pretrained_model}")
            pretrained_gpt2 = GPT2LMHeadModel.from_pretrained(pretrained_model)
            
            # Extract transformer and LM head weights
            self.transformer = pretrained_gpt2.transformer
            pretrained_lm_weights = pretrained_gpt2.lm_head.weight.data
            
            print(f"✅ Loaded pretrained transformer")
            print(f"🔧 Extracted LM head weights: {pretrained_lm_weights.shape}")
            
            # Initialize custom multi-perceptron LM head
            self.lm_head = MultiPerceptronLMHead(
                config.n_embd, 
                config.vocab_size, 
                config,
                pretrained_weights=pretrained_lm_weights
            )
            
            del pretrained_gpt2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print("🆕 Creating GPT2 model from scratch")
            self.transformer = GPT2Model(config)
            self.lm_head = MultiPerceptronLMHead(config.n_embd, config.vocab_size, config)

        self.gate = GatingMLP(hidden_size=self.config.n_embd,
                      num_heads=self.config.n_lm_perceptrons,
                      gate_hidden=getattr(self.config, "gate_hidden", 256))


    def forward(
        self,
        input_ids,
        attention_mask=None,
        return_all_logits: bool = False,
        selected_indices=None,            # kept for trainer compatibility (ignored here)
        **kwargs
    ):
        # 1) Backbone forward
        out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = out.last_hidden_state            # [B,S,E]

        # 2) Compute ALL head logits once
        all_logits = self.lm_head(
            hidden_states,
            return_all_logits=True,                      # force all heads
            selected_indices=None
        )                                                # [B,S,V,P]

        if return_all_logits:
            return all_logits                            # [B,S,V,P]

        # 3) ALWAYS use gate for inference
        if not hasattr(self, "gate") or self.gate is None:
            raise RuntimeError("GatingMLP (self.gate) is missing. Instantiate it in CustomGPT2Model.__init__.")

        gate_logits = self.gate(hidden_states)          # [B,S,P]
        log_pi      = F.log_softmax(gate_logits, dim=2) # [B,S,P]
        log_probs_h = F.log_softmax(all_logits, dim=2)  # [B,S,V,P]

        # Mixture of experts in probability space, returned as log-probs
        log_mix = torch.logsumexp(log_pi.unsqueeze(2) + log_probs_h, dim=3)  # [B,S,V]

        # HuggingFace-compatible tiny wrapper
        class OutputWithLogits:
            def __init__(self, logits):
                self.logits = logits     # NOTE: these are log-probs, not raw logits
        return OutputWithLogits(log_mix)
    
    def get_lm_head_stats(self):
        return self.lm_head.get_perceptron_stats()
