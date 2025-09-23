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

# Dual Path GPT Model with Multi LM Headers - COMPLETE VERSION
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel
import math
import contextlib
import json

try:
    from multi_headers.multi_headers_model import GatingMLP as _GatingMLP
except Exception:
    import torch.nn as nn, torch.nn.functional as F
    class _GatingMLP(nn.Module):
        def __init__(self, in_dim, out_dim, hidden=256, dropout=0.0):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden, out_dim)
            self.drop = nn.Dropout(dropout)
        def forward(self, x):
            return self.fc2(self.drop(F.gelu(self.fc1(x))))

def _any_grad(module):
    try:
        return any(p.requires_grad for p in module.parameters())
    except Exception:
        return False


class MultiPerceptronLMHead(nn.Module):
    """Multi-perceptron LM head with multiple headers for path specialization"""
    def __init__(self, hidden_size, vocab_size, num_perceptrons=2, pretrained_weights=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_perceptrons = num_perceptrons
        
        # Create multiple linear perceptrons with FULL input size
        self.perceptrons = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False) for _ in range(num_perceptrons)
        ])
        
        # Initialize with pretrained weights if provided
        if pretrained_weights is not None:
            for i, p in enumerate(self.perceptrons):
                p.weight.data.copy_(pretrained_weights)
                if i > 0:
                    # Slightly diversify non-anchor heads
                    p.weight.data *= (1.0 - 0.01 * i)
            print(f"All {num_perceptrons} perceptrons initialized with pretrained weights")
        else:
            for p in self.perceptrons:
                nn.init.normal_(p.weight, mean=0.0, std=0.02)
        
        # Tracking for monitoring
        self.register_buffer("usage_counts", torch.zeros(num_perceptrons, dtype=torch.float32))
        self.register_buffer("selection_counts", torch.zeros(num_perceptrons, dtype=torch.float32))

    def forward(self, x, head_index=None, return_all_logits=False):
        """
        x: [B, S, E] hidden states
        head_index: which specific head to use (0 or 1)
        return_all_logits: if True, return [B, S, V, P]
        """
        if return_all_logits:
            outs = [p(x) for p in self.perceptrons]  # each: [B, S, V]
            return torch.stack(outs, dim=-1)         # [B, S, V, P]
        
        if head_index is not None:
            return self.perceptrons[head_index](x)   # [B, S, V]
        else:
            # Default to first head
            return self.perceptrons[0](x)

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
            'hidden_size': self.hidden_size,
        }


class DualPathMultiHeadersGPT2(nn.Module):
    """
    GPT-2 model with dual paths and multiple specialized LM headers
    - Left path: frozen baseline 
    - Right path: trainable
    - Multiple LM headers with per-path head allocation and gates
    """

    def __init__(self, config, pretrained_model="gpt2", split_at_layer=6,
                 gate_hidden=256, gate_temp=1.0, **kwargs):
        super().__init__()
        self.config = config
        self.split_at_layer = split_at_layer
        self.gate_temp = gate_temp
        self._pretrained_model = pretrained_model

        self.head_gate_temp = getattr(self, "head_gate_temp", 1.0)
        self.head_fast_k    = getattr(self, "head_fast_k", None)
        
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
        
        # Left path (frozen baseline)
        self.left_path = nn.ModuleList()
        for i in range(split_at_layer, len(hf_model.transformer.h)):
            self.left_path.append(hf_model.transformer.h[i])
        
        # Right path (trainable)
        self.right_path = nn.ModuleList()
        for i in range(split_at_layer, len(hf_model.transformer.h)):
            right_layer = type(hf_model.transformer.h[i])(config)
            right_layer.load_state_dict(hf_model.transformer.h[i].state_dict())
            self.right_path.append(right_layer)
        
        # Layer norms for each path
        self.ln_f_left = hf_model.transformer.ln_f  # frozen
        self.ln_f_right = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Initialize trainable layer norm from frozen one
        self.ln_f_right.load_state_dict(self.ln_f_left.state_dict())
        
        # Create specialized LM headers
        num_heads = int(getattr(config, "n_lm_perceptrons", 4))
        self.lm_head = MultiPerceptronLMHead(
            hidden_size=config.n_embd,
            vocab_size=config.vocab_size,
            num_perceptrons=num_heads,
            pretrained_weights=hf_model.lm_head.weight.data,
        )

        # === Per-path head allocation + small per-path head gates ===
        # Expect a dict like: {"left":[0,1], "right":[2,3]}
        self.head_allocation = kwargs.get("head_allocation", None)

        # one tiny gate per path, each gating only its subset of heads
        self._heads_gate_by_path = nn.ModuleDict()
        if self.head_allocation is not None:
            gate_hidden = int(gate_hidden)
            for pname, head_ids in self.head_allocation.items():
                self._heads_gate_by_path[pname] = _GatingMLP(config.n_embd, len(head_ids), gate_hidden)

        # toggle: actually use mixture over heads
        self.use_head_mixture = True
        
        # Path selection gate
        self.gate = nn.Sequential(
            nn.Linear(config.n_embd, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, 2)
        )
        
        # Initialize gate with slight bias toward first option (left path)
        with torch.no_grad():
            self.gate[-1].bias.fill_(0.25)
        
        # Freeze frozen paths
        self._freeze_frozen_paths()
        self._assert_frozen_paths()
        
        # Set attention implementation
        try:
            attn_impl = getattr(self.config, "_attn_implementation", None)
        except AttributeError:
            attn_impl = None
        if attn_impl is None:
            self.config._attn_implementation = "eager"
        
        # Align all blocks to same attention implementation
        all_blocks = (list(self.shared_layers) + list(self.left_path) + list(self.right_path))
        
        for blk in all_blocks:
            if hasattr(blk, "config"):
                blk.config._attn_implementation = self.config._attn_implementation
            if hasattr(blk, "attn") and hasattr(blk.attn, "config"):
                blk.attn.config._attn_implementation = self.config._attn_implementation
        
        # Clean up
        del hf_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Dual path model with multi headers initialized:")
        print(f"  Shared layers (0-{split_at_layer-1}): {len(self.shared_layers)}")
        print(f"  Path layers ({split_at_layer}-end): {len(self.left_path)}")
        print(f"  Frozen path: left")
        print(f"  Trainable path: right")
        print(f"  LM Headers: {self.lm_head.num_perceptrons} perceptrons "
              f"(allocated per path via head_allocation)")

    def freeze_split_gates(self, freeze: bool = True):
        for p in self.gate.parameters():
            p.requires_grad = not freeze

    def set_head_gate_hparams(self, fast_k=None, temp: float = 1.0):
        self.head_fast_k = fast_k
        self.head_gate_temp = temp

    def set_path_freezing(self, freeze_config: dict):
        """
        freeze_config keys (booleans, all optional):
        shared, left_path, right_path, gate, lm_headers
        """

        def _freeze_module(mod, freeze: bool):
            if mod is None:
                return
            # Handle ModuleList vs single Module transparently
            if isinstance(mod, nn.ModuleList):
                for m in mod:
                    for p in m.parameters():
                        p.requires_grad = not freeze
            else:
                for p in mod.parameters():
                    p.requires_grad = not freeze

        # Map high-level names to the modules
        groups = {
            "shared": [getattr(self, "shared_layers", None),
                      getattr(self, "wte", None),
                      getattr(self, "wpe", None)],
            "left_path": [getattr(self, "left_path", None),
                         getattr(self, "ln_f_left", None)],
            "right_path": [getattr(self, "right_path", None),
                          getattr(self, "ln_f_right", None)],
        }

        # Apply freezes for path groups
        for name, mods in groups.items():
            if name in freeze_config:
                for mod in mods:
                    _freeze_module(mod, bool(freeze_config[name]))

        # Gate control
        if "gate" in freeze_config:
            _freeze_module(getattr(self, "gate", None), bool(freeze_config["gate"]))

        # LM headers control (specific perceptrons)
        if "lm_headers" in freeze_config:
            lm_config = freeze_config["lm_headers"]
            if isinstance(lm_config, dict):
                # e.g., {"0": True, "1": False} to freeze head 0, unfreeze head 1
                for head_idx, should_freeze in lm_config.items():
                    idx = int(head_idx)
                    if 0 <= idx < len(self.lm_head.perceptrons):
                        _freeze_module(self.lm_head.perceptrons[idx], bool(should_freeze))
            elif isinstance(lm_config, bool):
                # Freeze/unfreeze all headers
                _freeze_module(self.lm_head, bool(lm_config))

    def _freeze_frozen_paths(self):
        """Freeze the baseline paths"""
        # Freeze shared layers
        for param in self.shared_layers.parameters():
            param.requires_grad = False
        
        # Freeze embeddings
        for param in self.wte.parameters():
            param.requires_grad = False
        for param in self.wpe.parameters():
            param.requires_grad = False
        
        # Freeze left path (baseline)
        for param in self.left_path.parameters():
            param.requires_grad = False
        for param in self.ln_f_left.parameters():
            param.requires_grad = False
            
        print("Frozen paths: shared + left_path")

    def _assert_frozen_paths(self):
        """Assert that frozen paths are actually frozen"""
        def _all_frozen(mod):
            return all(not p.requires_grad for p in mod.parameters())
        
        assert _all_frozen(self.shared_layers), "Shared layers must be frozen!"
        assert _all_frozen(self.left_path), "Left path must be frozen!"
        assert _all_frozen(self.ln_f_left), "Left LN must be frozen!"
        assert _all_frozen(self.wte) and _all_frozen(self.wpe), "Embeddings must be frozen!"

    def _expand_attn_mask(self, attention_mask, dtype, tgt_len=None):
        """Expand attention mask for transformer layers"""
        if attention_mask is None:
            return None
        bsz, src_len = attention_mask.shape
        if tgt_len is None:
            tgt_len = src_len
        mask = attention_mask[:, None, None, :].to(dtype=dtype)
        mask = (1.0 - mask) * -1e4
        return mask

    def get_embeddings(self, input_ids, attention_mask=None):
        """Get initial embeddings (shared)"""
        batch_size, seq_len = input_ids.shape
        
        token_embeddings = self.wte(input_ids)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.wpe(position_ids)
        
        return token_embeddings + position_embeddings

    def forward_shared_layers(self, hidden_states, attention_mask):
        """Forward through shared layers (0 to split-1)"""
        attn = self._expand_attn_mask(attention_mask, hidden_states.dtype, tgt_len=hidden_states.size(1))
        for layer in self.shared_layers:
            hidden_states = layer(hidden_states, attention_mask=attn, use_cache=False)[0]
        return hidden_states

    def forward_path(self, hidden_states, attention_mask, path="left"):
        """Forward through path layers and return hidden states (before LM head)"""
        attn = self._expand_attn_mask(attention_mask, hidden_states.dtype, tgt_len=hidden_states.size(1))
        
        if path == "left":
            layers = self.left_path
            ln = self.ln_f_left
        else:  # right
            layers = self.right_path
            ln = self.ln_f_right
        
        for layer in layers:
            hidden_states = layer(hidden_states, attention_mask=attn, use_cache=False)[0]
        hidden_states = ln(hidden_states)
        return hidden_states

    def _gate_weights(self, hidden_states, hard=False):
        """Compute gate weights"""
        logits = self.gate(hidden_states)  # [B,S,2]
        if hard:
            soft = F.softmax(logits / max(self.gate_temp, 1e-6), dim=-1)
            idx = torch.argmax(soft, dim=-1)
            hard_onehot = F.one_hot(idx, num_classes=2).to(soft.dtype)
            gate = hard_onehot + (soft - soft.detach())  # STE
            return gate, logits
        else:
            gate = F.softmax(logits / max(self.gate_temp, 1e-6), dim=-1)
            return gate, logits

    def _mix_heads_logprobs_subset(self, x, head_ids, gate_module, fast_k=None, temp=None):
        """
        x:         [B,S,E]
        head_ids:  list[int]  — the subset of global head indices allocated to this path
        gate_module(x):       [B,S,K]  where K == len(head_ids)
        Returns:   [B,S,V]    log-probabilities mixed over the selected heads
        """
        # --- config (separate temp for LM-head gates) ---
        if temp is None:
            temp = getattr(self, "head_gate_temp", 1.0)
        if fast_k is None:
            fast_k = getattr(self, "head_fast_k", None)

        K = len(head_ids)
        assert K > 0, "head_ids must be non-empty"

        # --- per-head logits, stacked as [B,S,V,K] ---
        outs = [self.lm_head.perceptrons[i](x) for i in head_ids]     # each [B,S,V]
        all_logits = torch.stack(outs, dim=-1)                         # [B,S,V,K]

        # convert to log-probs per head in fp32 for stability
        logp_h = F.log_softmax(all_logits.float(), dim=2)              # [B,S,V,K]

        # --- gate over heads (soft) ---
        pi_logits = gate_module(x).float()                             # [B,S,K]
        if pi_logits.size(-1) != K:
            raise ValueError(f"gate_module produced K={pi_logits.size(-1)} but head_ids has {K}")

        # optional fast-k: mix only top-k heads (by gate score)
        if fast_k is not None and 0 < fast_k < K:
            top_vals, top_idx = torch.topk(pi_logits, k=fast_k, dim=-1)                  # [B,S,k]
            log_pi = F.log_softmax(top_vals / temp, dim=-1)                              # [B,S,k]

            # gather the corresponding head log-probs: [B,S,V,K] -> [B,S,V,k]
            gather_idx = top_idx.unsqueeze(2).expand(-1, -1, logp_h.size(2), -1)         # [B,S,V,k]
            sel_logp_h = torch.gather(logp_h, dim=-1, index=gather_idx)                  # [B,S,V,k]

            final_log_probs = torch.logsumexp(sel_logp_h + log_pi.unsqueeze(2), dim=-1)  # [B,S,V]
        else:
            log_pi = F.log_softmax(pi_logits / temp, dim=-1)                              # [B,S,K]
            final_log_probs = torch.logsumexp(logp_h + log_pi.unsqueeze(2), dim=-1)       # [B,S,V]

        return final_log_probs

    def _as_log_probs(self, x):
        # Detect if `x` are already log-probs: logsumexp ~ 0
        lse = torch.logsumexp(x.float(), dim=-1)              # [..., V] -> [...]
        is_logprob = lse.detach().median().abs() < 1e-3
        return x if is_logprob else F.log_softmax(x, dim=-1)

    def _logspace_mix(self, left_logits, right_logits, w_left, w_right):
        # Convert to log-probs if needed
        left_lp = self._as_log_probs(left_logits)
        right_lp = self._as_log_probs(right_logits)
        
        # weights are [B,S,1]; clamp and log
        eps = 1e-9
        l_w_left = (w_left.clamp_min(eps)).log()
        l_w_right = (w_right.clamp_min(eps)).log()
        
        # log-sum-exp mix: returns log-probs [B,S,V]
        stacked = torch.stack([left_lp + l_w_left, right_lp + l_w_right], dim=0)
        return torch.logsumexp(stacked, dim=0)

    def forward(self, input_ids, attention_mask=None, labels=None,
            return_both_paths=False, path_selection="gate_soft",
            return_all_logits=False, return_head_indices: bool=False):

        # ===== memory saver contexts =====
        use_no_grad = getattr(self, "backbone_is_frozen", False)
        ng = torch.no_grad if use_no_grad else contextlib.nullcontext

        # detect split-gate freeze from params (expects module-level _any_grad)
        split_gates_trainable = _any_grad(self.gate)
        gg = contextlib.nullcontext if split_gates_trainable else torch.no_grad

        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # ===== 1) Embeddings + shared stack =====
        with ng():
            hidden_states = self.get_embeddings(input_ids, attention_mask)
            shared_output = self.forward_shared_layers(hidden_states, attention_mask)

        # ===== 2) Path processing =====
        left_hidden = shared_output.clone()
        right_hidden = shared_output.clone()

        # ===== 3) Per-path scores (LM-head mixtures) =====
        path_logits = {}
        head_top_idx = {} if return_head_indices else None
        per_path_local_head_wins = {}
        head_gate_means = {}  # <— NEW: expose per-path mean head probs (for trainer regs)

        def _per_path_head_argmax(gate_module, x_path, path_name):
            """Return global head indices [B,S] chosen by the per-path head gate."""
            pi_logits = gate_module(x_path).float()                            # [B,S,K]
            temp = getattr(self, "head_gate_temp", 1.0)
            fast_k = getattr(self, "head_fast_k", None)
            if fast_k is not None and 0 < fast_k < pi_logits.size(-1):
                top_vals, top_idx = torch.topk(pi_logits, k=fast_k, dim=-1)    # [B,S,k]
                local = top_idx[..., 0]                                        # best of top-k
                global_ids = torch.tensor(self.head_allocation[path_name], device=local.device)
                return global_ids[local]
            else:
                pi = torch.softmax(pi_logits / temp, dim=-1)                   # [B,S,K]
                local = pi.argmax(dim=-1)                                      # [B,S]
                global_ids = torch.tensor(self.head_allocation[path_name], device=local.device)
                return global_ids[local]

        # Process each path
        for path_name in ["left", "right"]:
            path_hidden = left_hidden if path_name == "left" else right_hidden
            use_subset = (self.head_allocation is not None) and (path_name in self.head_allocation)

            if self.use_head_mixture and use_subset:
                gate = self._heads_gate_by_path[path_name]

                # run path transformer WITHOUT grad if backbone is frozen
                with ng():
                    path_output = self.forward_path(path_hidden, attention_mask, path_name)

                # --- NEW: compute head-gate probs (no no_grad; we want grads for regs)
                pi_logits = gate(path_output).float()                           # [B,S,K]
                temp = getattr(self, "head_gate_temp", 1.0)
                pi = torch.softmax(pi_logits / temp, dim=-1)                    # [B,S,K]

                # head-gates + LM heads stay trainable
                path_logits[path_name] = self._mix_heads_logprobs_subset(
                    path_output,
                    self.head_allocation[path_name],
                    gate,
                    fast_k=self.head_fast_k,
                    temp=self.head_gate_temp,
                )

                # --- NEW: store mean head probs as a global-size vector (for this path)
                p_bar = pi.mean(dim=(0, 1))                                     # [K]
                global_ids = torch.tensor(self.head_allocation[path_name],
                                        device=p_bar.device, dtype=torch.long)
                g = torch.zeros(self.lm_head.num_perceptrons,
                                device=p_bar.device, dtype=p_bar.dtype)
                g.scatter_(0, global_ids, p_bar)                                 # zeros outside this path
                head_gate_means[path_name] = g                                   # [num_heads]

                if return_head_indices:
                    head_top_idx[path_name] = _per_path_head_argmax(gate, path_output, path_name)

                if self.training:
                    with torch.no_grad():
                        per_path_local_head_wins[path_name] = _per_path_head_argmax(gate, path_output, path_name)

            else:
                # Fallback: 2 fixed heads (left→0, right→1)
                with ng():
                    path_output = self.forward_path(path_hidden, attention_mask, path_name)

                head_idx = 0 if path_name == "left" else 1
                path_logits[path_name] = self.lm_head(path_output, head_index=head_idx)

                if return_head_indices:
                    head_top_idx[path_name] = torch.full(
                        (batch_size, seq_len), head_idx, dtype=torch.long, device=path_output.device
                    )
                if self.training:
                    with torch.no_grad():
                        per_path_local_head_wins[path_name] = torch.full(
                            (batch_size, seq_len), head_idx, dtype=torch.long, device=path_output.device
                        )

        if return_both_paths:
            return path_logits["left"], path_logits["right"]

        if return_all_logits:
            return torch.stack([path_logits["left"], path_logits["right"]], dim=-1)  # [B,S,V,2]

        # ===== 4) Path selection & mixing =====
        used_log_probs = False
        active_path_idx = None  # [B,S]
        gate_info = None

        if path_selection in ("gate_soft", "gate_hard"):
            hard = (path_selection == "gate_hard")

            # split gates: wrap with gg() so no graph if frozen
            with gg():
                gate_weights, gate_logits = self._gate_weights(shared_output, hard=hard)

            w_left = gate_weights[..., 0].unsqueeze(-1)
            w_right = gate_weights[..., 1].unsqueeze(-1)

            final_log_probs = self._logspace_mix(path_logits["left"], path_logits["right"], w_left, w_right)
            used_log_probs = True

            weights2 = torch.stack([w_left, w_right], dim=-1).squeeze(-2)  # [B,S,2]
            active_path_idx = torch.argmax(weights2, dim=-1)               # [B,S]

            gate_info = {
                "gate": gate_weights,
                "gate_logits": gate_logits,
                "final_weights": {"left": w_left, "right": w_right},
            }

        elif path_selection == "left_only":
            final_logits = path_logits["left"]
            used_log_probs = False
            one = torch.ones((batch_size, seq_len, 1), device=final_logits.device)
            zero = torch.zeros_like(one)
            gate_info = {
                "final_weights": {"left": one, "right": zero},
                "mode": "forced_left",
                "gate": torch.stack([one.squeeze(-1), zero.squeeze(-1)], dim=-1),  # [B,S,2]
                "gate_logits": None,
            }
            active_path_idx = torch.zeros((batch_size, seq_len), dtype=torch.long, device=final_logits.device)

        elif path_selection == "right_only":
            final_logits = path_logits["right"]
            used_log_probs = False
            one = torch.ones((batch_size, seq_len, 1), device=final_logits.device)
            zero = torch.zeros_like(one)
            gate_info = {
                "final_weights": {"left": zero, "right": one},
                "mode": "forced_right",
                "gate": torch.stack([zero.squeeze(-1), one.squeeze(-1)], dim=-1),  # [B,S,2]
                "gate_logits": None,
            }
            active_path_idx = torch.ones((batch_size, seq_len), dtype=torch.long, device=final_logits.device)

        elif path_selection == "max_prob":
            left_lp = self._as_log_probs(path_logits["left"])
            right_lp = self._as_log_probs(path_logits["right"])

            left_max = left_lp.exp().max(dim=-1).values
            right_max = right_lp.exp().max(dim=-1).values

            best_idx = (right_max > left_max).long()  # [B,S]

            final_logits = torch.where(best_idx.unsqueeze(-1), path_logits["right"], path_logits["left"])
            gate_info = {"best_path_indices": best_idx}
            active_path_idx = best_idx

        else:
            raise ValueError(f"Unknown path_selection: {path_selection}")

        if not used_log_probs:
            x = final_logits
            lse = torch.logsumexp(x.float(), dim=-1)
            final_log_probs = x if (lse.detach().median().abs() < 1e-3) else F.log_softmax(x, dim=-1)
            used_log_probs = True

        # ===== 5) Loss (mask-aware) =====
        loss = None
        if labels is not None:
            gold = labels[..., 1:].contiguous()                      # [B,S-1]
            mask = (attention_mask[..., 1:].to(torch.float)
                    if attention_mask is not None else torch.ones_like(gold, dtype=torch.float))
            if (gold == -100).any():
                mask = mask * (gold != -100).to(mask.dtype)
                gold = gold.masked_fill(gold == -100, 0)

            logp = final_log_probs[..., :-1, :].contiguous()         # [B,S-1,V]
            token_nll = -logp.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
            token_nll = token_nll * mask
            denom = mask.sum().clamp_min(1.0)
            loss = token_nll.sum() / denom

        # ===== 6) Build a combined head index for logging =====
        head_top_idx_combined = None
        with torch.no_grad():
            if active_path_idx is not None and per_path_local_head_wins:
                dev = active_path_idx.device
                head_top_idx_combined = torch.zeros((batch_size, seq_len), dtype=torch.long, device=dev)
                path_names = ["left", "right"]
                for i, path_name in enumerate(path_names):
                    if path_name in per_path_local_head_wins:
                        head_top_idx_combined = torch.where(
                            active_path_idx == i, per_path_local_head_wins[path_name], head_top_idx_combined
                        )

        returned_head_top_idx = head_top_idx if return_head_indices else None
        returned_head_top_idx_combined = head_top_idx_combined if return_head_indices else None

        # ===== 7) ADD USAGE TRACKING =====
        if self.training:
            with torch.no_grad():
                L = seq_len
                if attention_mask is not None:
                    valid_mask = attention_mask[..., 1:].contiguous().bool()
                else:
                    valid_mask = torch.ones_like(input_ids[..., 1:], dtype=torch.bool)

                def _align_to_targets(x: torch.Tensor) -> torch.Tensor:
                    return x[..., 1:] if x.size(-1) == L else x[..., : L-1]

                if self.use_head_mixture and head_top_idx_combined is not None:
                    hidx_aligned = _align_to_targets(head_top_idx_combined)
                    vm = valid_mask
                    for head_id in range(self.lm_head.num_perceptrons):
                        cnt = int(((hidx_aligned == head_id) & vm).sum().item())
                        self.lm_head.usage_counts[head_id] += cnt
                        self.lm_head.selection_counts[head_id] += cnt
                else:
                    if active_path_idx is not None:
                        api = _align_to_targets(active_path_idx)
                        vm = valid_mask
                        path_names = ["left", "right"]
                        for i, path_name in enumerate(path_names):
                            path_mask = (api == i) & vm
                            ntok = int(path_mask.sum().item())
                            if ntok <= 0:
                                continue
                            if self.head_allocation and path_name in self.head_allocation:
                                heads = self.head_allocation[path_name]
                                if len(heads) == 1:
                                    hid = heads[0]
                                    self.lm_head.usage_counts[hid] += ntok
                                    self.lm_head.selection_counts[hid] += ntok
                                else:
                                    q, r = divmod(ntok, len(heads))
                                    for j, hid in enumerate(heads):
                                        add = q + (1 if j < r else 0)
                                        self.lm_head.usage_counts[hid] += add
                                        self.lm_head.selection_counts[hid] += add
                            else:
                                hid = 0 if path_name == "left" else 1
                                self.lm_head.usage_counts[hid] += ntok
                                self.lm_head.selection_counts[hid] += ntok

        return {
            "loss": loss,
            "logits": final_log_probs,       # normalized scores (log-probs)
            "log_probs": final_log_probs,
            "path_logits": path_logits,
            "gate": gate_info,
            "head_top_idx": returned_head_top_idx,
            "head_top_idx_combined": returned_head_top_idx_combined,
            "active_path_idx": active_path_idx,
            "head_gate_means": head_gate_means,  # <— NEW
        }

    def get_trainable_parameters(self):
        """Get list of trainable parameters"""
        params = []
        # split router (path selection gate)
        params += list(self.gate.parameters())
        # per-path head gates (the small MLPs over each subset)
        params += list(self._heads_gate_by_path.parameters())
        # right branch + its LN (left is frozen)
        params += list(self.right_path.parameters())
        params += list(self.ln_f_right.parameters())
        # LM heads: keep head 0 frozen as anchor; train all others (1..N-1)
        for i, head in enumerate(self.lm_head.perceptrons):
            if i == 0:
                continue  # Skip head 0 (anchor)
            params += list(head.parameters())
        return [p for p in params if p.requires_grad]

    def get_parameter_count(self):
        """Get parameter counts for analysis"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        # LM head breakdown
        lm_head_total = sum(p.numel() for p in self.lm_head.parameters())
        lm_head_trainable = sum(p.numel() for p in self.lm_head.parameters() if p.requires_grad)
        lm_head_frozen = lm_head_total - lm_head_trainable
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'trainable_ratio': trainable_params / total_params,
            'lm_head_total': lm_head_total,
            'lm_head_trainable': lm_head_trainable,
            'lm_head_frozen': lm_head_frozen,
        }
    
    def get_lm_head_stats(self):
        """Get LM head statistics (compatibility with multi_headers_model)"""
        return self.lm_head.get_perceptron_stats()

    def print_model_info(self):
        """Print detailed model information"""
        print(f"\nDual Path Multi Headers Model Info:")
        print(f"  Architecture: GPT-2 with dual paths + {self.lm_head.num_perceptrons} LM headers")
        print(f"  Split layer: {self.split_at_layer}")
        print(f"  Shared layers: 0-{self.split_at_layer-1} (frozen)")
        print(f"  Left path: {self.split_at_layer}-end (frozen baseline)")
        print(f"  Right path: {self.split_at_layer}-end (trainable)")
        
        if self.head_allocation:
            print(f"  Head allocation:")
            for path, heads in self.head_allocation.items():
                print(f"    {path}: heads {heads}")
        
        print(f"  Use head mixture: {self.use_head_mixture}")
        print(f"  Head gate temp: {getattr(self, 'head_gate_temp', 'N/A')}")
        print(f"  Head fast-k: {getattr(self, 'head_fast_k', 'N/A')}")
        
        # Parameter statistics
        param_stats = self.get_parameter_count()
        print(f"\nParameter Statistics:")
        print(f"  Total: {param_stats['total']:,}")
        print(f"  Trainable: {param_stats['trainable']:,} ({param_stats['trainable_ratio']:.1%})")
        print(f"  Frozen: {param_stats['frozen']:,}")
        print(f"  LM heads: {param_stats['lm_head_trainable']:,} trainable / {param_stats['lm_head_total']:,} total")

    def save_model_config(self, filepath):
        """Save model configuration to JSON"""
        config = {
            "model_type": "DualPathMultiHeadersGPT2",
            "pretrained_model": getattr(self, "_pretrained_model", "gpt2"),
            "split_at_layer": self.split_at_layer,
            "gate_temp": self.gate_temp,
            "head_gate_temp": getattr(self, "head_gate_temp", 1.0),
            "head_fast_k": getattr(self, "head_fast_k", None),
            "use_head_mixture": getattr(self, "use_head_mixture", False),
            "head_allocation": self.head_allocation,
            "n_lm_perceptrons": self.lm_head.num_perceptrons,
            "vocab_size": self.config.vocab_size,
            "n_embd": self.config.n_embd,
            "n_layer": self.config.n_layer,
            "n_head": self.config.n_head,
            "parameter_count": self.get_parameter_count(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model config saved to: {filepath}")

    @classmethod
    def from_config_file(cls, config_path, checkpoint_path=None):
        """Load model from config file and optionally load weights"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create GPT2Config
        config = GPT2Config(
            vocab_size=config_dict['vocab_size'],
            n_embd=config_dict['n_embd'],
            n_layer=config_dict['n_layer'],
            n_head=config_dict['n_head'],
        )
        config.n_lm_perceptrons = config_dict['n_lm_perceptrons']
        
        # Create model
        model = cls(
            config=config,
            pretrained_model=config_dict.get('pretrained_model', 'gpt2'),
            split_at_layer=config_dict['split_at_layer'],
            head_allocation=config_dict.get('head_allocation'),
            gate_temp=config_dict.get('gate_temp', 1.0),
        )
        
        # Set additional attributes
        model.use_head_mixture = config_dict.get('use_head_mixture', False)
        model.head_gate_temp = config_dict.get('head_gate_temp', 1.0)
        model.head_fast_k = config_dict.get('head_fast_k', None)
        
        # Load checkpoint if provided
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded weights from: {checkpoint_path}")
        
        return model

    def freeze_backbone_for_inference(self):
        """Freeze all transformer layers for memory-efficient inference"""
        self.backbone_is_frozen = True
        print("Backbone frozen for inference (memory optimization enabled)")

    def unfreeze_backbone_for_training(self):
        """Unfreeze trainable parts for training"""
        self.backbone_is_frozen = False
        print("Backbone unfrozen for training")