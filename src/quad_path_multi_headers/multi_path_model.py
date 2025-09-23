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

# Multi-Path GPT Model with 2 LM Headers (Fixed)
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel
import math
import contextlib

try:
    from multi_headers_model import GatingMLP as _GatingMLP
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

class MultiPerceptronLMHead(nn.Module):
    """Multi-perceptron LM head with 2 headers for path specialization"""
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

        #print(f"get_perceptron_stats: '{total_usage}'") 
        
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


class HierarchicalMultiPathGPT2(nn.Module):
    """
    GPT-2 model with hierarchical splits and 4 specialized LM headers:
    - First split at layer 6: left (frozen baseline) vs right (trainable)
    - Second split at layer 9: each path splits into two more paths  
    - Final paths: left_left, left_right, right_left, right_right
    """

    def __init__(self, config, pretrained_model="gpt2",
                 split_at_layer_1=6, split_at_layer_2=9,
                 gate_hidden=256, gate_temp=1.0, **kwargs):
        super().__init__()
        self.config = config
        self.split_at_layer_1 = split_at_layer_1  # First split (6)
        self.split_at_layer_2 = split_at_layer_2  # Second split (9)
        self.gate_temp = gate_temp

        self.head_gate_temp = getattr(self, "head_gate_temp", 1.0)
        self.head_fast_k    = getattr(self, "head_fast_k", None)

        assert split_at_layer_2 > split_at_layer_1, "Second split must be after first split"
        
        # Load pretrained model for initialization
        print(f"Loading pretrained model: {pretrained_model}")
        hf_model = GPT2LMHeadModel.from_pretrained(pretrained_model)
        
        # Shared layers (0 to split_at_layer_1-1)
        self.shared_layers = nn.ModuleList()
        for i in range(split_at_layer_1):
            self.shared_layers.append(hf_model.transformer.h[i])
        
        # Copy other transformer components
        self.wte = hf_model.transformer.wte  # token embeddings
        self.wpe = hf_model.transformer.wpe  # position embeddings
        
        # Intermediate layers (split_at_layer_1 to split_at_layer_2-1)
        # Left intermediate (frozen)
        self.left_intermediate = nn.ModuleList()
        for i in range(split_at_layer_1, split_at_layer_2):
            self.left_intermediate.append(hf_model.transformer.h[i])
        
        # Right intermediate (trainable)
        self.right_intermediate = nn.ModuleList()
        for i in range(split_at_layer_1, split_at_layer_2):
            right_layer = type(hf_model.transformer.h[i])(config)
            right_layer.load_state_dict(hf_model.transformer.h[i].state_dict())
            self.right_intermediate.append(right_layer)
        
        # Final paths (split_at_layer_2 to end)
        # Path 0: left_left (frozen)
        self.path_left_left = nn.ModuleList()
        for i in range(split_at_layer_2, len(hf_model.transformer.h)):
            self.path_left_left.append(hf_model.transformer.h[i])
        
        # Path 1: left_right (trainable - diverges from left at layer 9)
        self.path_left_right = nn.ModuleList()
        for i in range(split_at_layer_2, len(hf_model.transformer.h)):
            layer = type(hf_model.transformer.h[i])(config)
            layer.load_state_dict(hf_model.transformer.h[i].state_dict())
            self.path_left_right.append(layer)
        
        # Path 2: right_left (trainable - diverges from right at layer 9)
        self.path_right_left = nn.ModuleList()
        for i in range(split_at_layer_2, len(hf_model.transformer.h)):
            layer = type(hf_model.transformer.h[i])(config)
            layer.load_state_dict(hf_model.transformer.h[i].state_dict())
            self.path_right_left.append(layer)
        
        # Path 3: right_right (trainable)
        self.path_right_right = nn.ModuleList()
        for i in range(split_at_layer_2, len(hf_model.transformer.h)):
            layer = type(hf_model.transformer.h[i])(config)
            layer.load_state_dict(hf_model.transformer.h[i].state_dict())
            self.path_right_right.append(layer)
        
        # Layer norms for each final path
        self.ln_f_left_left = hf_model.transformer.ln_f  # frozen
        self.ln_f_left_right = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_f_right_left = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_f_right_right = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Initialize trainable layer norms from frozen one
        for ln in [self.ln_f_left_right, self.ln_f_right_left, self.ln_f_right_right]:
            ln.load_state_dict(self.ln_f_left_left.state_dict())
        
        # Create specialized LM headers
        num_heads = int(getattr(config, "n_lm_perceptrons", 8))
        self.lm_head = MultiPerceptronLMHead(
            hidden_size=config.n_embd,
            vocab_size=config.vocab_size,
            num_perceptrons=num_heads,
            pretrained_weights=hf_model.lm_head.weight.data,
        )

        # === Per-path head allocation + small per-path head gates ===
        # Expect a dict like: {"left_left":[0,1,2], "left_right":[3], "right_left":[4,5], "right_right":[6,7]}
        self.head_allocation = kwargs.get("head_allocation", None)

        # one tiny gate per path, each gating only its subset of heads
        self._heads_gate_by_path = nn.ModuleDict()
        if self.head_allocation is not None:
            gate_hidden = int(gate_hidden)  # reuse your gate_hidden arg
            for pname, head_ids in self.head_allocation.items():
                self._heads_gate_by_path[pname] = _GatingMLP(config.n_embd, len(head_ids), gate_hidden)

        # toggle: actually use mixture over heads (you'll call this in forward)
        self.use_head_mixture = True
        
        # Gates
        # First gate at layer 6 output (left vs right)
        self.gate_1 = nn.Sequential(
            nn.Linear(config.n_embd, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, 2)
        )
        
        # Second gates at layer 9 output (for each intermediate path)
        self.gate_2_left = nn.Sequential(
            nn.Linear(config.n_embd, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, 2)
        )
        
        self.gate_2_right = nn.Sequential(
            nn.Linear(config.n_embd, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, 2)
        )
        
        # Initialize gates with slight bias toward first option
        for gate in [self.gate_1, self.gate_2_left, self.gate_2_right]:
            with torch.no_grad():
                gate[-1].bias.fill_(0.25)
        
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
        all_blocks = (list(self.shared_layers) + list(self.left_intermediate) + 
                     list(self.right_intermediate) + list(self.path_left_left) + 
                     list(self.path_left_right) + list(self.path_right_left) + 
                     list(self.path_right_right))
        
        for blk in all_blocks:
            if hasattr(blk, "config"):
                blk.config._attn_implementation = self.config._attn_implementation
            if hasattr(blk, "attn") and hasattr(blk.attn, "config"):
                blk.attn.config._attn_implementation = self.config._attn_implementation
        
        # Clean up
        del hf_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Quad path model initialized:")
        print(f"  Shared layers (0-{split_at_layer_1-1}): {len(self.shared_layers)}")
        print(f"  Intermediate layers ({split_at_layer_1}-{split_at_layer_2-1}): {len(self.left_intermediate)}")
        print(f"  Final paths ({split_at_layer_2}-end): {len(self.path_left_left)}")
        print(f"  Frozen paths: left_left + left_intermediate + shared")
        print(f"  Trainable paths: left_right, right_left, right_right + right_intermediate")
        print(f"  LM Headers: {self.lm_head.num_perceptrons} perceptrons "
            f"(allocated per leaf via head_allocation)")

    def freeze_split_gates(self, freeze: bool = True):
        for g in (self.gate_1, self.gate_2_left, self.gate_2_right):
            for p in g.parameters():
                p.requires_grad = not freeze

    def set_head_gate_hparams(self, fast_k=None, temp: float = 1.0):
        self.head_fast_k = fast_k
        self.head_gate_temp = temp

    def set_path_freezing(self, freeze_config: dict):
        """
        freeze_config keys (booleans, all optional):
        shared, left_intermediate, right_intermediate,
        left_right, right_left, right_right,
        gate1, gate2_left, gate2_right, gates  (legacy: all gates)
        lm_headers: freeze specific LM header perceptrons
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

        # Map high-level names to the modules commonly used in this quad model
        groups = {
            # pre-split (0..s1-1)
            "shared": [getattr(self, "shared_layers", None),
                    getattr(self, "wte", None),
                    getattr(self, "wpe", None)],

            # intermediates (s1..s2-1)
            "left_intermediate":  [getattr(self, "left_intermediate", None)],
            "right_intermediate": [getattr(self, "right_intermediate", None)],

            # deep leaves (s2..end) + their layer norms
            "left_right":  [getattr(self, "path_left_right", None),
                            getattr(self, "ln_f_left_right", None)],
            "right_left":  [getattr(self, "path_right_left", None),
                            getattr(self, "ln_f_right_left", None)],
            "right_right": [getattr(self, "path_right_right", None),
                            getattr(self, "ln_f_right_right", None)],
        }

        # Apply freezes for path groups
        for name, mods in groups.items():
            if name in freeze_config:
                for mod in mods:
                    _freeze_module(mod, bool(freeze_config[name]))

        # Per-gate control
        if "gate1" in freeze_config:
            _freeze_module(getattr(self, "gate_1", None), bool(freeze_config["gate1"]))
        if "gate2_left" in freeze_config:
            _freeze_module(getattr(self, "gate_2_left", None), bool(freeze_config["gate2_left"]))
        if "gate2_right" in freeze_config:
            _freeze_module(getattr(self, "gate_2_right", None), bool(freeze_config["gate2_right"]))

        # Legacy: "gates" freezes all gates at once
        if "gates" in freeze_config:
            freeze = bool(freeze_config["gates"])
            for g in (getattr(self, "gate_1", None),
                    getattr(self, "gate_2_left", None),
                    getattr(self, "gate_2_right", None)):
                _freeze_module(g, freeze)

        # LM headers control (specific perceptrons) - now only 0 and 1
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
        
        # Freeze left intermediate (baseline path)
        for param in self.left_intermediate.parameters():
            param.requires_grad = False
        
        # Freeze left_left path (fully baseline)
        for param in self.path_left_left.parameters():
            param.requires_grad = False
        for param in self.ln_f_left_left.parameters():
            param.requires_grad = False
        
        # Freeze LM head perceptron 0 (left paths anchor)
        #for param in self.lm_head.perceptrons[0].parameters():
        #    param.requires_grad = False
            
        print("Frozen paths: shared + left_intermediate + left_left")

    def _assert_frozen_paths(self):
        """Assert that frozen paths are actually frozen"""
        def _all_frozen(mod):
            return all(not p.requires_grad for p in mod.parameters())
        
        assert _all_frozen(self.shared_layers), "Shared layers must be frozen!"
        assert _all_frozen(self.left_intermediate), "Left intermediate must be frozen!"
        assert _all_frozen(self.path_left_left), "Left-left path must be frozen!"
        assert _all_frozen(self.ln_f_left_left), "Left-left LN must be frozen!"
        #assert not any(p.requires_grad for p in self.lm_head.perceptrons[0].parameters()), "LM head perceptron 0 must be frozen!"
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
        """Forward through shared layers (0 to split_1-1)"""
        attn = self._expand_attn_mask(attention_mask, hidden_states.dtype, tgt_len=hidden_states.size(1))
        for layer in self.shared_layers:
            hidden_states = layer(hidden_states, attention_mask=attn, use_cache=False)[0]
        return hidden_states

    def forward_intermediate_layers(self, hidden_states, attention_mask, path="left"):
        """Forward through intermediate layers (split_1 to split_2-1)"""
        attn = self._expand_attn_mask(attention_mask, hidden_states.dtype, tgt_len=hidden_states.size(1))
        
        if path == "left":
            layers = self.left_intermediate
        else:  # right
            layers = self.right_intermediate
            
        for layer in layers:
            hidden_states = layer(hidden_states, attention_mask=attn, use_cache=False)[0]
        return hidden_states

    def forward_final_path(self, hidden_states, attention_mask, path_name):
        """Forward through final path layers and return hidden states (before LM head)"""
        attn = self._expand_attn_mask(attention_mask, hidden_states.dtype, tgt_len=hidden_states.size(1))
        
        if path_name == "left_left":
            layers = self.path_left_left
            ln = self.ln_f_left_left
        elif path_name == "left_right":
            layers = self.path_left_right
            ln = self.ln_f_left_right
        elif path_name == "right_left":
            layers = self.path_right_left
            ln = self.ln_f_right_left
        else:  # right_right
            layers = self.path_right_right
            ln = self.ln_f_right_right
        
        for layer in layers:
            hidden_states = layer(hidden_states, attention_mask=attn, use_cache=False)[0]
        hidden_states = ln(hidden_states)
        return hidden_states

    def _gate_weights(self, hidden_states, gate_module, hard=False):
        """Compute gate weights"""
        logits = gate_module(hidden_states)  # [B,S,2]
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
        # (compute all K once; K ≤ 3 for LL, ≤2 for RL/RR, so cost is tiny)
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

        # (optional) keep fp32 for stability; cast back if you prefer x.dtype
        # final_log_probs = final_log_probs.to(x.dtype)

        return final_log_probs

    def _as_log_probs(self, x):
        # Detect if `x` are already log-probs: logsumexp ~ 0
        lse = torch.logsumexp(x.float(), dim=-1)              # [..., V] -> [...]
        is_logprob = lse.detach().median().abs() < 1e-3
        return x if is_logprob else F.log_softmax(x, dim=-1)

    def _logspace_mix(self, path_logits, w_ll, w_lr, w_rl, w_rr):
        # path_logits: dict{name->[B,S,V]} (some entries may be logits, some log-probs)
        ll = self._as_log_probs(path_logits["left_left"])
        lr = self._as_log_probs(path_logits["left_right"])
        rl = self._as_log_probs(path_logits["right_left"])
        rr = self._as_log_probs(path_logits["right_right"])
        # weights are [B,S,1]; clamp and log
        eps = 1e-9
        l_w_ll = (w_ll.clamp_min(eps)).log()
        l_w_lr = (w_lr.clamp_min(eps)).log()
        l_w_rl = (w_rl.clamp_min(eps)).log()
        l_w_rr = (w_rr.clamp_min(eps)).log()
        # log-sum-exp mix: returns log-probs [B,S,V]
        stacked = torch.stack([ll + l_w_ll, lr + l_w_lr, rl + l_w_rl, rr + l_w_rr], dim=0)
        return torch.logsumexp(stacked, dim=0)

    def forward(self, input_ids, attention_mask=None, labels=None,
                return_all_paths=False, path_selection="gate_soft",
                return_all_logits=False, return_head_indices: bool=False):

        # ===== memory saver contexts =====
        # trainer should set: self.backbone_is_frozen = bool(args.freeze_all_transformer)
        use_no_grad = getattr(self, "backbone_is_frozen", False)
        ng = torch.no_grad if use_no_grad else contextlib.nullcontext

        # detect split-gate freeze from params (so we don't need an extra flag)
        def _any_grad(module):
            try:
                return any(p.requires_grad for p in module.parameters())
            except StopIteration:
                return False
        split_gates_trainable = _any_grad(self.gate_1) or _any_grad(self.gate_2_left) or _any_grad(self.gate_2_right)
        gg = contextlib.nullcontext if split_gates_trainable else torch.no_grad  # gate-MLPs

        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # ===== 1) Embeddings + shared stack =====
        with ng():
            hidden_states = self.get_embeddings(input_ids, attention_mask)
            shared_output = self.forward_shared_layers(hidden_states, attention_mask)

        # ===== 2) First split =====
        left_hidden  = shared_output.clone()
        right_hidden = shared_output.clone()

        # ===== 3) Intermediate stacks =====
        with ng():
            left_intermediate_output  = self.forward_intermediate_layers(left_hidden,  attention_mask, "left")
            right_intermediate_output = self.forward_intermediate_layers(right_hidden, attention_mask, "right")

        # ===== 4) Leaf inputs =====
        path_inputs = {
            "left_left":  left_intermediate_output.clone(),
            "left_right": left_intermediate_output.clone(),
            "right_left": right_intermediate_output.clone(),
            "right_right":right_intermediate_output.clone(),
        }

        # ===== 5) Per-leaf scores (LM-head mixtures) =====
        path_logits = {}
        head_top_idx = {} if return_head_indices else None   # returned to caller only if requested

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

        # Compute per-path logits (and optionally per-path head winners)
        per_path_local_head_wins = {}   # for training-time logging regardless of return_head_indices
        for path_name, x_path in path_inputs.items():
            use_subset = (self.head_allocation is not None) and (path_name in self.head_allocation)

            if self.use_head_mixture and use_subset:
                gate = self._heads_gate_by_path[path_name]

                # run leaf transformer WITHOUT grad if backbone is frozen
                with ng():
                    leaf_h = self.forward_final_path(x_path, attention_mask, path_name)

                # head-gates + LM heads stay trainable (no no_grad here)
                path_logits[path_name] = self._mix_heads_logprobs_subset(
                    leaf_h,
                    self.head_allocation[path_name],
                    gate,
                    fast_k=self.head_fast_k,
                    temp=self.head_gate_temp,
                )

                if return_head_indices:
                    head_top_idx[path_name] = _per_path_head_argmax(gate, leaf_h, path_name)

                if self.training:
                    with torch.no_grad():
                        per_path_local_head_wins[path_name] = _per_path_head_argmax(gate, leaf_h, path_name)

            else:
                # Fallback: 2 fixed heads (left→0, right→1); still run leaf transformer
                with ng():
                    leaf_h = self.forward_final_path(x_path, attention_mask, path_name)

                head_idx = 0 if path_name in ("left_left", "left_right") else 1
                path_logits[path_name] = self.lm_head(leaf_h, head_index=head_idx)  # [B,S,V] logits

                if return_head_indices:
                    head_top_idx[path_name] = torch.full(
                        (batch_size, seq_len), head_idx, dtype=torch.long, device=leaf_h.device
                    )
                if self.training:
                    with torch.no_grad():
                        per_path_local_head_wins[path_name] = torch.full(
                            (batch_size, seq_len), head_idx, dtype=torch.long, device=leaf_h.device
                        )

        if return_all_paths:
            return path_logits

        if return_all_logits:
            return torch.stack([
                path_logits["left_left"],
                path_logits["left_right"],
                path_logits["right_left"],
                path_logits["right_right"],
            ], dim=-1)  # [B,S,V,4]

        # ===== 6) Path selection & mixing =====
        used_log_probs = False
        names = ("left_left","left_right","right_left","right_right")
        active_path_idx = None  # [B,S]

        if path_selection in ("hierarchical_gate", "gate_soft", "gate_hard"):
            hard = (path_selection == "gate_hard")

            # split gates: wrap with gg() so no graph if frozen
            with gg():
                gate1, gate1_logits = self._gate_weights(shared_output, self.gate_1, hard=hard)
                gate2_left,  gate2_left_logits  = self._gate_weights(left_intermediate_output,  self.gate_2_left,  hard=hard)
                gate2_right, gate2_right_logits = self._gate_weights(right_intermediate_output, self.gate_2_right, hard=hard)

            w_left  = gate1[..., 0].unsqueeze(-1)
            w_right = gate1[..., 1].unsqueeze(-1)
            w_ll = w_left  * gate2_left[...,  0].unsqueeze(-1)
            w_lr = w_left  * gate2_left[...,  1].unsqueeze(-1)
            w_rl = w_right * gate2_right[..., 0].unsqueeze(-1)
            w_rr = w_right * gate2_right[..., 1].unsqueeze(-1)

            final_log_probs = self._logspace_mix(path_logits, w_ll, w_lr, w_rl, w_rr)
            used_log_probs = True

            weights4 = torch.stack([w_ll, w_lr, w_rl, w_rr], dim=-1).squeeze(-2)  # [B,S,4]
            active_path_idx = torch.argmax(weights4, dim=-1)                      # [B,S]

            gate_info = {
                "gate1": gate1, "gate1_logits": gate1_logits,
                "gate2_left": gate2_left, "gate2_left_logits": gate2_left_logits,
                "gate2_right": gate2_right, "gate2_right_logits": gate2_right_logits,
                "final_weights": {"left_left": w_ll, "left_right": w_lr, "right_left": w_rl, "right_right": w_rr},
            }

        elif path_selection == "left_left_only":
            final_logits = path_logits["left_left"]
            used_log_probs = False
            gate_info = None
            active_path_idx = torch.zeros((batch_size, seq_len), dtype=torch.long, device=final_logits.device)

        elif path_selection == "max_prob":
            all_max = []
            for n in names:
                x = path_logits[n]
                lse = torch.logsumexp(x.float(), dim=-1)
                is_lp = lse.detach().median().abs() < 1e-3
                probs = x.exp() if is_lp else F.softmax(x, dim=-1)
                all_max.append(probs.max(dim=-1).values)
            stacked_max = torch.stack(all_max, dim=-1)      # [B,S,4]
            best_idx = torch.argmax(stacked_max, dim=-1)    # [B,S]

            final_logits = torch.zeros_like(path_logits[names[0]])
            for i, n in enumerate(names):
                mask = (best_idx == i).unsqueeze(-1)
                final_logits = torch.where(mask, path_logits[n], final_logits)
            gate_info = {"best_path_indices": best_idx}
            active_path_idx = best_idx

        else:
            raise ValueError(f"Unknown path_selection: {path_selection}")

        if not used_log_probs:
            x = final_logits
            lse = torch.logsumexp(x.float(), dim=-1)
            final_log_probs = x if (lse.detach().median().abs() < 1e-3) else F.log_softmax(x, dim=-1)
            used_log_probs = True

        # ===== 7) Loss (mask-aware) =====
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

        # ===== 8) Build a combined head index for logging =====
        head_top_idx_combined = None
        with torch.no_grad():
            if active_path_idx is not None and per_path_local_head_wins:
                dev = active_path_idx.device
                head_top_idx_combined = torch.zeros((batch_size, seq_len), dtype=torch.long, device=dev)
                for i, n in enumerate(names):
                    if n in per_path_local_head_wins:
                        head_top_idx_combined = torch.where(
                            active_path_idx == i, per_path_local_head_wins[n], head_top_idx_combined
                        )

        returned_head_top_idx = head_top_idx if return_head_indices else None
        returned_head_top_idx_combined = head_top_idx_combined if return_head_indices else None

        # ===== 9) ADD USAGE TRACKING =====
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
                        for i, path_name in enumerate(names):
                            path_mask = (api == i) & vm
                            ntok = int(path_mask.sum().item())
                            if ntok <= 0: continue
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
                                hid = 0 if path_name in ("left_left", "left_right") else 1
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
        }

    def forward2(self, input_ids, attention_mask=None, labels=None,
                return_all_paths=False, path_selection="gate_soft",
                return_all_logits=False, return_head_indices: bool=False):
        
        use_no_grad = getattr(self, "backbone_is_frozen", False)
        ng = torch.no_grad if use_no_grad else contextlib.nullcontext

        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # ===== 1) Embeddings + shared stack =====
        hidden_states = self.get_embeddings(input_ids, attention_mask)
        shared_output = self.forward_shared_layers(hidden_states, attention_mask)

        # ===== 2) First split =====
        left_hidden  = shared_output.clone()
        right_hidden = shared_output.clone()

        # ===== 3) Intermediate stacks =====
        left_intermediate_output  = self.forward_intermediate_layers(left_hidden,  attention_mask, "left")
        right_intermediate_output = self.forward_intermediate_layers(right_hidden, attention_mask, "right")

        # ===== 4) Leaf inputs =====
        path_inputs = {
            "left_left":  left_intermediate_output.clone(),
            "left_right": left_intermediate_output.clone(),
            "right_left": right_intermediate_output.clone(),
            "right_right":right_intermediate_output.clone(),
        }

        # ===== 5) Per-leaf scores (LM-head mixtures) =====
        path_logits = {}
        head_top_idx = {} if return_head_indices else None   # returned to caller only if requested

        # helpers
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

        # Compute per-path logits (and optionally per-path head winners)
        per_path_local_head_wins = {}   # <-- A) populated for training-time logging regardless of return_head_indices
        for path_name, x_path in path_inputs.items():
            use_subset = (self.head_allocation is not None) and (path_name in self.head_allocation)

            if self.use_head_mixture and use_subset:
                gate = self._heads_gate_by_path[path_name]
                leaf_h = self.forward_final_path(x_path, attention_mask, path_name)

                path_logits[path_name] = self._mix_heads_logprobs_subset(
                    leaf_h,
                    self.head_allocation[path_name],
                    gate,
                    fast_k=self.head_fast_k,
                    temp=self.head_gate_temp,
                )

                # Caller-visible indices (only if requested)
                if return_head_indices:
                    head_top_idx[path_name] = _per_path_head_argmax(gate, leaf_h, path_name)

                # A) Training-time logging head winners (independent of return_head_indices)
                if self.training:
                    with torch.no_grad():
                        per_path_local_head_wins[path_name] = _per_path_head_argmax(gate, leaf_h, path_name)

            else:
                # Fallback: 2 fixed heads (left→0, right→1); still run leaf transformer
                leaf_h = self.forward_final_path(x_path, attention_mask, path_name)
                head_idx = 0 if path_name in ("left_left", "left_right") else 1
                path_logits[path_name] = self.lm_head(leaf_h, head_index=head_idx)  # [B,S,V] logits

                if return_head_indices:
                    head_top_idx[path_name] = torch.full(
                        (batch_size, seq_len), head_idx, dtype=torch.long, device=leaf_h.device
                    )
                if self.training:
                    with torch.no_grad():
                        per_path_local_head_wins[path_name] = torch.full(
                            (batch_size, seq_len), head_idx, dtype=torch.long, device=leaf_h.device
                        )

        if return_all_paths:
            return path_logits

        # (Optional) stacked per-path scores
        if return_all_logits:
            return torch.stack([
                path_logits["left_left"],
                path_logits["left_right"],
                path_logits["right_left"],
                path_logits["right_right"],
            ], dim=-1)  # [B,S,V,4]

        # ===== 6) Path selection & mixing =====
        used_log_probs = False
        names = ("left_left","left_right","right_left","right_right")
        active_path_idx = None  # [B,S], filled below

        if path_selection in ("hierarchical_gate", "gate_soft", "gate_hard"):
            hard = (path_selection == "gate_hard")

            gate1, gate1_logits = self._gate_weights(shared_output, self.gate_1, hard=hard)
            gate2_left,  gate2_left_logits  = self._gate_weights(left_intermediate_output,  self.gate_2_left,  hard=hard)
            gate2_right, gate2_right_logits = self._gate_weights(right_intermediate_output, self.gate_2_right, hard=hard)

            w_left  = gate1[..., 0].unsqueeze(-1)
            w_right = gate1[..., 1].unsqueeze(-1)
            w_ll = w_left  * gate2_left[...,  0].unsqueeze(-1)
            w_lr = w_left  * gate2_left[...,  1].unsqueeze(-1)
            w_rl = w_right * gate2_right[..., 0].unsqueeze(-1)
            w_rr = w_right * gate2_right[..., 1].unsqueeze(-1)

            final_log_probs = self._logspace_mix(path_logits, w_ll, w_lr, w_rl, w_rr)
            used_log_probs = True

            weights4 = torch.stack([w_ll, w_lr, w_rl, w_rr], dim=-1).squeeze(-2)  # [B,S,4]
            active_path_idx = torch.argmax(weights4, dim=-1)                      # [B,S]

            gate_info = {
                "gate1": gate1, "gate1_logits": gate1_logits,
                "gate2_left": gate2_left, "gate2_left_logits": gate2_left_logits,
                "gate2_right": gate2_right, "gate2_right_logits": gate2_right_logits,
                "final_weights": {"left_left": w_ll, "left_right": w_lr, "right_left": w_rl, "right_right": w_rr},
            }

        elif path_selection == "left_left_only":
            final_logits = path_logits["left_left"]
            used_log_probs = False
            gate_info = None
            active_path_idx = torch.zeros((batch_size, seq_len), dtype=torch.long, device=final_logits.device)

        elif path_selection == "max_prob":
            # choose the path with the highest per-token max prob
            all_max = []
            for n in names:
                x = path_logits[n]
                lse = torch.logsumexp(x.float(), dim=-1)
                is_lp = lse.detach().median().abs() < 1e-3
                probs = x.exp() if is_lp else F.softmax(x, dim=-1)
                all_max.append(probs.max(dim=-1).values)
            stacked_max = torch.stack(all_max, dim=-1)      # [B,S,4]
            best_idx = torch.argmax(stacked_max, dim=-1)    # [B,S]

            final_logits = torch.zeros_like(path_logits[names[0]])
            for i, n in enumerate(names):
                mask = (best_idx == i).unsqueeze(-1)
                final_logits = torch.where(mask, path_logits[n], final_logits)
            gate_info = {"best_path_indices": best_idx}
            active_path_idx = best_idx

        else:
            raise ValueError(f"Unknown path_selection: {path_selection}")

        # ensure final_log_probs exists if not mixed above
        if not used_log_probs:
            x = final_logits
            lse = torch.logsumexp(x.float(), dim=-1)
            final_log_probs = x if (lse.detach().median().abs() < 1e-3) else F.log_softmax(x, dim=-1)
            used_log_probs = True

        # ===== 7) Loss (mask-aware) =====
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

        # ===== 8) Build a combined head index for logging =====
        # Build regardless of return_head_indices so training usage is always correct (A)
        head_top_idx_combined = None
        with torch.no_grad():
            if active_path_idx is not None and per_path_local_head_wins:
                dev = active_path_idx.device
                head_top_idx_combined = torch.zeros((batch_size, seq_len), dtype=torch.long, device=dev)
                for i, n in enumerate(names):
                    if n in per_path_local_head_wins:
                        head_top_idx_combined = torch.where(
                            active_path_idx == i, per_path_local_head_wins[n], head_top_idx_combined
                        )

        # If caller asked for per-path maps, keep returning them (unchanged)
        returned_head_top_idx = head_top_idx if return_head_indices else None
        returned_head_top_idx_combined = head_top_idx_combined if return_head_indices else None

        # ===== 9) ADD USAGE TRACKING =====
        if self.training:
            with torch.no_grad():
                # B) Valid mask aligned to targets/logits range [B,S-1]
                L = seq_len
                if attention_mask is not None:
                    valid_mask = attention_mask[..., 1:].contiguous().bool()
                else:
                    valid_mask = torch.ones_like(input_ids[..., 1:], dtype=torch.bool)

                # align any [B,S] map to [B,S-1] by slicing off the first position
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
                    # Fallback: path-based tokens split across allocated heads
                    if active_path_idx is not None:
                        api = _align_to_targets(active_path_idx)
                        vm = valid_mask
                        for i, path_name in enumerate(names):
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
                                hid = 0 if path_name in ("left_left", "left_right") else 1
                                self.lm_head.usage_counts[hid] += ntok
                                self.lm_head.selection_counts[hid] += ntok

        return {
            "loss": loss,
            "logits": final_log_probs,       # expose normalized scores (log-probs)
            "log_probs": final_log_probs,
            "path_logits": path_logits,
            "gate": gate_info,
            "head_top_idx": returned_head_top_idx,                      # per-path maps (if requested)
            "head_top_idx_combined": returned_head_top_idx_combined,    # single map (if requested)
        }

    def get_trainable_parameters(self):
        params = []
        # split routers
        params += list(self.gate_1.parameters())
        params += list(self.gate_2_left.parameters())
        params += list(self.gate_2_right.parameters())
        # per-path head gates (the small MLPs over each subset)
        params += list(self._heads_gate_by_path.parameters())
        # right branch + trainable leaves + their LNs
        params += list(self.right_intermediate.parameters())
        params += list(self.path_left_right.parameters())
        params += list(self.path_right_left.parameters())
        params += list(self.path_right_right.parameters())
        params += list(self.ln_f_left_right.parameters())
        params += list(self.ln_f_right_left.parameters())
        params += list(self.ln_f_right_right.parameters())
        # LM heads: keep head 0 frozen as anchor; train all others (1..N-1)
        for i, head in enumerate(self.lm_head.perceptrons):
            if i == 0:
                continue
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
    
        """
        path_logits: dict with keys 'left_left','left_right','right_left','right_right' (each [B,S,V])
        weights:     tensors [B,S,1] for each leaf; not softmaxed here
        returns:     log-prob tensor [B,S,V]
        """
        eps = 1e-8
        logp_ll = F.log_softmax(path_logits["left_left"],  dim=-1)
        logp_lr = F.log_softmax(path_logits["left_right"], dim=-1)
        logp_rl = F.log_softmax(path_logits["right_left"], dim=-1)
        logp_rr = F.log_softmax(path_logits["right_right"],dim=-1)

        stack_lp = torch.stack([logp_ll, logp_lr, logp_rl, logp_rr], dim=-2)  # [B,S,4,V]
        W = torch.stack([w_ll, w_lr, w_rl, w_rr], dim=-2).clamp_min(eps)      # [B,S,4,1]

        return torch.logsumexp(W.log() + stack_lp, dim=-2)  # [B,S,V]