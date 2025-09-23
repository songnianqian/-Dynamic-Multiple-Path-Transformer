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

# Quad Path GPT Model with Hierarchical Split Architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel
import math
from torch.optim import AdamW

class HierarchicalQuadPathGPT2(nn.Module):
    """
    GPT-2 model with hierarchical splits:
    - First split at layer 6: left (frozen baseline) vs right (trainable)
    - Second split at layer 9: each path splits into two more paths
    - Final paths: left_left, left_right, right_left, right_right
    """

    def __init__(self, config, pretrained_model="gpt2",
                 split_at_layer_1=6, split_at_layer_2=9,
                 gate_hidden=256, gate_temp=1.0,
                 dual_path_checkpoint=None):  # NEW
        super().__init__()
        self.config = config
        self.split_at_layer_1 = split_at_layer_1  # First split (6)
        self.split_at_layer_2 = split_at_layer_2  # Second split (9)
        self.gate_temp = gate_temp
        
        assert split_at_layer_2 > split_at_layer_1, "Second split must be after first split"
        
        # Load pretrained model for initialization
        print(f"Loading pretrained model: {pretrained_model}")
        hf_model = GPT2LMHeadModel.from_pretrained(pretrained_model)

        # Load and adapt dual path checkpoint if provided
        if dual_path_checkpoint is not None:
            self._load_from_dual_path_checkpoint(dual_path_checkpoint)
        
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
        
        # LM heads for each final path
        self.lm_head_left_left = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head_left_right = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head_right_left = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head_right_right = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize all heads with pretrained weights
        for head in [self.lm_head_left_left, self.lm_head_left_right, 
                    self.lm_head_right_left, self.lm_head_right_right]:
            head.weight.data.copy_(hf_model.lm_head.weight.data)
        
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
        
        # Load and adapt dual path checkpoint if provided
        if dual_path_checkpoint is not None:
            self._load_from_dual_path_checkpoint(dual_path_checkpoint)
            
        print(f"Quad path model initialized:")
        print(f"  Shared layers (0-{split_at_layer_1-1}): {len(self.shared_layers)}")
        print(f"  Intermediate layers ({split_at_layer_1}-{split_at_layer_2-1}): {len(self.left_intermediate)}")
        print(f"  Final paths ({split_at_layer_2}-end): {len(self.path_left_left)}")
        print(f"  Frozen paths: left_left + left_intermediate + shared")
        print(f"  Trainable paths: left_right, right_left, right_right + right_intermediate")
        if dual_path_checkpoint is not None:
            print(f"  Initialized from dual path checkpoint: {dual_path_checkpoint}")

    def set_path_freezing(self, freeze_config: dict):
        """
        freeze_config keys (booleans, all optional):
        shared, left_intermediate, right_intermediate,
        left_right, right_left, right_right,
        gate1, gate2_left, gate2_right, gates  (legacy: all gates)
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

            # deep leaves (s2..end) + their heads / layer norms
            "left_right":  [getattr(self, "path_left_right", None),
                            getattr(self, "ln_f_left_right", None),
                            getattr(self, "lm_head_left_right", None)],
            "right_left":  [getattr(self, "path_right_left", None),
                            getattr(self, "ln_f_right_left", None),
                            getattr(self, "lm_head_right_left", None)],
            "right_right": [getattr(self, "path_right_right", None),
                            getattr(self, "ln_f_right_right", None),
                            getattr(self, "lm_head_right_right", None)],
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

    def _load_from_dual_path_checkpoint(self, checkpoint_path):
        """
        Load weights from a dual path checkpoint and properly initialize quad paths
        """
        print(f"Loading dual path checkpoint: {checkpoint_path}")
        
        try:
            # Try loading with weights_only first
            dual_ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except Exception as e:
            print(f"Failed weights_only=True: {e}")
            try:
                dual_ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                print(f"[warn] Loaded with weights_only=False")
            except Exception as e2:
                raise RuntimeError(f"Failed to load dual path checkpoint: {e2}")
        
        # Extract state dict
        if isinstance(dual_ckpt, dict):
            dual_state = dual_ckpt.get("model_state_dict", dual_ckpt)
        else:
            dual_state = dual_ckpt
        
        print("Mapping dual path weights to quad path structure...")
        
        # Create mapping for quad path state dict
        quad_state = self.state_dict()
        loaded_keys = set()
        
        # 1. Direct mappings (these should load exactly)
        direct_mappings = [
            # Shared components
            'wte.', 'wpe.',
            # Shared layers (0 to split_1-1)
            'shared_layers.',
            # Left intermediate becomes our left intermediate (6-8)
            'left_intermediate.',
            # Left path becomes our left_left path (9-11)
            'path_left_left.',
            'ln_f_left_left.',  # was ln_f_left
            'lm_head_left_left.',  # was left_lm_head
        ]
        
        for key in list(dual_state.keys()):
            new_key = key
            
            # Handle naming differences
            if key.startswith('left_path.'):
                new_key = key.replace('left_path.', 'path_left_left.')
            elif key.startswith('ln_f_left.'):
                new_key = key.replace('ln_f_left.', 'ln_f_left_left.')
            elif key.startswith('left_lm_head.'):
                new_key = key.replace('left_lm_head.', 'lm_head_left_left.')
            
            # Check if this is a direct mapping
            for prefix in direct_mappings:
                if new_key.startswith(prefix):
                    if new_key in quad_state:
                        quad_state[new_key] = dual_state[key].clone()
                        loaded_keys.add(new_key)
                        print(f"  Direct: {key} -> {new_key}")
                    break
        
        # 2. Duplicate right intermediate to our right intermediate
        for key in dual_state.keys():
            if key.startswith('right_path.'):
                # Extract layer index from dual path
                parts = key.split('.')
                if len(parts) >= 2 and parts[1].isdigit():
                    layer_idx = int(parts[1])
                    # Map to right_intermediate if it's in the intermediate range
                    if layer_idx < (self.split_at_layer_2 - self.split_at_layer_1):
                        new_key = key.replace('right_path.', 'right_intermediate.')
                        if new_key in quad_state:
                            quad_state[new_key] = dual_state[key].clone()
                            loaded_keys.add(new_key)
                            print(f"  Right->RightIntermediate: {key} -> {new_key}")
        
        # 3. Duplicate weights for the new quad paths
        # right_path (layers after intermediate) -> right_right
        for key in dual_state.keys():
            if key.startswith('right_path.'):
                parts = key.split('.')
                if len(parts) >= 2 and parts[1].isdigit():
                    layer_idx = int(parts[1])
                    # Map to right_right if it's in the final layers range
                    if layer_idx >= (self.split_at_layer_2 - self.split_at_layer_1):
                        adjusted_idx = layer_idx - (self.split_at_layer_2 - self.split_at_layer_1)
                        new_key = key.replace(f'right_path.{layer_idx}.', f'path_right_right.{adjusted_idx}.')
                        if new_key in quad_state:
                            quad_state[new_key] = dual_state[key].clone()
                            loaded_keys.add(new_key)
                            print(f"  Right->RightRight: {key} -> {new_key}")
        
        # 4. Handle right LN and LM head -> right_right versions
        ln_lm_mappings = [
            ('ln_f_right.', 'ln_f_right_right.'),
            ('right_lm_head.', 'lm_head_right_right.'),
        ]
        
        for old_prefix, new_prefix in ln_lm_mappings:
            for key in dual_state.keys():
                if key.startswith(old_prefix):
                    new_key = key.replace(old_prefix, new_prefix)
                    if new_key in quad_state:
                        quad_state[new_key] = dual_state[key].clone()
                        loaded_keys.add(new_key)
                        print(f"  Right LN/LM: {key} -> {new_key}")
        
        # 5. Initialize left_right path by duplicating from left_left (frozen baseline)
        print("  Initializing left_right from left_left...")
        for key in quad_state.keys():
            if key.startswith('path_left_right.') and key not in loaded_keys:
                left_left_key = key.replace('path_left_right.', 'path_left_left.')
                if left_left_key in quad_state:
                    quad_state[key] = quad_state[left_left_key].clone()
                    loaded_keys.add(key)
                    print(f"    Duplicate: {left_left_key} -> {key}")
        
        # left_right LN and LM head
        for key in quad_state.keys():
            if (key.startswith('ln_f_left_right.') or key.startswith('lm_head_left_right.')) and key not in loaded_keys:
                if key.startswith('ln_f_left_right.'):
                    source_key = key.replace('ln_f_left_right.', 'ln_f_left_left.')
                else:
                    source_key = key.replace('lm_head_left_right.', 'lm_head_left_left.')
                
                if source_key in quad_state:
                    quad_state[key] = quad_state[source_key].clone()
                    loaded_keys.add(key)
                    print(f"    Duplicate LN/LM: {source_key} -> {key}")
        
        # 6. Initialize right_left path by duplicating from right_right
        print("  Initializing right_left from right_right...")
        for key in quad_state.keys():
            if key.startswith('path_right_left.') and key not in loaded_keys:
                right_right_key = key.replace('path_right_left.', 'path_right_right.')
                if right_right_key in quad_state:
                    quad_state[key] = quad_state[right_right_key].clone()
                    loaded_keys.add(key)
                    print(f"    Duplicate: {right_right_key} -> {key}")
        
        # right_left LN and LM head
        for key in quad_state.keys():
            if (key.startswith('ln_f_right_left.') or key.startswith('lm_head_right_left.')) and key not in loaded_keys:
                if key.startswith('ln_f_right_left.'):
                    source_key = key.replace('ln_f_right_left.', 'ln_f_right_right.')
                else:
                    source_key = key.replace('lm_head_right_left.', 'lm_head_right_right.')
                
                if source_key in quad_state:
                    quad_state[key] = quad_state[source_key].clone()
                    loaded_keys.add(key)
                    print(f"    Duplicate LN/LM: {source_key} -> {key}")
        
        # 7. Initialize gates from dual path gate if available
        print("  Initializing gates...")
        if 'gate.0.weight' in dual_state:
            # Initialize gate_1 from dual path gate
            for key in dual_state.keys():
                if key.startswith('gate.'):
                    new_key = key.replace('gate.', 'gate_1.')
                    if new_key in quad_state:
                        quad_state[new_key] = dual_state[key].clone()
                        loaded_keys.add(new_key)
                        print(f"    Gate: {key} -> {new_key}")
            
            # Initialize gate_2_left and gate_2_right from gate_1 (will be trained differently)
            for gate_name in ['gate_2_left', 'gate_2_right']:
                for key in quad_state.keys():
                    if key.startswith(f'{gate_name}.') and key not in loaded_keys:
                        gate1_key = key.replace(f'{gate_name}.', 'gate_1.')
                        if gate1_key in quad_state:
                            quad_state[key] = quad_state[gate1_key].clone()
                            loaded_keys.add(key)
                            print(f"    Gate duplicate: {gate1_key} -> {key}")
        
        # Load the adapted state dict
        missing, unexpected = self.load_state_dict(quad_state, strict=False)
        
        print(f"Dual path checkpoint loading complete:")
        print(f"  Loaded keys: {len(loaded_keys)}")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        
        if missing:
            print(f"  Missing (will use random init): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        
        print("Quad path model successfully initialized from dual path checkpoint!")

    def load_dual_path_checkpoint_weights_only(self, checkpoint_path):
        """
        Alternative method to load just the weights from a dual path checkpoint
        after the model is already created. Useful for fine-tuning scenarios.
        """
        return self._load_from_dual_path_checkpoint(checkpoint_path)

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
        for param in self.lm_head_left_left.parameters():
            param.requires_grad = False
            
        print("Frozen paths: shared + left_intermediate + left_left")

    def _assert_frozen_paths(self):
        """Assert that frozen paths are actually frozen"""
        def _all_frozen(mod):
            return all(not p.requires_grad for p in mod.parameters())
        
        assert _all_frozen(self.shared_layers), "Shared layers must be frozen!"
        assert _all_frozen(self.left_intermediate), "Left intermediate must be frozen!"
        assert _all_frozen(self.path_left_left), "Left-left path must be frozen!"
        assert _all_frozen(self.ln_f_left_left), "Left-left LN must be frozen!"
        assert _all_frozen(self.lm_head_left_left), "Left-left LM head must be frozen!"
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
        """Forward through final path layers"""
        attn = self._expand_attn_mask(attention_mask, hidden_states.dtype, tgt_len=hidden_states.size(1))
        
        if path_name == "left_left":
            layers = self.path_left_left
            ln = self.ln_f_left_left
            lm_head = self.lm_head_left_left
        elif path_name == "left_right":
            layers = self.path_left_right
            ln = self.ln_f_left_right
            lm_head = self.lm_head_left_right
        elif path_name == "right_left":
            layers = self.path_right_left
            ln = self.ln_f_right_left
            lm_head = self.lm_head_right_left
        else:  # right_right
            layers = self.path_right_right
            ln = self.ln_f_right_right
            lm_head = self.lm_head_right_right
        
        for layer in layers:
            hidden_states = layer(hidden_states, attention_mask=attn, use_cache=False)[0]
        hidden_states = ln(hidden_states)
        logits = lm_head(hidden_states)
        return logits

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

    def forward(self, input_ids, attention_mask=None, labels=None,
                return_all_paths=False, path_selection="gate_soft"):
        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # 1) Embeddings and shared layers
        hidden_states = self.get_embeddings(input_ids, attention_mask)
        shared_output = self.forward_shared_layers(hidden_states, attention_mask)

        # 2) First split at layer 6
        left_hidden = shared_output.clone()
        right_hidden = shared_output.clone()

        # 3) Intermediate layers
        left_intermediate_output = self.forward_intermediate_layers(left_hidden, attention_mask, "left")
        right_intermediate_output = self.forward_intermediate_layers(right_hidden, attention_mask, "right")

        # 4) Second split at layer 9 - create all four final paths
        path_inputs = {
            "left_left": left_intermediate_output.clone(),
            "left_right": left_intermediate_output.clone(),
            "right_left": right_intermediate_output.clone(),
            "right_right": right_intermediate_output.clone()
        }

        # 5) Forward through all final paths
        path_logits = {}
        for path_name, path_input in path_inputs.items():
            if path_name == "left_left":
                # Frozen path - use no_grad
                with torch.no_grad():
                    path_logits[path_name] = self.forward_final_path(path_input, attention_mask, path_name)
            else:
                # Trainable paths
                path_logits[path_name] = self.forward_final_path(path_input, attention_mask, path_name)

        if return_all_paths:
            return path_logits

        # 6) Path selection and mixing
        used_log_probs = False

        if path_selection in ("hierarchical_gate", "gate_soft", "gate_hard"):
            # unify soft/hard gating
            hard = (path_selection == "gate_hard")

            gate1, gate1_logits = self._gate_weights(shared_output, self.gate_1, hard=hard)
            gate2_left,  gate2_left_logits  = self._gate_weights(left_intermediate_output,  self.gate_2_left,  hard=hard)
            gate2_right, gate2_right_logits = self._gate_weights(right_intermediate_output, self.gate_2_right, hard=hard)

            # Weights per leaf: [B,S,1]
            w_left  = gate1[..., 0].unsqueeze(-1)
            w_right = gate1[..., 1].unsqueeze(-1)

            w_ll = w_left  * gate2_left[...,  0].unsqueeze(-1)   # left_left
            w_lr = w_left  * gate2_left[...,  1].unsqueeze(-1)   # left_right
            w_rl = w_right * gate2_right[..., 0].unsqueeze(-1)   # right_left
            w_rr = w_right * gate2_right[..., 1].unsqueeze(-1)   # right_right

            # >>> log-space mixture for numerical stability (returns log-probs) <<<
            final_log_probs = self._logspace_mix(path_logits, w_ll, w_lr, w_rl, w_rr)
            used_log_probs = True

            gate_info = {
                "gate1": gate1, "gate1_logits": gate1_logits,
                "gate2_left": gate2_left, "gate2_left_logits": gate2_left_logits,
                "gate2_right": gate2_right, "gate2_right_logits": gate2_right_logits,
                "final_weights": {"left_left": w_ll, "left_right": w_lr, "right_left": w_rl, "right_right": w_rr}
            }

        elif path_selection == "left_left_only":
            final_logits = path_logits["left_left"]
            gate_info = None

        elif path_selection == "max_prob":
            # Select leaf with highest per-token max prob (no gates)
            all_max = []
            for name in ("left_left","left_right","right_left","right_right"):
                probs = F.softmax(path_logits[name], dim=-1)
                max_probs, _ = probs.max(dim=-1)           # [B,S]
                all_max.append(max_probs)
            stacked_max = torch.stack(all_max, dim=-1)     # [B,S,4]
            best_idx = torch.argmax(stacked_max, dim=-1)   # [B,S]

            # gather chosen logits per token
            names = ("left_left","left_right","right_left","right_right")
            final_logits = torch.zeros_like(path_logits[names[0]])
            for i, name in enumerate(names):
                mask = (best_idx == i).unsqueeze(-1)       # [B,S,1]
                final_logits = torch.where(mask, path_logits[name], final_logits)

            gate_info = {"best_path_indices": best_idx}

        else:
            raise ValueError(f"Unknown path_selection: {path_selection}")

        # 7) Loss computation
        loss = None
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()

            if used_log_probs:
                # final_log_probs: [B,S,V] -> shift to [B,S-1,V]
                shift_log_probs = final_log_probs[..., :-1, :].contiguous()
                loss = F.nll_loss(
                    shift_log_probs.view(-1, shift_log_probs.size(-1)),
                    shift_labels.view(-1),
                    reduction="mean"
                )
            else:
                # final_logits: raw logits
                shift_logits = final_logits[..., :-1, :].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="mean"
                )

        return {
            "loss": loss,
            # For back-compat, expose 'logits' even when using log-probs
            "logits": final_log_probs if used_log_probs else final_logits,
            "log_probs": final_log_probs if used_log_probs else None,
            "path_logits": path_logits,
            "gate": gate_info
        }

    def get_trainable_parameters(self):
        """Get all trainable parameters"""
        params = []
        
        # Gates
        params += list(self.gate_1.parameters())
        params += list(self.gate_2_left.parameters())
        params += list(self.gate_2_right.parameters())
        
        # Right intermediate path
        params += list(self.right_intermediate.parameters())
        
        # All trainable final paths and their components
        params += list(self.path_left_right.parameters())
        params += list(self.path_right_left.parameters())
        params += list(self.path_right_right.parameters())
        
        params += list(self.ln_f_left_right.parameters())
        params += list(self.ln_f_right_left.parameters())
        params += list(self.ln_f_right_right.parameters())
        
        params += list(self.lm_head_left_right.parameters())
        params += list(self.lm_head_right_left.parameters())
        params += list(self.lm_head_right_right.parameters())
        
        return [p for p in params if p.requires_grad]

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
    
    def _logspace_mix(self, path_logits, w_ll, w_lr, w_rl, w_rr):
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


# =========================
# QuadPathTrainer
# =========================
class QuadPathTrainer:
    """Trainer for the quad path model"""

    def __init__(
        self,
        model,
        tokenizer,
        device,
        checkpoint_dir,
        *,
        # loss weights
        lb_coef=1e-3,
        gold_aux_coef=1e-3,
        tether_coef=5e-4,
        gate_temp=1.0,
        clip_grad=1.0
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.global_step = 0

        # loss coefficients
        self.lb_coef = lb_coef
        self.gold_aux_coef = gold_aux_coef
        self.tether_coef = tether_coef
        self.gate_temp = gate_temp
        self.clip_grad = clip_grad
        self.consistency_lambda = getattr(self, "consistency_lambda", 0.0)

        # report parameters
        info = self.model.get_parameter_count()
        print(f"Model parameters:\n  Total: {info['total']:,}\n  Trainable: {info['trainable']:,}\n  Frozen: {info['frozen']:,}\n  Trainable ratio: {info['trainable_ratio']:.2%}")

    def _as_log_probs(self, x):
        # x: [..., V]
        # Detect if already log-probs (logsumexp≈0); if not, turn logits -> log-probs
        lse = torch.logsumexp(x.float(), dim=-1)
        is_logprob = lse.median().abs() < 1e-3
        return x if is_logprob else F.log_softmax(x, dim=-1)

    def create_optimizer(self, lr: float,
                        head_lr: float = None,
                        gate_lr: float = None,
                        weight_decay: float = 0.01,
                        betas=(0.9, 0.95), eps=1e-8):
        m = self.model

        def trainable_named():
            for n, p in m.named_parameters():
                if p.requires_grad:
                    yield n, p

        # Buckets
        base_params = []
        head_params = []         # lm_head.perceptrons.*
        gate_head_params = []    # _heads_gate_by_path.<path>.net.*
        split_gate_params = []   # gate_1.*, gate_2_left.*, gate_2_right.*

        # Optional: exclude bias/LayerNorm from weight decay
        def no_wd(n, p):
            if n.endswith(".bias"):
                return True
            # LayerNorm weight names vary; this catches common ones
            return (".ln" in n) or (".layernorm" in n.lower()) or (".ln_f" in n)

        base_params_wd, base_params_nowd = [], []

        for n, p in trainable_named():
            if n.startswith("_heads_gate_by_path"):
                gate_head_params.append(p)
            elif n.startswith(("gate_1", "gate_2_left", "gate_2_right")):
                split_gate_params.append(p)
            elif n.startswith("lm_head.perceptrons"):
                head_params.append(p)
            else:
                (base_params_nowd if no_wd(n, p) else base_params_wd).append(p)

        groups = []
        # Base weights: split WD / no-WD for best AdamW hygiene
        if base_params_wd:
            groups.append({"params": base_params_wd, "lr": lr, "weight_decay": weight_decay, "name": "base_wd"})
        if base_params_nowd:
            groups.append({"params": base_params_nowd, "lr": lr, "weight_decay": 0.0, "name": "base_nowd"})

        # LM heads
        if head_params:
            groups.append({"params": head_params, "lr": head_lr or lr, "weight_decay": weight_decay, "name": "lm_heads"})

        # Gates: per-path head gates + split gates (often zero WD)
        gates_bucket = gate_head_params + split_gate_params
        if gates_bucket:
            groups.append({"params": gates_bucket, "lr": gate_lr or lr, "weight_decay": 0.0, "name": "gates"})

        opt = AdamW(groups, betas=betas, eps=eps)

        # Debug summary
        try:
            for g in opt.param_groups:
                n_params = sum(p.numel() for p in g["params"])
                print(f"[optimizer] {g.get('name','?'):>12} | lr={g['lr']:.2e} | wd={g.get('weight_decay',0)} | params={n_params}")
        except Exception:
            pass

        return opt

    def zero_grad(self, optimizer):
        optimizer.zero_grad()

    def optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), max_norm=self.clip_grad)
        optimizer.step()
        optimizer.zero_grad()
        self.global_step += 1

    def _token_ce(self, log_probs, labels, weights=None, ignore_index: int = -100):
        """Cross-entropy loss with optional token weights"""
        if log_probs.dim() == 2:
            log_probs = log_probs.unsqueeze(0)
        elif log_probs.dim() != 3:
            raise ValueError(f"log_probs must be 2D or 3D, got {log_probs.shape}")

        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        elif labels.dim() != 2:
            raise ValueError(f"labels must be 1D or 2D, got {labels.shape}")

        if weights is not None:
            if weights.dim() == 1:
                weights = weights.unsqueeze(0)
            elif weights.dim() != 2:
                raise ValueError(f"weights must be 1D or 2D, got {weights.shape}")

        B, S, V = log_probs.shape

        # Standard next-token shift
        shift_logp = log_probs[:, :-1, :]
        shift_y = labels[:, 1:]

        # Valid mask
        valid_mask = (shift_y != ignore_index)
        if not valid_mask.any():
            return shift_logp.new_zeros(())

        safe_y = shift_y.masked_fill(~valid_mask, 0)

        # Gather gold log-probs
        gold_logp = shift_logp.gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)
        nll = -gold_logp

        # Optional token weights
        if weights is not None:
            w = weights[:, 1:]
            w = torch.where(valid_mask, w, torch.zeros_like(w))
            denom = (w.sum() + 1e-8)
            if denom > 0:
                w = w * (w.numel() / denom)
                nll = nll * w

        # Mean over valid tokens only
        loss = nll[valid_mask].mean()
        return loss

    @staticmethod
    def _kl_mean(p_log, q_log):
        """Mean KL divergence KL(p || q) where both are log-probs"""
        p = p_log.exp()
        return (p * (p_log - q_log)).mean()

    def _hierarchical_log_mix(self, path_logits, gate1_logits, gate2_left_logits, gate2_right_logits, hard=False):
        # gates -> probs (or one-hot if hard)
        temp = max(getattr(self, "gate_temp", 1.0), 1e-6)
        g1  = F.softmax(gate1_logits     / temp, dim=-1)    # [B,S,2]
        g2L = F.softmax(gate2_left_logits  / temp, dim=-1)  # [B,S,2]
        g2R = F.softmax(gate2_right_logits / temp, dim=-1)  # [B,S,2]
        if hard:
            g1  = F.one_hot(g1.argmax(-1),  num_classes=2).float()
            g2L = F.one_hot(g2L.argmax(-1), num_classes=2).float()
            g2R = F.one_hot(g2R.argmax(-1), num_classes=2).float()

        # leaf weights [B,S,1]
        w_left  = g1[..., 0:1]
        w_right = g1[..., 1:2]
        w_ll = w_left  * g2L[..., 0:1]
        w_lr = w_left  * g2L[..., 1:2]
        w_rl = w_right * g2R[..., 0:1]
        w_rr = w_right * g2R[..., 1:2]

        # path log-probs stacked on the **leaf axis** => [B,S,4,V]
        stack_lp = torch.stack([
            self._as_log_probs(path_logits["left_left"]),
            self._as_log_probs(path_logits["left_right"]),
            self._as_log_probs(path_logits["right_left"]),
            self._as_log_probs(path_logits["right_right"]),
        ], dim=-2)  # [B,S,4,V]

        # weights stacked the same way => [B,S,4,1]
        W = torch.stack([w_ll, w_lr, w_rl, w_rr], dim=-2).clamp_min(1e-8)

        # log-sum-exp over the 4 leaves => [B,S,V]
        log_mix = torch.logsumexp(W.log() + stack_lp, dim=-2)
        
        # Return the log_mix and properly structured return values for gold aux loss
        return log_mix, {
            "left_left": F.log_softmax(path_logits["left_left"], dim=-1),
            "left_right": F.log_softmax(path_logits["left_right"], dim=-1),
            "right_left": F.log_softmax(path_logits["right_left"], dim=-1),
            "right_right": F.log_softmax(path_logits["right_right"], dim=-1)
        }, g1, g2L, g2R

    def backward_only(self, batch, *, path_selection="hierarchical_gate", loss_scale=1.0):
        """One micro-batch: forward + loss + backward (no optimizer step)"""
        self.model.train()
        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels         = batch["labels"].to(self.device)
        token_weights  = batch.get("token_weights", None)
        if token_weights is not None:
            token_weights = token_weights.to(self.device)

        # --- NEW: use live settings (keeps Phase A if you don't change them) ---
        sel = path_selection or getattr(self, "train_path_selection", "hierarchical_gate")
        # keep model's temp in sync with trainer.gate_temp if present
        if hasattr(self, "gate_temp"):
            self.model.gate_temp = self.gate_temp

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            path_selection=sel
        )
        
        path_logits = outputs["path_logits"]
        gate_info   = outputs["gate"]
        
        if sel in ("hierarchical_gate", "gate_soft") and gate_info is not None:
            # Hierarchical gating losses
            gate1_logits      = gate_info["gate1_logits"]
            gate2_left_logits = gate_info["gate2_left_logits"]
            gate2_right_logits= gate_info["gate2_right_logits"]
            
            log_mix, log_probs_dict, gate1_soft, gate2_left_soft, gate2_right_soft = self._hierarchical_log_mix(
                path_logits, gate1_logits, gate2_left_logits, gate2_right_logits
            )
            
            # Main CE loss (soft mixture)
            ce_loss = self._token_ce(log_mix, labels, token_weights)
            
            # Gold routing auxiliary loss
            gold_aux_loss = self._compute_gold_aux_loss(
                log_probs_dict, gold=labels, mask=attention_mask
            )
            
            # Load balance for both levels
            lb_loss = self._compute_load_balance_loss(gate1_soft, gate2_left_soft, gate2_right_soft)
            
            # Tether to baseline (left_left path)
            tether_loss = self._kl_mean(log_mix, log_probs_dict["left_left"])

            # --- NEW (optional): soft→hard consistency, default off (Phase A unchanged) ---
            lambda_cons = getattr(self, "consistency_lambda", 0.0)
            if lambda_cons > 0.0:
                # Soft gold log-probs used as target (stop-grad)
                soft_lp   = log_mix  # already log-probs of mixture [B,S,V]
                soft_gold = soft_lp[..., :-1, :].gather(-1, labels[..., 1:].unsqueeze(-1)).squeeze(-1)
                if attention_mask is not None:
                    soft_gold = soft_gold * attention_mask[..., 1:].to(soft_gold.dtype)

                # Hard routing gold log-probs (second forward with gate_hard)
                hard_out = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=None,
                    path_selection="gate_hard"
                )
                hard_lp   = hard_out.get("log_probs")
                if hard_lp is None:
                    hard_lp = torch.log_softmax(hard_out["logits"], dim=-1)
                hard_gold = hard_lp[..., :-1, :].gather(-1, labels[..., 1:].unsqueeze(-1)).squeeze(-1)
                if attention_mask is not None:
                    hard_gold = hard_gold * attention_mask[..., 1:].to(hard_gold.dtype)

                cons_loss = F.mse_loss(hard_gold, soft_gold.detach())
            else:
                cons_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        else:
            # Simple path selection (e.g., gate_hard / left_left_only / max_prob)
            final_logits = outputs["logits"]
            log_mix      = torch.log_softmax(final_logits, dim=-1)
            ce_loss      = self._token_ce(log_mix, labels, token_weights)
            gold_aux_loss= torch.tensor(0.0, device=self.device, requires_grad=True)
            lb_loss      = torch.tensor(0.0, device=self.device, requires_grad=True)
            tether_loss  = torch.tensor(0.0, device=self.device, requires_grad=True)
            cons_loss    = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Total loss (consistency is additive and optional)
        total_loss = (
            ce_loss
            + self.gold_aux_coef * gold_aux_loss
            + self.lb_coef       * lb_loss
            + self.tether_coef   * tether_loss
            + getattr(self, "consistency_lambda", 0.0) * cons_loss
        )

        # Backward with scaling
        (total_loss * loss_scale).backward()

        # Accuracy on the distribution we trained against (log_mix)
        with torch.no_grad():
            preds        = log_mix.argmax(dim=-1)
            shift_labels = labels[:, 1:]
            shift_preds  = preds[:, :-1]
            valid        = (shift_labels != -100)
            acc          = ((shift_preds == shift_labels) & valid).sum().float() / (valid.sum().float() + 1e-8)

        # Gate usage stats
        gate_stats = self._compute_gate_stats(gate_info) if gate_info else {}

        return {
            "loss":     float(total_loss.detach()),
            "ce":       float(ce_loss.detach()),
            "gold_aux": float(gold_aux_loss.detach()),
            "lb_loss":  float(lb_loss.detach()),
            "tether":   float(tether_loss.detach()),
            "accuracy": float(acc.detach()),
            **gate_stats
        }
    
    def _compute_gold_aux_loss(
        self,
        path_log_probs,          # dict {'left_left','left_right','right_left','right_right'} of [B,S,V] log-probs
                                # or a stacked tensor [B,S,4,V] / [B,4,S,V]
        gold,                    # labels: Long IDs [B,S] or one-hot [B,S,V] (dense/sparse)
        gate1_logits=None,       # accepted but unused (keep signature compatible)
        gate2_left_logits=None,  # accepted but unused
        gate2_right_logits=None, # accepted but unused
        mask=None,               # optional [B,S] 0/1
    ):
        import torch
        import torch.nn.functional as F

        # ---- 0) Normalize per-path log-probs to 4 tensors [B,S,V] ----
        if isinstance(path_log_probs, dict):
            lp_ll = path_log_probs["left_left"]
            lp_lr = path_log_probs["left_right"]
            lp_rl = path_log_probs["right_left"]
            lp_rr = path_log_probs["right_right"]
            B, S, V = lp_ll.shape
            device = lp_ll.device
        else:
            lp = path_log_probs
            if lp.dim() != 4:
                raise ValueError(f"Expected path_log_probs dim=4, got {lp.shape}")
            # Accept [B,S,4,V] or [B,4,S,V] -> standardize to [B,S,4,V]
            if lp.shape[-2] == 4:               # [B,S,4,V]
                lp_stacked = lp
                B, S, _, V = lp_stacked.shape
            elif lp.shape[1] == 4:              # [B,4,S,V] -> [B,S,4,V]
                lp_stacked = lp.permute(0, 2, 1, 3).contiguous()
                B, S, _, V = lp_stacked.shape
            else:
                raise ValueError(f"Cannot find leaf axis of size 4 in {lp.shape}")
            device = lp_stacked.device
            lp_ll = lp_stacked[..., 0, :]       # [B,S,V]
            lp_lr = lp_stacked[..., 1, :]
            lp_rl = lp_stacked[..., 2, :]
            lp_rr = lp_stacked[..., 3, :]

        # ---- 1) Normalize GOLD to dense Long IDs [B,S] ----
        if not isinstance(gold, torch.Tensor):
            gold = torch.as_tensor(gold, device=device)

        if gold.layout != torch.strided:
            gold = gold.to_dense()

        if gold.dim() == 3 and gold.size(-1) == V:  # one-hot -> ids
            gold = gold.argmax(dim=-1)

        gold = gold.to(dtype=torch.long, device=device).clamp_(0, V - 1)

        # ---- 2) Shift for next-token prediction; build batched index [B,S-1,1] ----
        gold_shift = gold[..., 1:]  # [B,S-1] or [S-1] or [1,S-1]
        if gold_shift.dim() == 1:                 # [S-1]
            gold_shift = gold_shift.unsqueeze(0).expand(B, -1)     # [B,S-1]
        elif gold_shift.size(0) == 1 and B > 1:   # [1,S-1]
            gold_shift = gold_shift.expand(B, -1)                   # [B,S-1]
        index = gold_shift.unsqueeze(-1)                              # [B,S-1,1]

        # Optional mask
        mask_shift = None
        if mask is not None:
            mask_shift = mask.to(device)[..., 1:].to(lp_ll.dtype)     # [B,S-1]

        # ---- 3) Gather per-path gold log-prob: [B,S-1] each ----
        def gather_gold(lp):  # lp: [B,S,V]
            lp_trunc = lp[..., :-1, :]                                # [B,S-1,V]
            # Ensure dims match for gather
            if index.dim() != lp_trunc.dim():
                raise RuntimeError(f"Index dims {index.shape} vs input {lp_trunc.shape}")
            return lp_trunc.gather(-1, index).squeeze(-1)             # [B,S-1]

        g_ll = gather_gold(lp_ll)
        g_lr = gather_gold(lp_lr)
        g_rl = gather_gold(lp_rl)
        g_rr = gather_gold(lp_rr)

        gold_lp_stack = torch.stack([g_ll, g_lr, g_rr, g_rl], dim=-1)  # [B,S-1,4]  (order doesn't matter)
        if mask_shift is not None:
            gold_lp_stack = gold_lp_stack * mask_shift.unsqueeze(-1)

        # ---- 4) Aux objective: encourage all leaves to carry gold probability ----
        gold_aux_loss = -(gold_lp_stack.mean())
        return gold_aux_loss

    def _compute_load_balance_loss(self, gate1_soft, gate2_left_soft, gate2_right_soft):
        """Compute load balance loss for all gates"""
        losses = []
        
        for gate_soft in [gate1_soft, gate2_left_soft, gate2_right_soft]:
            usage = gate_soft.mean(dim=(0, 1))  # [2]
            u = usage + 1e-8
            kl = (u * (u.log() - math.log(0.5))).sum()
            losses.append(kl)
        
        return sum(losses)

    def _compute_gate_stats(self, gate_info):
        """Compute gate usage statistics"""
        if gate_info is None:
            return {}
        
        stats = {}
        
        # First gate stats
        if "gate1" in gate_info:
            gate1_soft = gate_info["gate1"]
            right_pct = gate1_soft[..., 1].mean().item()
            entropy = -(gate1_soft * (gate1_soft.clamp_min(1e-8)).log()).sum(dim=-1).mean().item()
            
            usage = gate1_soft.mean(dim=(0,1))
            u = usage.clamp_min(1e-8)
            usage_kl = (u * (u.log() - math.log(0.5))).sum().item()
            
            stats.update({
                "gate1_right_pct": right_pct,
                "gate1_entropy": entropy,
                "gate1_usage_kl": usage_kl
            })
        
        # Second gate stats
        for gate_name in ["gate2_left", "gate2_right"]:
            if gate_name in gate_info:
                gate_soft = gate_info[gate_name]
                right_pct = gate_soft[..., 1].mean().item()
                entropy = -(gate_soft * (gate_soft.clamp_min(1e-8)).log()).sum(dim=-1).mean().item()
                
                stats[f"{gate_name}_right_pct"] = right_pct
                stats[f"{gate_name}_entropy"] = entropy
        
        # Overall path usage from final weights
        if "final_weights" in gate_info:
            fw = gate_info["final_weights"]
            for path_name, weights in fw.items():
                stats[f"{path_name}_usage"] = weights.mean().item()
        
        return stats

    def train_step(self, batch, optimizer, path_selection="hierarchical_gate"):
        """Single training step with optimizer update"""
        metrics = self.backward_only(batch, path_selection=path_selection, loss_scale=1.0)
        self.optimizer_step(optimizer)
        return metrics

    def evaluate(self, dataloader, path_selection="hierarchical_gate"):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        all_gate_stats = {}
        n = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                    path_selection=path_selection
                )
                
                final_logits = outputs["logits"]
                log_probs = torch.log_softmax(final_logits, dim=-1)
                
                # CE loss
                ce_loss = self._token_ce(log_probs, labels)
                total_loss += ce_loss.item()
                
                # Accuracy
                preds = log_probs.argmax(dim=-1)
                shift_labels = labels[:, 1:]
                shift_preds = preds[:, :-1]
                valid = (shift_labels != -100)
                acc = ((shift_preds == shift_labels) & valid).sum().float() / (valid.sum().float() + 1e-8)
                total_acc += acc.item()
                
                # Gate stats
                gate_stats = self._compute_gate_stats(outputs.get("gate"))
                for key, value in gate_stats.items():
                    if key not in all_gate_stats:
                        all_gate_stats[key] = 0.0
                    all_gate_stats[key] += value
                
                n += 1

        avg_loss = total_loss / max(n, 1)
        avg_acc = total_acc / max(n, 1)
        ppl = math.exp(avg_loss)
        
        # Average gate stats
        for key in all_gate_stats:
            all_gate_stats[key] /= max(n, 1)

        return {
            "loss": avg_loss,
            "perplexity": ppl,
            "accuracy": avg_acc,
            **all_gate_stats
        }

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

    def generate_sample(self, prompt, max_length=50, path_selection="hierarchical_gate", 
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
                next_token_logits = outputs["logits"][:, -1, :] / temperature
                
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
    
