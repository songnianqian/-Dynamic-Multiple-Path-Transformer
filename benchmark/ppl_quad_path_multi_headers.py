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
"""
ppl_model_multipath.py  (multi-path GPT-2 with hierarchical gated mixture)

Fixed version that resolves tensor shape mismatch in head selection processing.

The issue was in the alignment of tensors when processing per-path head data.
The fix ensures proper tensor dimension alignment throughout the processing pipeline.
"""
import os, json, argparse, math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.serialization import add_safe_globals
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from collections import defaultdict, Counter
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

add_safe_globals([GPT2Config])

# ✅ Multi-path model
from quad_path_multi_headers.multi_path_model import HierarchicalMultiPathGPT2


def _ensure_log_probs(t: torch.Tensor) -> torch.Tensor:
    """Convert to log-probs iff needed using logsumexp≈0 heuristic."""
    # t shape [B,S,V] or [S,V] or [V]
    sample = t[0, 0] if t.dim() == 3 else (t[0] if t.dim() == 2 else t)
    lse = torch.logsumexp(sample.float(), dim=-1)
    is_logprob = torch.allclose(lse, torch.tensor(0.0, device=sample.device), atol=1e-3)
    
    if is_logprob:
        return t
    else:
        result = F.log_softmax(t, dim=-1)
        return result

def _extract_log_probs(output) -> torch.Tensor:
    """Return log-probs [B,S,V] regardless of what the model returned."""
    if isinstance(output, dict):
        t = output.get("logits")
        if t is None:
            raise ValueError("Model output missing 'logits'.")
        return _ensure_log_probs(t)
    # HF-style object
    return _ensure_log_probs(output.logits)

def _extract_path_probs(output) -> torch.Tensor:
    """Extract path probabilities from model output if available."""
    if isinstance(output, dict):
        # Check for gate structure with final_weights
        if "gate" in output:
            gate = output["gate"]
            if isinstance(gate, dict) and "final_weights" in gate:
                final_weights = gate["final_weights"]
                if isinstance(final_weights, dict):
                    # Extract weights for each path and combine
                    w_ll = final_weights.get("left_left")    # [B, S, 1]
                    w_lr = final_weights.get("left_right")   # [B, S, 1] 
                    w_rl = final_weights.get("right_left")   # [B, S, 1]
                    w_rr = final_weights.get("right_right")  # [B, S, 1]
                    
                    if all(w is not None for w in [w_ll, w_lr, w_rl, w_rr]):
                        # Combine into [B, S, 4] path probabilities
                        quad_probs = torch.cat([w_ll, w_lr, w_rl, w_rr], dim=-1)
                        return quad_probs
        
        # Fallback: try other possible keys
        for key in ["path_probs", "path_weights", "routing_probs"]:
            if key in output:
                value = output[key]
                if isinstance(value, torch.Tensor):
                    return value
        
        return None
    
    # Try multiple possible attributes
    for attr in ["path_probs", "path_weights", "gate_probs", "routing_probs"]:
        if hasattr(output, attr):
            return getattr(output, attr)
    return None

def _extract_head_probs(output) -> torch.Tensor:
    """Extract head probabilities/indices from model output if available."""
    if isinstance(output, dict):
        # Try the actual head selection outputs from your model
        if "head_top_idx_combined" in output and output["head_top_idx_combined"] is not None:
            return output["head_top_idx_combined"]
        
        if "head_top_idx" in output and output["head_top_idx"] is not None:
            head_idx = output["head_top_idx"]
            if isinstance(head_idx, dict):
                # If it's a dict per path, we'll handle this differently
                return head_idx
            elif isinstance(head_idx, torch.Tensor):
                return head_idx
        
        # Try other possible keys
        for key in ["head_probs", "head_weights", "lm_head_probs", "mixture_weights"]:
            if key in output:
                value = output[key]
                if isinstance(value, torch.Tensor):
                    return value
        
        return None
    
    # Try multiple possible attributes
    for attr in ["head_probs", "head_weights", "lm_head_probs", "mixture_weights"]:
        if hasattr(output, attr):
            return getattr(output, attr)
    return None

def _derive_head_from_path(path_selections: torch.Tensor) -> torch.Tensor:
    """
    Derive head selections from path selections.
    Based on: header 0=left paths, header 1=right paths
    Paths: [left_left, left_right, right_left, right_right] = [0, 1, 2, 3]
    """
    if path_selections is None:
        return None
    
    # Convert path indices to head indices
    # left_left (0) -> head 0, left_right (1) -> head 0
    # right_left (2) -> head 1, right_right (3) -> head 1
    head_selections = (path_selections >= 2).long()  # 0,1 -> 0 (left), 2,3 -> 1 (right)
    return head_selections

def _get_path_selection_from_probs(path_probs: torch.Tensor) -> torch.Tensor:
    """
    Convert path probabilities to path selections.
    Assumes hierarchical structure: [left_left, left_right, right_left, right_right]
    """
    if path_probs is None:
        return None
    
    # path_probs shape: [B, S, 4] for quad paths
    # Return the path index with highest probability
    return path_probs.argmax(dim=-1)  # [B, S]

def _get_head_selection_from_probs(head_data: torch.Tensor) -> torch.Tensor:
    """
    Get the selected head index. 
    If head_data contains indices, return directly.
    If head_data contains probabilities, return argmax.
    """
    if head_data is None:
        return None
    
    # If head_data looks like indices (integers), return directly
    if head_data.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
        return head_data
    
    # If head_data contains probabilities, take argmax
    if head_data.dim() >= 2 and head_data.size(-1) > 1:
        return head_data.argmax(dim=-1)
    
    # Otherwise assume it's already indices
    return head_data

@torch.no_grad()
def score_texts_hf_stride_concat_with_stats(model, tokenizer, texts, device, *,
                                            max_length=1024, stride=512,
                                            path_selection="hierarchical_gate"):
    """
    Enhanced scoring function that tracks path and header usage statistics.
    FIXED: Proper tensor alignment to avoid shape mismatch errors.
    """
    sequences = []
    total_scored = 0
    total_logprob = 0.0
    total_correct = 0

    # Statistics tracking
    path_usage_counts = Counter()                           # Track which paths are used (0..3)
    head_usage_by_path = defaultdict(Counter)              # Track head usage per path
    global_head_usage = Counter()                          # Track overall header usage across all paths
    path_probs_all = []                                    # Store path probabilities for analysis
    head_probs_all = []                                    # Store head probabilities for analysis

    # === Global LM-header usage (all paths combined) ===
    global_head_hard_wins   = Counter()                    # head_id -> int (argmax wins)
    global_head_soft_credit = defaultdict(float)           # head_id -> float (prob credit)
    total_head_tokens       = 0

    # Consistent path order used across the file
    path_names = ["left_left", "left_right", "right_left", "right_right"]

    # Header competition tracking (indices selected per path)
    header_competition_stats = defaultdict(list)

    eos = tokenizer.eos_token_id
    chunks = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids[0]
        chunks.append(ids)
        chunks.append(torch.tensor([eos], dtype=torch.long))
    token_ids = torch.cat(chunks, dim=0).to(device)  # [N]
    N = int(token_ids.numel())

    sequences.append({
        "text_index": 0,
        "text": "<CONCAT>",
        "token_ids": [int(x) for x in token_ids.tolist()],
        "scores": []
    })

    begin = 0
    while begin < N:
        end = min(begin + max_length, N)
        window = token_ids[begin:end].unsqueeze(0)      # [1, W]
        attn   = torch.ones_like(window, device=device) # [1, W]

        # Forward pass with enhanced output - ENABLE head index return
        out = model(input_ids=window, attention_mask=attn, labels=None,
                    path_selection=path_selection, return_head_indices=True)

        # DEBUG (first window only)
        if begin == 0:
            print(f"DEBUG: Model output type: {type(out)}")
            if isinstance(out, dict):
                print(f"DEBUG: Model output keys: {list(out.keys())}")
            print(f"DEBUG: Model use_head_mixture: {getattr(model, 'use_head_mixture', 'Not set')}")
            print(f"DEBUG: Model head_allocation: {getattr(model, 'head_allocation', 'Not set')}")

        logprobs_full = _extract_log_probs(out)   # [1, W, V]
        path_probs    = _extract_path_probs(out)  # [1, W, 4] or None
        head_data     = _extract_head_probs(out)  # [1, W, H] (probs) | [1, W] (indices) | dict per path | None

        # CRITICAL FIX: Ensure consistent slicing for all tensors
        W = logprobs_full.size(1)  # Full window size
        
        logprobs_full = logprobs_full[:, :-1, :]  # [1, W-1, V]
        labels        = window[:, 1:]             # [1, W-1]
        
        # Align aux tensors to [1, W-1, ...]
        if path_probs is not None:
            if path_probs.size(1) == W:
                path_probs = path_probs[:, :-1, :]
            elif path_probs.size(1) == W-1:
                pass  # Already correct size
            else:
                print(f"WARNING: path_probs size {path_probs.shape} doesn't match expected window size")
                path_probs = path_probs[:, :W-1, :] if path_probs.size(1) > W-1 else path_probs
        
        if head_data is not None:
            if isinstance(head_data, torch.Tensor):
                if head_data.dim() == 2 and head_data.size(1) == W:  # [1, W] indices
                    head_data = head_data[:, :-1]  # [1, W-1]
                elif head_data.dim() == 3 and head_data.size(1) == W:  # [1, W, H] probs
                    head_data = head_data[:, :-1, :]  # [1, W-1, H]
                elif head_data.dim() == 1 and head_data.size(0) == W:  # [W] indices
                    head_data = head_data[:-1]  # [W-1]
            elif isinstance(head_data, dict):
                # Per-path head data - align each tensor
                for pname in head_data:
                    td = head_data[pname]
                    if td.dim() == 2 and td.size(1) == W:  # [1, W] indices
                        head_data[pname] = td[:, :-1]  # [1, W-1]
                    elif td.dim() == 3 and td.size(1) == W:  # [1, W, H] probs
                        head_data[pname] = td[:, :-1, :]  # [1, W-1, H]

        Wm1 = logprobs_full.size(1)  # Should be W-1
        window_len = end - begin
        eff_stride = min(stride, max(0, window_len - 1))
        target_len = max(0, window_len - eff_stride)

        if target_len > 0 and Wm1 > 0:
            ignore = Wm1 - target_len
            labels_scored = labels.clone()
            if ignore > 0:
                labels_scored[:, :ignore] = -100

            V = logprobs_full.size(-1)
            flat_logp   = logprobs_full.reshape(-1, V)   # [W-1, V]
            flat_labels = labels_scored.reshape(-1)      # [W-1]
            keep_mask   = (flat_labels != -100)          # [W-1]
            kept_labels = flat_labels[keep_mask]         # [K]
            kept_logp   = flat_logp[keep_mask, :]        # [K, V]

            # ----- PATH SELECTIONS -----
            if path_probs is not None:
                flat_path_probs = path_probs.reshape(-1, path_probs.size(-1))  # [W-1, 4]
                kept_path_probs = flat_path_probs[keep_mask, :]                # [K, 4]
                path_selections = _get_path_selection_from_probs(
                    kept_path_probs.unsqueeze(0)
                ).squeeze(0)                                                   # [K]
                # Count path usage
                for p in path_selections.cpu().numpy():
                    path_usage_counts[int(p)] += 1
                # stash
                path_probs_all.append(kept_path_probs.cpu().numpy())
            else:
                path_selections = None

            # ----- HEAD SELECTIONS (FIXED) -----
            head_selections = None
            kept_head_data = None  # will hold [K, H] if we have probs

            if head_data is not None:
                if isinstance(head_data, dict):
                    # Per-path head indices or probs: {'left_left': [1,W-1, H_sub or 1], ...}
                    if path_selections is not None:
                        combined_head_selections = torch.zeros(Wm1, dtype=torch.long, device=window.device)

                        for pidx, pname in enumerate(path_names):
                            if pname not in head_data:
                                continue
                            td = head_data[pname]  # [1, W-1] indices OR [1, W-1, H_sub] probs
                            
                            # CRITICAL FIX: Ensure td is properly sized
                            if td.size(1) != Wm1:
                                print(f"WARNING: head_data[{pname}] has size {td.shape}, expected [..., {Wm1}]")
                                if td.size(1) > Wm1:
                                    td = td[:, :Wm1]  # Truncate
                                else:
                                    # Pad with zeros if needed
                                    pad_size = Wm1 - td.size(1)
                                    if td.dim() == 2:  # indices
                                        pad = torch.zeros(1, pad_size, dtype=td.dtype, device=td.device)
                                    else:  # probs
                                        pad = torch.zeros(1, pad_size, td.size(-1), dtype=td.dtype, device=td.device)
                                    td = torch.cat([td, pad], dim=1)
                            
                            if td.dim() == 3:
                                # probabilities per path; we will handle soft credit below
                                # for hard wins we take argmax per-row later after masking
                                pass
                            
                            # Build a path mask over all W-1 positions (pre-keep)
                            full_path_sel = torch.full((Wm1,), -1, dtype=torch.long, device=window.device)
                            if ignore > 0:
                                full_path_sel[:ignore] = 0
                            if ignore < Wm1:
                                path_sel_aligned = path_selections.to(window.device)
                                available_positions = min(len(path_sel_aligned), Wm1 - ignore)
                                full_path_sel[ignore:ignore+available_positions] = path_sel_aligned[:available_positions]
                            
                            path_mask_full = (full_path_sel == pidx)            # [W-1]

                            if td.dim() == 2:  # [1, W-1] indices
                                td = td.squeeze(0)                              # [W-1]
                                # FIXED: Both tensors now have same shape [W-1]
                                combined_head_selections[path_mask_full] = td[path_mask_full]

                        head_selections = combined_head_selections[keep_mask]    # [K]
                    # if path_selections is None we cannot combine; skip
                elif isinstance(head_data, torch.Tensor):
                    if head_data.dim() == 2:  # [1, W-1] indices
                        flat = head_data.reshape(-1)           # [W-1]
                        head_selections = flat[keep_mask]      # [K]
                    elif head_data.dim() == 3:  # [1, W-1, H] probs
                        flat = head_data.reshape(-1, head_data.size(-1))  # [W-1, H]
                        kept_head_data = flat[keep_mask, :]                # [K, H]
                        head_selections = kept_head_data.argmax(dim=-1)    # [K]
                        head_probs_all.append(kept_head_data.cpu().numpy())
                    elif head_data.dim() == 1:  # [W-1] indices
                        head_selections = head_data[keep_mask]
                else:
                    print(f"WARNING: Unexpected head_data type: {type(head_data)}")

            # ===== GLOBAL LM-HEADER TALLIES (wins + soft credit) =====
            # Hard wins: just use head_selections (argmax already if probs case)
            if head_selections is not None:
                for h in head_selections.cpu().numpy():
                    global_head_hard_wins[int(h)] += 1
                total_head_tokens += int(head_selections.numel())

            # Soft credit:
            if head_data is not None:
                if kept_head_data is not None:
                    # Combined probs case: kept_head_data is [K, H]
                    for row in kept_head_data.cpu().numpy():
                        for hid, p in enumerate(row):
                            global_head_soft_credit[int(hid)] += float(p)
                elif isinstance(head_data, dict) and path_selections is not None:
                    # Dict probs per path: sum credit only where that path is selected
                    for pidx, pname in enumerate(path_names):
                        if pname not in head_data:
                            continue
                        td = head_data[pname]
                        if td.dim() == 3:  # [1, W-1, H_sub]
                            flat = td.reshape(-1, td.size(-1))     # [W-1, H_sub]
                            kept = flat[keep_mask, :]              # [K, H_sub]
                            # mask positions that chose this path
                            mask_np = (path_selections.cpu().numpy() == pidx)
                            if mask_np.any() and hasattr(model, 'head_allocation') and pname in model.head_allocation:
                                kept_this_path = kept[torch.from_numpy(mask_np)]
                                subset_ids = model.head_allocation[pname]
                                for row in kept_this_path.cpu().numpy():  # [H_sub]
                                    for j, p in enumerate(row):
                                        if j < len(subset_ids):
                                            global_head_soft_credit[int(subset_ids[j])] += float(p)

            # ===== PER-PATH + GLOBAL HEAD USAGE (win counts) =====
            if path_selections is not None and head_selections is not None:
                for p, h in zip(path_selections.cpu().numpy(), head_selections.cpu().numpy()):
                    head_usage_by_path[int(p)][int(h)] += 1
                    global_head_usage[int(h)] += 1

                # Track raw selected heads per path (competition view)
                if isinstance(head_data, dict):
                    for i, pname in enumerate(path_names):
                        mask_np = (path_selections.cpu().numpy() == i)
                        if mask_np.any():
                            selected_heads = head_selections[torch.from_numpy(mask_np)].cpu().numpy().tolist()
                            header_competition_stats[pname].extend(selected_heads)
            elif head_selections is not None:
                # No path info → bucket under -1
                for h in head_selections.cpu().numpy():
                    head_usage_by_path[-1][int(h)] += 1
                    global_head_usage[int(h)] += 1
            elif path_selections is not None:
                # No explicit head data → derive if you have a rule; else skip
                pass

            # ===== TOKEN LOGPROBS / ACC =====
            token_logprobs = kept_logp.gather(1, kept_labels.view(-1, 1)).squeeze(1)  # [K]
            preds = kept_logp.argmax(dim=-1)                                          # [K]
            correct = (preds == kept_labels).sum().item()
            total_correct += correct

            # ===== RECORD PER-TOKEN SCORES =====
            ptr = 0
            for k in range(Wm1):
                if k < ignore:
                    continue
                global_pos = begin + 1 + k
                tok_id     = int(window[0, k+1].item())
                lp         = float(token_logprobs[ptr].item())
                pred_id    = int(preds[ptr].item())
                is_correct = (pred_id == tok_id)

                score_entry = {
                    "pos": int(global_pos),
                    "token_id": tok_id,
                    "logprob": lp,
                    "predicted_id": pred_id,
                    "correct": is_correct
                }
                if path_selections is not None:
                    pidx = int(path_selections[ptr].item())
                    score_entry["selected_path"] = pidx
                    score_entry["path_probs"] = kept_path_probs[ptr].cpu().numpy().tolist()
                if head_selections is not None:
                    hidx = int(head_selections[ptr].item())
                    score_entry["selected_head"] = hidx
                    if isinstance(head_data, dict):
                        score_entry["head_source"] = "per_path_explicit"
                    elif head_data is not None:
                        score_entry["head_source"] = "combined_explicit"
                    else:
                        score_entry["head_source"] = "derived_or_unknown"

                sequences[0]["scores"].append(score_entry)
                total_scored += 1
                total_logprob += lp
                ptr += 1

        if end == N:
            break
        begin = end - eff_stride

    # ===== Aggregate usage statistics =====
    usage_stats = {}

    # Path usage
    if path_usage_counts:
        total_path_selections = sum(path_usage_counts.values())
        path_usage_percentages = {}
        for i in range(4):
            count = path_usage_counts.get(i, 0)
            percentage = (count / total_path_selections) * 100 if total_path_selections > 0 else 0.0
            path_usage_percentages[path_names[i]] = {
                "count": count,
                "percentage": round(percentage, 2)
            }
        usage_stats["path_usage"] = {
            "total_selections": total_path_selections,
            "usage_by_path": path_usage_percentages
        }

    # Global header usage (win counts from per-path/global tallies)
    if global_head_usage:
        total_global_head_selections = sum(global_head_usage.values())
        global_head_percentages = {}
        n_heads_cfg = getattr(model.config, "n_lm_perceptrons", 8)
        for head_idx in range(int(n_heads_cfg)):
            count = global_head_usage.get(head_idx, 0)
            percentage = (count / total_global_head_selections) * 100 if total_global_head_selections > 0 else 0.0
            global_head_percentages[f"head_{head_idx}"] = {
                "count": count,
                "percentage": round(percentage, 2),
                "win_count": count
            }
        usage_stats["global_header_usage"] = {
            "total_selections": total_global_head_selections,
            "usage_by_header": global_head_percentages
        }

    # Head usage percentages by path
    if head_usage_by_path:
        head_usage_stats = {}
        for pidx, head_counts in head_usage_by_path.items():
            total_head_sel = sum(head_counts.values())
            pname = path_names[pidx] if 0 <= pidx < 4 else ("no_path_info" if pidx == -1 else f"path_{pidx}")
            head_percentages = {}
            for hidx, cnt in head_counts.items():
                pct = (cnt / total_head_sel) * 100 if total_head_sel > 0 else 0.0
                head_percentages[f"head_{hidx}"] = {
                    "count": cnt,
                    "percentage": round(pct, 2),
                    "win_count": cnt
                }
            # Competition summary
            competition_stats = {}
            if pname in header_competition_stats:
                sels = header_competition_stats[pname]
                if sels:
                    cc = Counter(sels)
                    total_s = len(sels)
                    entropy = 0.0
                    for c in cc.values():
                        p = c / total_s
                        if p > 0:
                            entropy -= p * math.log2(p)
                    competition_stats = {
                        "competing_headers": sorted(list(cc.keys())),
                        "num_competing": len(cc),
                        "selection_distribution": dict(cc),
                        "entropy": round(entropy, 3)
                    }
            head_usage_stats[pname] = {
                "total_selections": total_head_sel,
                "most_used_head": max(head_counts, key=head_counts.get) if head_counts else None,
                "usage_by_head": head_percentages,
                "competition_stats": competition_stats
            }
        usage_stats["head_usage_by_path"] = head_usage_stats

    # Probability stats (optional)
    if path_probs_all:
        all_path_probs = np.concatenate(path_probs_all, axis=0)
        usage_stats["path_probability_stats"] = {
            "mean_probs": np.mean(all_path_probs, axis=0).tolist(),
            "std_probs": np.std(all_path_probs, axis=0).tolist(),
            "entropy_mean": float(np.mean(-np.sum(all_path_probs * np.log(all_path_probs + 1e-10), axis=1)))
        }
    if head_probs_all:
        all_head_probs = np.concatenate(head_probs_all, axis=0)
        usage_stats["head_probability_stats"] = {
            "mean_probs": np.mean(all_head_probs, axis=0).tolist(),
            "std_probs": np.std(all_head_probs, axis=0).tolist(),
            "entropy_mean": float(np.mean(-np.sum(all_head_probs * np.log(all_head_probs + 1e-10), axis=1)))
        }

    # Expose global header tallies for printing in main
    usage_stats["global_header_hard_wins"] = {int(h): int(c) for h, c in global_head_hard_wins.items()}
    usage_stats["global_header_soft_credit"] = {int(h): float(c) for h, c in global_head_soft_credit.items()}
    usage_stats["global_header_total_tokens"] = int(total_head_tokens)

    return sequences, total_scored, total_logprob, total_correct, usage_stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # model specifics
    ap.add_argument("--pretrained_model", default="gpt2")
    ap.add_argument("--split_at_layer_1", type=int, default=6)
    ap.add_argument("--split_at_layer_2", type=int, default=9)
    ap.add_argument("--gate_temp", type=float, default=1.0)
    ap.add_argument("--gate_hidden", type=int, default=256)
    ap.add_argument("--n_lm_perceptrons", type=int, default=8)
    ap.add_argument("--use_head_mixture", action="store_true",
                    help="Enable per-path LM-head mixture (requires head_allocation).")
    ap.add_argument("--head_allocation", type=str, default=None,
                    help="Path to JSON mapping leaf names to head indices (e.g., 3/1/2/2).")
    ap.add_argument("--path_selection", default="hierarchical_gate",
                    choices=["hierarchical_gate", "gate_soft", "gate_hard", "left_left_only", "max_prob"])
    args = ap.parse_args()

    with open(args.inputs, "r", encoding="utf-8") as f:
        inp = json.load(f)
    meta   = inp.get("meta", {})
    texts  = inp.get("texts", [])
    if not texts:
        raise SystemExit("No texts found in inputs.json")

    tok_name   = meta.get("tokenizer", args.pretrained_model)
    max_length = int(meta.get("max_length", 1024))
    stride     = int(meta.get("stride",     512))

    device    = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build config consistent with training
    config = GPT2Config.from_pretrained(tok_name)
    config.n_positions = max_length
    config.n_lm_perceptrons = args.n_lm_perceptrons
    if not hasattr(config, "_attn_implementation") or config._attn_implementation is None:
        config._attn_implementation = "eager"

    # Optional head allocation
    head_alloc = None
    if args.head_allocation:
        with open(args.head_allocation, "r", encoding="utf-8") as f:
            head_alloc = json.load(f)

    # Instantiate multi-path model (loads HF weights internally)
    model = HierarchicalMultiPathGPT2(
        config=config,
        pretrained_model=tok_name,
        split_at_layer_1=args.split_at_layer_1,
        split_at_layer_2=args.split_at_layer_2,
        gate_temp=args.gate_temp,
        gate_hidden=args.gate_hidden,
        head_allocation=head_alloc,
    ).to(device).eval()
    model.use_head_mixture = bool(args.use_head_mixture)

    # Load checkpoint (training ckpt with model_state_dict)
    ckpt = None
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except Exception as e1:
        try:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            print("[warn] Loaded checkpoint with weights_only=False (trusted source).")
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load checkpoint.\n"
                f"weights_only=True error: {e1}\n"
                f"weights_only=False error: {e2}"
            )

    state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else None
    if state is None:
        state = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    if state is None and isinstance(ckpt, dict):
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt
    if state is None:
        raise RuntimeError(f"Checkpoint missing state dict. Keys: {list(ckpt.keys())[:10]}")

    incompat = model.load_state_dict(state, strict=False)
    try:
        missing, unexpected = incompat.missing_keys, incompat.unexpected_keys
    except AttributeError:
        missing, unexpected = incompat  # for older torch
    print(f"Loaded checkpoint. missing={len(missing)}, unexpected={len(unexpected)}")

    # Score with enhanced statistics
    sequences, num_scored, sum_logprob, num_correct, usage_stats = score_texts_hf_stride_concat_with_stats(
        model, tokenizer, texts, device,
        max_length=max_length, stride=stride,
        path_selection=args.path_selection
    )
    
    out = {
        "meta": {
            "tokenizer": tok_name,
            "max_length": max_length,
            "stride": stride,
            "log_base": "e",
            "model": "multi_path",
            "path_selection": args.path_selection,
            "split_at_layer_1": args.split_at_layer_1,
            "split_at_layer_2": args.split_at_layer_2,
            "n_lm_perceptrons": args.n_lm_perceptrons,
            "use_head_mixture": bool(args.use_head_mixture),
            "head_allocation": head_alloc,
        },
        "totals": {
            "num_scored_tokens": int(num_scored),
            "sum_logprob": float(sum_logprob),
            "num_correct_predictions": int(num_correct),
            "perplexity": math.exp(-sum_logprob / max(1, num_scored))
        },
        "usage_statistics": usage_stats,  # New section with path and head usage stats
        "sequences": sequences
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    
    # Print usage statistics summary to console
    print(f"Wrote {args.out} with {num_scored} scored tokens, {num_correct} correct predictions. "
          f"PPL={out['totals']['perplexity']:.4f}")
    
    print("\n" + "="*60)
    print("USAGE SUMMARY")
    print("="*60)
    
    if "path_usage" in usage_stats:
        print("\nPATH USAGE:")
        path_data = usage_stats["path_usage"]["usage_by_path"]
        # Sort by percentage for better readability
        sorted_paths = sorted(path_data.items(), key=lambda x: x[1]['percentage'], reverse=True)
        for path_name, stats in sorted_paths:
            print(f"  {path_name:12}: {stats['percentage']:6.2f}% ({stats['count']:,} tokens)")
    
    if "head_usage_by_path" in usage_stats:
        print("\nHEAD USAGE BY PATH:")
        for path_name, path_stats in usage_stats["head_usage_by_path"].items():
            most_used = path_stats.get('most_used_head')
            total = path_stats['total_selections']
            
            # Get the percentage of the most used head
            most_used_pct = 0
            if most_used is not None:
                head_key = f"head_{most_used}"
                if head_key in path_stats["usage_by_head"]:
                    most_used_pct = path_stats["usage_by_head"][head_key]["percentage"]
            
            print(f"  {path_name:12}: Head-{most_used} ({most_used_pct:5.1f}%) | Total: {total:,} tokens")
    
    print("\nHEADER USAGE (GLOBAL, ALL PATHS COMBINED):")
    n_heads_cfg = int(getattr(model.config, "n_lm_perceptrons", 8))

    wins_dict  = usage_stats.get("global_header_hard_wins", {})
    soft_dict  = usage_stats.get("global_header_soft_credit", {})
    total_hdr  = usage_stats.get("global_header_total_tokens",
                                sum(wins_dict.values()))

    for hid in range(n_heads_cfg):
        wins = int(wins_dict.get(hid, 0))
        pct  = (100.0 * wins / max(1, total_hdr))
        print(f"  head_{hid:<2}: {wins:8d} wins ({pct:6.2f}%)")

    print("\nHEADER SOFT CREDIT (fractional, sums ~ tokens):")
    soft_total = max(1.0, float(total_hdr))
    for hid in range(n_heads_cfg):
        credit = float(soft_dict.get(hid, 0.0))
        pct    = 100.0 * credit / soft_total
        print(f"  head_{hid:<2}: {credit:10.2f} credit ({pct:6.2f}%)")

    # ---- OVERALL (print once, not inside the loop) ----
    if "path_usage" in usage_stats:
        path_data = usage_stats["path_usage"]["usage_by_path"]
        total_tokens = usage_stats["path_usage"]["total_selections"]
        num_paths = len([p for p in path_data.values() if p['count'] > 0])

        print("\nOVERALL:")
        print(f"  Total analyzed tokens: {total_tokens:,}")
        print(f"  Active paths: {num_paths}/4")

        path_percentages = [stats['percentage'] for stats in path_data.values()]
        balance_score = 100 - (max(path_percentages) - min(path_percentages)) if path_percentages else 0.0
        print(f"  Path balance score: {balance_score:.1f}/100 (higher = more balanced)")

    print("="*60)


if __name__ == "__main__":
    main()