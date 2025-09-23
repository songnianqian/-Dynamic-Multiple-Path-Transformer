#!/usr/bin/env python3
"""
ppl_dual_path_multi_headers.py - Perplexity evaluation for DualPathMultiHeadersGPT2

%run ppl_dual_path_multi_headers.py \
 --inputs "/content/drive/My Drive/Project1/DMPT/benchmark/wikitext103_test_inputs.json" \
    --checkpoint "/content/drive/My Drive/Project1/quad_path_headers_checkpoints/checkpoint_step_6000.pt" \
    --out "/content/drive/My Drive/Project1/DMPTresult/result.json" \
    --split_at_layer 6 --n_lm_perceptrons 8 --head_allocation "/content/drive/My Drive/Project1/DMPT/benchmark/head_allocation.json" \
    --path_selection hierarchical_gate --use_head_mixture

Evaluates dual-path GPT models with multiple LM headers and provides detailed
statistics on path selection and header usage.
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

# Import the dual path model
from dual_path_multi_headers.dual_path_multi_headers_model import DualPathMultiHeadersGPT2

def _extract_path_indices(output):
    """Return [B,S] hard path indices if available."""
    if isinstance(output, dict):
        api = output.get("active_path_idx", None)
        if isinstance(api, torch.Tensor):
            return api
    return None

def _ensure_log_probs(t: torch.Tensor) -> torch.Tensor:
    """Convert to log-probs iff needed using logsumexp≈0 heuristic."""
    sample = t[0, 0] if t.dim() == 3 else (t[0] if t.dim() == 2 else t)
    lse = torch.logsumexp(sample.float(), dim=-1)
    is_logprob = torch.allclose(lse, torch.tensor(0.0, device=sample.device), atol=1e-3)
    
    if is_logprob:
        return t
    else:
        result = F.log_softmax(t, dim=-1)
        return result

def _extract_log_probs(output) -> torch.Tensor:
    """Return log-probs [B,S,V] from dual model output."""
    if isinstance(output, dict):
        # DualPath model returns log_probs directly
        if "log_probs" in output and output["log_probs"] is not None:
            return output["log_probs"]
        
        t = output.get("logits")
        if t is None:
            raise ValueError("Model output missing 'logits'.")
        return _ensure_log_probs(t)
    
    # Fallback for HF-style output
    return _ensure_log_probs(output.logits)

def _extract_path_probs(output) -> torch.Tensor:
    """Extract path probabilities [B,S,2] from dual model output."""
    if not isinstance(output, dict):
        return None
    
    if "gate" in output:
        gate_info = output["gate"]
        if isinstance(gate_info, dict):
            # Try direct gate weights [B, S, 2]
            if "gate" in gate_info:
                return gate_info["gate"]
            
            # Try final_weights format
            elif "final_weights" in gate_info:
                fw = gate_info["final_weights"]
                w_left = fw.get("left")   # [B, S, 1]
                w_right = fw.get("right") # [B, S, 1]
                
                if w_left is not None and w_right is not None:
                    return torch.cat([w_left, w_right], dim=-1)  # [B, S, 2]
    
    # Fallback keys
    for key in ["path_probs", "path_weights", "routing_probs"]:
        if key in output:
            value = output[key]
            if isinstance(value, torch.Tensor):
                return value
    
    return None

def _extract_head_probs(output) -> torch.Tensor:
    """Extract head probabilities/indices from dual model output."""
    if not isinstance(output, dict):
        return None
    
    # Try combined head indices first
    if "head_top_idx_combined" in output and output["head_top_idx_combined"] is not None:
        return output["head_top_idx_combined"]
    
    # Try per-path head indices
    if "head_top_idx" in output and output["head_top_idx"] is not None:
        head_idx = output["head_top_idx"]
        if isinstance(head_idx, dict):
            return head_idx  # Dict with left/right keys
        elif isinstance(head_idx, torch.Tensor):
            return head_idx
    
    # Fallback keys
    for key in ["head_probs", "head_weights", "lm_head_probs", "mixture_weights"]:
        if key in output:
            value = output[key]
            if isinstance(value, torch.Tensor):
                return value
    
    return None

def _get_path_selection_from_probs(path_probs: torch.Tensor) -> torch.Tensor:
    """Convert path probabilities to path selections."""
    if path_probs is None:
        return None
    # path_probs shape: [B, S, 2] for dual paths
    return path_probs.argmax(dim=-1)  # [B, S]

def _get_head_selection_from_probs(head_data: torch.Tensor) -> torch.Tensor:
    """Get the selected head index."""
    if head_data is None:
        return None
    
    # If already indices, return directly
    if head_data.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
        return head_data
    
    # If probabilities, take argmax
    if head_data.dim() >= 2 and head_data.size(-1) > 1:
        return head_data.argmax(dim=-1)
    
    return head_data

@torch.no_grad()
def score_texts_dual_with_stats(model, tokenizer, texts, device, *,
                                max_length=1024, stride=512,
                                path_selection="gate_soft"):
    """
    Enhanced scoring for DualPathMultiHeadersGPT2 with detailed stats.
    Uses active_path_idx (hard routing) when available and merges per-path head winners.
    """
    from collections import Counter, defaultdict
    import math
    import numpy as np
    import torch.nn.functional as F

    def _extract_path_indices_from_out(output):
        if isinstance(output, dict):
            api = output.get("active_path_idx", None)
            if isinstance(api, torch.Tensor):
                return api
        return None

    sequences = []
    total_scored = 0
    total_logprob = 0.0
    total_correct = 0

    # Stats
    path_usage_counts   = Counter()
    head_usage_by_path  = defaultdict(Counter)
    global_head_usage   = Counter()
    path_probs_all      = []
    head_probs_all      = []
    global_head_hard_wins   = Counter()
    global_head_soft_credit = defaultdict(float)
    total_head_tokens       = 0
    header_competition_stats = defaultdict(list)

    path_names = ["left", "right"]

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

        # Forward pass
        try:
            out = model(input_ids=window, attention_mask=attn, labels=None,
                        path_selection=path_selection, return_head_indices=True)
        except TypeError:
            out = model(input_ids=window, attention_mask=attn, labels=None,
                        path_selection=path_selection)

        # DEBUG (first window only)
        if begin == 0:
            print(f"DEBUG: Model output type: {type(out)}")
            if isinstance(out, dict):
                print(f"DEBUG: Model output keys: {list(out.keys())}")
            print(f"DEBUG: Model use_head_mixture: {getattr(model, 'use_head_mixture', 'Not set')}")
            print(f"DEBUG: Model head_allocation: {getattr(model, 'head_allocation', 'Not set')}")

        logprobs_full = _extract_log_probs(out)   # [1, W, V]
        path_probs    = _extract_path_probs(out)  # [1, W, 2] or None
        head_data     = _extract_head_probs(out)  # tensor | dict | None

        # Align to next-token targets
        W = logprobs_full.size(1)
        logprobs_full = logprobs_full[:, :-1, :]  # [1, W-1, V]
        labels        = window[:, 1:]             # [1, W-1]

        # Align aux tensors
        if path_probs is not None:
            if path_probs.size(1) == W:
                path_probs = path_probs[:, :-1, :]
            elif path_probs.size(1) != W - 1:
                print(f"WARNING: path_probs size {tuple(path_probs.shape)} != expected (*, {W-1}, *)")
                path_probs = path_probs[:, :W-1, :] if path_probs.size(1) > W-1 else path_probs

        if head_data is not None:
            if isinstance(head_data, torch.Tensor):
                if head_data.dim() == 2 and head_data.size(1) == W:      # [1,W] idx
                    head_data = head_data[:, :-1]
                elif head_data.dim() == 3 and head_data.size(1) == W:    # [1,W,H] probs
                    head_data = head_data[:, :-1, :]
                elif head_data.dim() == 1 and head_data.size(0) == W:    # [W] idx
                    head_data = head_data[:-1]
            elif isinstance(head_data, dict):
                for pname, td in list(head_data.items()):
                    if td.dim() == 2 and td.size(1) == W:
                        head_data[pname] = td[:, :-1]
                    elif td.dim() == 3 and td.size(1) == W:
                        head_data[pname] = td[:, :-1, :]

        Wm1 = logprobs_full.size(1)
        window_len = end - begin
        eff_stride = min(stride, max(0, window_len - 1))
        target_len = max(0, window_len - eff_stride)

        if target_len > 0 and Wm1 > 0:
            ignore = Wm1 - target_len

            labels_scored = labels.clone()
            if ignore > 0:
                labels_scored[:, :ignore] = -100

            V = logprobs_full.size(-1)
            flat_logp   = logprobs_full.reshape(-1, V)        # [W-1, V]
            flat_labels = labels_scored.reshape(-1)           # [W-1]
            keep_mask   = (flat_labels != -100)               # [W-1]
            K           = int(keep_mask.sum().item())
            if K == 0:
                # nothing to score in this window
                if end == N:
                    break
                begin = end - eff_stride
                continue

            kept_labels = flat_labels[keep_mask]              # [K]
            kept_logp   = flat_logp[keep_mask, :]             # [K, V]

            # ===== PATH SELECTIONS =====
            kept_path_probs = None
            path_selections = None

            path_indices = _extract_path_indices_from_out(out)  # [1, W] or None
            if path_indices is not None:
                if path_indices.size(1) == W:
                    path_indices = path_indices[:, :-1]          # [1, W-1]
                kept_idx = path_indices.reshape(-1)[keep_mask]   # [K]
                for p in kept_idx.cpu().numpy():
                    path_usage_counts[int(p)] += 1
                path_selections = kept_idx                       # [K]
                if path_probs is not None:
                    flat_path_probs = path_probs.reshape(-1, path_probs.size(-1))  # [W-1, 2]
                    kept_path_probs = flat_path_probs[keep_mask, :]                # [K, 2]
                    path_probs_all.append(kept_path_probs.cpu().numpy())
            elif path_probs is not None:
                flat_path_probs = path_probs.reshape(-1, path_probs.size(-1))      # [W-1, 2]
                kept_path_probs = flat_path_probs[keep_mask, :]                    # [K, 2]
                path_selections = kept_path_probs.argmax(dim=-1)                   # [K]
                for p in path_selections.cpu().numpy():
                    path_usage_counts[int(p)] += 1
                path_probs_all.append(kept_path_probs.cpu().numpy())
            else:
                if begin == 0:
                    print("INFO: No path data available.")

            # ===== HEAD SELECTIONS =====
            head_selections = None
            kept_head_data = None

            combined = out.get("head_top_idx_combined", None) if isinstance(out, dict) else None
            if isinstance(combined, torch.Tensor):
                if combined.dim() == 2 and combined.size(1) == W:
                    combined = combined[:, :-1]                        # [1, W-1]
                head_selections = combined.reshape(-1)[keep_mask]       # [K]
            else:
                if isinstance(head_data, dict) and path_selections is not None:
                    # build a full-length selection map aligned with the *kept* tokens only
                    full_path_sel = torch.full((Wm1,), -1, dtype=torch.long, device=window.device)
                    # directly place selections at the positions that are scored
                    full_path_sel[keep_mask] = path_selections.to(window.device)


                    # gather per-path head winners
                    combined_head = torch.zeros(Wm1, dtype=torch.long, device=window.device)
                    for pidx, pname in enumerate(path_names):
                        td = head_data.get(pname, None)
                        if td is None:
                            continue
                        # td: [1, W-1] indices OR [1, W-1, H] probs
                        if td.dim() == 3:
                            # probs -> argmax indices
                            td = td.squeeze(0).argmax(dim=-1, keepdim=True)  # [W-1, 1]
                            td = td.transpose(0, 1)                          # [1, W-1]
                        if td.dim() == 2:
                            td = td.squeeze(0)                               # [W-1]
                            mask_full = (full_path_sel == pidx)
                            if mask_full.any():
                                combined_head[mask_full] = td[mask_full]

                    head_selections = combined_head[keep_mask]  # [K]

                elif isinstance(head_data, torch.Tensor):
                    if head_data.dim() == 2:      # [1, W-1] indices
                        head_selections = head_data.reshape(-1)[keep_mask]
                    elif head_data.dim() == 3:    # [1, W-1, H] probs
                        flat = head_data.reshape(-1, head_data.size(-1))     # [W-1, H]
                        kept_head_data = flat[keep_mask, :]                  # [K, H]
                        head_selections = kept_head_data.argmax(dim=-1)
                        head_probs_all.append(kept_head_data.cpu().numpy())
                else:
                    # fallback: map path→head (0/1) only if we have paths
                    if path_selections is not None:
                        head_selections = path_selections.clone()

            # ===== LM-HEADER TALLIES =====
            if head_selections is not None:
                for h in head_selections.cpu().numpy():
                    global_head_hard_wins[int(h)] += 1
                total_head_tokens += int(head_selections.numel())

            if kept_head_data is not None:
                for row in kept_head_data.cpu().numpy():
                    for hid, p in enumerate(row):
                        global_head_soft_credit[int(hid)] += float(p)

            # Per-path + global
            if path_selections is not None and head_selections is not None:
                for p, h in zip(path_selections.cpu().numpy(), head_selections.cpu().numpy()):
                    head_usage_by_path[int(p)][int(h)] += 1
                    global_head_usage[int(h)] += 1

                # competition snapshots
                sel_np = path_selections.cpu().numpy()
                for i, pname in enumerate(path_names):
                    mask_np = (sel_np == i)
                    if mask_np.any():
                        selected_heads = head_selections[torch.from_numpy(mask_np)].cpu().numpy().tolist()
                        header_competition_stats[pname].extend(selected_heads)
            elif head_selections is not None:
                for h in head_selections.cpu().numpy():
                    head_usage_by_path[-1][int(h)] += 1
                    global_head_usage[int(h)] += 1

            # ===== TOKEN LOGPROBS / ACC =====
            token_logprobs = kept_logp.gather(1, kept_labels.view(-1, 1)).squeeze(1)  # [K]
            preds = kept_logp.argmax(dim=-1)                                          # [K]
            correct = (preds == kept_labels).sum().item()
            total_correct += correct

            # ===== PER-TOKEN RECORD =====
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
                    score_entry["path_name"] = path_names[pidx]
                    if kept_path_probs is not None:
                        score_entry["path_probs"] = kept_path_probs[ptr].cpu().numpy().tolist()

                if head_selections is not None:
                    hidx = int(head_selections[ptr].item())
                    score_entry["selected_head"] = hidx
                    score_entry["head_source"] = "dual_model"

                sequences[0]["scores"].append(score_entry)
                total_scored += 1
                total_logprob += lp
                ptr += 1

        if end == N:
            break
        begin = end - eff_stride

    # ===== Aggregate usage statistics =====
    usage_stats = {"model_type": "dual", "num_paths": 2}

    # Path usage
    if path_usage_counts:
        total_path_selections = sum(path_usage_counts.values())
        path_usage_percentages = {}
        for i in range(2):
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

    # Global header usage
    if global_head_usage:
        total_global_head_selections = sum(global_head_usage.values())
        global_head_percentages = {}
        n_heads_cfg = getattr(model.config, "n_lm_perceptrons", 8)
        if hasattr(model, "lm_head") and hasattr(model.lm_head, "num_perceptrons"):
            n_heads_cfg = int(model.lm_head.num_perceptrons)

        for head_idx in range(n_heads_cfg):
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

    # Head usage by path
    if head_usage_by_path:
        head_usage_stats = {}
        for pidx, head_counts in head_usage_by_path.items():
            total_head_sel = sum(head_counts.values())
            pname = path_names[pidx] if 0 <= pidx < 2 else ("no_path_info" if pidx == -1 else f"path_{pidx}")
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
            sels = header_competition_stats.get(pname, [])
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

    # Probability stats
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

    usage_stats["global_header_hard_wins"]   = {int(h): int(c) for h, c in global_head_hard_wins.items()}
    usage_stats["global_header_soft_credit"] = {int(h): float(c) for h, c in global_head_soft_credit.items()}
    usage_stats["global_header_total_tokens"] = int(total_head_tokens)

    return sequences, total_scored, total_logprob, total_correct, usage_stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Model specifics
    ap.add_argument("--pretrained_model", default="gpt2")
    ap.add_argument("--split_at_layer", type=int, default=6)
    ap.add_argument("--gate_temp", type=float, default=1.0)
    ap.add_argument("--gate_hidden", type=int, default=256)
    ap.add_argument("--n_lm_perceptrons", type=int, default=8)
    ap.add_argument("--use_head_mixture", action="store_true",
                    help="Enable per-path LM-head mixture (requires head_allocation).")
    ap.add_argument("--head_allocation", type=str, default=None,
                    help="Path to JSON mapping path names to head indices.")
    ap.add_argument("--path_selection", default="gate_soft",
                    choices=["gate_soft", "gate_hard", "left_only", "right_only", "max_prob"])
    
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

    # Create dual path model
    model = DualPathMultiHeadersGPT2(
        config=config,
        pretrained_model=tok_name,
        split_at_layer=args.split_at_layer,
        gate_temp=args.gate_temp,
        gate_hidden=args.gate_hidden,
        head_allocation=head_alloc,
        n_lm_perceptrons=args.n_lm_perceptrons,
    ).to(device).eval()
        
    model.use_head_mixture = bool(args.use_head_mixture)

    # Load checkpoint
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
    sequences, num_scored, sum_logprob, num_correct, usage_stats = score_texts_dual_with_stats(
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
            "model": "dual_path_multi_headers",
            "path_selection": args.path_selection,
            "split_at_layer": args.split_at_layer,
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
        "usage_statistics": usage_stats,
        "sequences": sequences
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    
    # Print usage statistics summary to console
    print(f"Wrote {args.out} with {num_scored} scored tokens, {num_correct} correct predictions. "
          f"PPL={out['totals']['perplexity']:.4f}")
    
    print("\n" + "="*60)
    print("DUAL PATH MULTI HEADERS MODEL - USAGE SUMMARY")
    print("="*60)
    
    if "path_usage" in usage_stats:
        print("\nPATH USAGE:")
        path_data = usage_stats["path_usage"]["usage_by_path"]
        for path_name, stats in path_data.items():
            print(f"  {path_name:8}: {stats['percentage']:6.2f}% ({stats['count']:,} tokens)")
    
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
            
            print(f"  {path_name:8}: Head-{most_used} ({most_used_pct:5.1f}%) | Total: {total:,} tokens")
            
            # Show competition stats if available
            comp_stats = path_stats.get("competition_stats", {})
            if comp_stats.get("competing_headers"):
                headers = comp_stats["competing_headers"]
                entropy = comp_stats.get("entropy", 0)
                print(f"    Competition: {len(headers)} headers, entropy={entropy:.2f}")
    
    print("\nGLOBAL HEADER USAGE (ALL PATHS COMBINED):")
    n_heads_cfg = getattr(model.config, "n_lm_perceptrons", 8)
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'num_perceptrons'):
        n_heads_cfg = model.lm_head.num_perceptrons

    wins_dict  = usage_stats.get("global_header_hard_wins", {})
    soft_dict  = usage_stats.get("global_header_soft_credit", {})
    total_hdr  = usage_stats.get("global_header_total_tokens", sum(wins_dict.values()))

    for hid in range(n_heads_cfg):
        wins = int(wins_dict.get(hid, 0))
        pct  = (100.0 * wins / max(1, total_hdr))
        print(f"  head_{hid:<2}: {wins:8d} wins ({pct:6.2f}%)")

    if soft_dict:
        print("\nHEADER SOFT CREDIT (PROBABILITY WEIGHTED):")
        soft_total = max(1.0, float(total_hdr))
        for hid in range(n_heads_cfg):
            credit = float(soft_dict.get(hid, 0.0))
            pct    = 100.0 * credit / soft_total
            print(f"  head_{hid:<2}: {credit:10.2f} credit ({pct:6.2f}%)")

    # Path balance analysis
    if "path_usage" in usage_stats:
        path_data = usage_stats["path_usage"]["usage_by_path"]
        total_tokens = usage_stats["path_usage"]["total_selections"]
        
        left_pct = path_data.get("left", {}).get("percentage", 0)
        right_pct = path_data.get("right", {}).get("percentage", 0)
        
        print("\nPATH BALANCE ANALYSIS:")
        print(f"  Total analyzed tokens: {total_tokens:,}")
        print(f"  Left vs Right balance: {left_pct:.1f}% vs {right_pct:.1f}%")
        
        balance_score = 100 - abs(left_pct - right_pct)
        print(f"  Balance score: {balance_score:.1f}/100 (100 = perfectly balanced)")
        
        if left_pct > 75:
            print("  → Model heavily favors LEFT path")
        elif right_pct > 75:
            print("  → Model heavily favors RIGHT path")
        elif abs(left_pct - right_pct) < 10:
            print("  → Model shows good path balance")
        else:
            dominant = "LEFT" if left_pct > right_pct else "RIGHT"
            print(f"  → Model moderately favors {dominant} path")

    # Header specialization analysis
    if "head_usage_by_path" in usage_stats:
        print("\nHEADER SPECIALIZATION ANALYSIS:")
        left_stats = usage_stats["head_usage_by_path"].get("left", {})
        right_stats = usage_stats["head_usage_by_path"].get("right", {})
        
        if left_stats.get("most_used_head") is not None and right_stats.get("most_used_head") is not None:
            left_head = left_stats["most_used_head"]
            right_head = right_stats["most_used_head"]
            
            if left_head == right_head:
                print(f"  → Both paths prefer head_{left_head} (low specialization)")
            else:
                print(f"  → Left path prefers head_{left_head}, Right path prefers head_{right_head}")
                print("  → Good path-header specialization")

    print("="*60)


if __name__ == "__main__":
    main()