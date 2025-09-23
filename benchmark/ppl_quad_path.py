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
ppl_quad_path.py  (quad-path GPT-2 with hierarchical gated mixture + usage statistics)

Enhanced version that tracks:
- Quad path usage percentages 
- Per-path statistics and competition
- Gate entropy and balance metrics

- Reads evaluator inputs.json (texts + meta with tokenizer/max_length/stride)
- Instantiates HierarchicalQuadPathGPT2 from quad_path_model.py
- Loads `model_state_dict` from a training checkpoint
- Computes gold next-token logprobs with HF stride (concat mode)
- Computes accuracy (top-1) and writes model_outputs.json with usage stats

Usage:
  python ppl_quad_path.py \
    --inputs /path/inputs.json \
    --checkpoint /path/ckpt.pt \
    --out /path/model_outputs.json \
    --pretrained_model gpt2 \
    --split_at_layer_1 6 --split_at_layer_2 9 \
    --path_selection hierarchical_gate
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

# ✅ Quad-path model
from quad_path_multi_headers.quad_path_model import HierarchicalQuadPathGPT2


def _extract_logits_or_logprobs(output):
    """
    For quad model, outputs may contain 'log_probs' (preferred) or 'logits'.
    Return a tensor [B,S,V] that is *log-probs*.
    """
    if isinstance(output, dict):
        if output.get("log_probs") is not None:
            t = output["log_probs"]
            # Should already be log-probs; quick sanity
            if t.dim() != 3:
                raise ValueError(f"log_probs has wrong shape {tuple(t.shape)}")
            return t
        t = output.get("logits")
        if t is None:
            raise ValueError("Model output missing 'log_probs' and 'logits'.")
        return _ensure_log_probs(t)
    # Fallback: HF-style object with .logits
    return _ensure_log_probs(getattr(output, "logits"))


def _ensure_log_probs(t):
    """
    Convert raw logits -> log-probs iff needed.
    Heuristic: logsumexp ≈ 0 over vocab for log-probs.
    """
    if t.dim() == 3:  # [B,S,V]
        sample = t[0, 0]
    elif t.dim() == 2:  # [B,V] or [S,V]
        sample = t[0]
    else:
        raise ValueError(f"Unsupported logits shape {tuple(t.shape)}")
    is_logprob = torch.allclose(
        torch.logsumexp(sample, dim=-1),
        torch.tensor(0.0, device=sample.device),
        atol=1e-3
    )
    return t if is_logprob else torch.log_softmax(t, dim=-1)


def _extract_path_probs(output):
    """Extract path probabilities from model output if available."""
    if isinstance(output, dict) and "gate" in output:
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
    return None


def _get_path_selection_from_probs(path_probs):
    """
    Convert path probabilities to path selections.
    Assumes hierarchical structure: [left_left, left_right, right_left, right_right]
    """
    if path_probs is None:
        return None
    
    # path_probs shape: [B, S, 4] for quad paths
    # Return the path index with highest probability
    return path_probs.argmax(dim=-1)  # [B, S]


def _extract_gate_stats(output):
    """Extract gate statistics from model output."""
    gate_stats = {}
    
    if isinstance(output, dict) and "gate" in output:
        gate = output["gate"]
        if isinstance(gate, dict):
            # Extract gate probabilities and compute statistics
            for gate_name in ["gate1", "gate2_left", "gate2_right"]:
                if gate_name in gate:
                    gate_probs = gate[gate_name]  # [B, S, 2]
                    
                    # Usage percentages
                    usage = gate_probs.mean(dim=(0, 1))  # [2]
                    gate_stats[f"{gate_name}_left_pct"] = float(usage[0] * 100)
                    gate_stats[f"{gate_name}_right_pct"] = float(usage[1] * 100)
                    
                    # Entropy (measure of uncertainty/balance)
                    entropy = -(gate_probs * (gate_probs.clamp_min(1e-8)).log()).sum(dim=-1).mean()
                    gate_stats[f"{gate_name}_entropy"] = float(entropy)
                    
                    # Balance score (how close to 50/50)
                    balance = 100 - abs(usage[0] - usage[1]) * 100
                    gate_stats[f"{gate_name}_balance"] = float(balance)
    
    return gate_stats


@torch.no_grad()
def score_texts_hf_stride_concat_with_stats(model, tokenizer, texts, device, *,
                                            max_length=1024, stride=512, 
                                            path_selection="hierarchical_gate"):
    """
    Enhanced scoring function that tracks path usage statistics for quad-path model.
    """
    sequences = []
    total_scored = 0
    total_logprob = 0.0
    total_correct = 0

    # Statistics tracking
    path_usage_counts = Counter()                           # Track which paths are used (0..3)
    path_probs_all = []                                    # Store path probabilities for analysis
    gate_stats_accumulated = defaultdict(list)             # Accumulate gate statistics
    path_names = ["left_left", "left_right", "right_left", "right_right"]

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
        window = token_ids[begin:end].unsqueeze(0)            # [1, W]
        attn   = torch.ones_like(window, device=device)       # [1, W]

        # Forward pass with path tracking
        out = model(input_ids=window, attention_mask=attn, labels=None,
                    path_selection=path_selection)
        
        # DEBUG (first window only)
        if begin == 0:
            print(f"DEBUG: Model output type: {type(out)}")
            if isinstance(out, dict):
                print(f"DEBUG: Model output keys: {list(out.keys())}")
        
        logprobs_full = _extract_logits_or_logprobs(out)      # [1, W, V] (log-probs)
        path_probs    = _extract_path_probs(out)              # [1, W, 4] or None
        gate_stats    = _extract_gate_stats(out)              # dict of gate statistics
        
        # Accumulate gate statistics
        for key, value in gate_stats.items():
            gate_stats_accumulated[key].append(value)
        
        logprobs_full = logprobs_full[:, :-1, :]              # [1, W-1, V] predicts next token
        labels        = window[:, 1:]                          # [1, W-1]

        # Align path probabilities to [1, W-1, ...]
        if path_probs is not None:
            path_probs = path_probs[:, :-1, :]

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
            keep_mask   = (flat_labels != -100)
            kept_labels = flat_labels[keep_mask]              # [K]
            kept_logp   = flat_logp[keep_mask, :]             # [K, V]

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
                # Store for analysis
                path_probs_all.append(kept_path_probs.cpu().numpy())
            else:
                path_selections = None

            # Gold-token log-probs
            token_logprobs = kept_logp.gather(1, kept_labels.view(-1, 1)).squeeze(1)  # [K]

            # Top-1 predictions (accuracy) — use same distribution as scoring
            preds = kept_logp.argmax(dim=-1)                                          # [K]
            correct = (preds == kept_labels).sum().item()
            total_correct += correct

            # Map back to global positions and record detailed scores
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
                
                # Add path information if available
                if path_selections is not None:
                    pidx = int(path_selections[ptr].item())
                    score_entry["selected_path"] = pidx
                    score_entry["path_probs"] = kept_path_probs[ptr].cpu().numpy().tolist()

                sequences[0]["scores"].append(score_entry)
                total_scored += 1
                total_logprob += lp
                ptr += 1

        if end == N:
            break
        begin = end - eff_stride

    # ===== Aggregate usage statistics =====
    usage_stats = {}

    # Path usage statistics
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

    # Path probability statistics
    if path_probs_all:
        all_path_probs = np.concatenate(path_probs_all, axis=0)
        usage_stats["path_probability_stats"] = {
            "mean_probs": np.mean(all_path_probs, axis=0).tolist(),
            "std_probs": np.std(all_path_probs, axis=0).tolist(),
            "entropy_mean": float(np.mean(-np.sum(all_path_probs * np.log(all_path_probs + 1e-10), axis=1)))
        }

    # Gate statistics (averaged across windows)
    if gate_stats_accumulated:
        gate_stats_final = {}
        for key, values in gate_stats_accumulated.items():
            gate_stats_final[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
        usage_stats["gate_statistics"] = gate_stats_final

    return sequences, total_scored, total_logprob, total_correct, usage_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # quad-path specifics
    ap.add_argument("--pretrained_model", default="gpt2")
    ap.add_argument("--split_at_layer_1", type=int, default=6)
    ap.add_argument("--split_at_layer_2", type=int, default=9)
    ap.add_argument("--path_selection", default="hierarchical_gate",
                    choices=["hierarchical_gate", "gate_hard", "left_left_only", "max_prob"])
    ap.add_argument("--gate_temp", type=float, default=1.0)
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
    # Optional: be explicit about attention backend (matches your model)
    if not hasattr(config, "_attn_implementation") or config._attn_implementation is None:
        config._attn_implementation = "eager"

    # Instantiate quad-path model (loads HF weights internally)
    model = HierarchicalQuadPathGPT2(
        config=config,
        pretrained_model=tok_name,
        split_at_layer_1=args.split_at_layer_1,
        split_at_layer_2=args.split_at_layer_2,
        gate_temp=args.gate_temp
    ).to(device).eval()

    # Load checkpoint (quad training ckpt)
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
    # PyTorch returns IncompatibleKeys; fall back if older version returns tuple
    try:
        missing, unexpected = incompat.missing_keys, incompat.unexpected_keys
    except AttributeError:
        missing, unexpected = incompat  # type: ignore
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
            "model": "quad_path",
            "path_selection": args.path_selection,
            "split_at_layer_1": args.split_at_layer_1,
            "split_at_layer_2": args.split_at_layer_2,
        },
        "totals": {
            "num_scored_tokens": int(num_scored),
            "sum_logprob": float(sum_logprob),
            "num_correct_predictions": int(num_correct),
            "perplexity": math.exp(-sum_logprob / max(1, num_scored))
        },
        "usage_statistics": usage_stats,  # New section with path usage stats
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
    
    if "gate_statistics" in usage_stats:
        print("\nGATE STATISTICS:")
        gate_stats = usage_stats["gate_statistics"]
        
        # Gate 1 (main split: left vs right)
        if "gate1_left_pct" in gate_stats and "gate1_right_pct" in gate_stats:
            left_pct = gate_stats["gate1_left_pct"]["mean"]
            right_pct = gate_stats["gate1_right_pct"]["mean"]
            entropy = gate_stats.get("gate1_entropy", {}).get("mean", 0)
            print(f"  Gate 1 (Left/Right): {left_pct:.1f}% / {right_pct:.1f}% | Entropy: {entropy:.3f}")
        
        # Gate 2 Left (left branch split)
        if "gate2_left_left_pct" in gate_stats and "gate2_left_right_pct" in gate_stats:
            ll_pct = gate_stats["gate2_left_left_pct"]["mean"]
            lr_pct = gate_stats["gate2_left_right_pct"]["mean"]
            entropy = gate_stats.get("gate2_left_entropy", {}).get("mean", 0)
            print(f"  Gate 2L (LL/LR):     {ll_pct:.1f}% / {lr_pct:.1f}% | Entropy: {entropy:.3f}")
        
        # Gate 2 Right (right branch split)
        if "gate2_right_left_pct" in gate_stats and "gate2_right_right_pct" in gate_stats:
            rl_pct = gate_stats["gate2_right_left_pct"]["mean"]
            rr_pct = gate_stats["gate2_right_right_pct"]["mean"]
            entropy = gate_stats.get("gate2_right_entropy", {}).get("mean", 0)
            print(f"  Gate 2R (RL/RR):     {rl_pct:.1f}% / {rr_pct:.1f}% | Entropy: {entropy:.3f}")
    
    if "path_probability_stats" in usage_stats:
        print("\nPATH PROBABILITY ANALYSIS:")
        prob_stats = usage_stats["path_probability_stats"]
        mean_probs = prob_stats["mean_probs"]
        entropy = prob_stats["entropy_mean"]
        
        # Define path names for this scope
        path_names = ["left_left", "left_right", "right_left", "right_right"]
        
        print(f"  Average path probabilities:")
        for i, (path_name, prob) in enumerate(zip(path_names, mean_probs)):
            print(f"    {path_name:12}: {prob:.3f}")
        print(f"  Mean path entropy: {entropy:.3f}")
    
    # Overall summary
    if "path_usage" in usage_stats:
        path_data = usage_stats["path_usage"]["usage_by_path"]
        total_tokens = usage_stats["path_usage"]["total_selections"]
        num_active_paths = len([p for p in path_data.values() if p['count'] > 0])

        print("\nOVERALL:")
        print(f"  Total analyzed tokens: {total_tokens:,}")
        print(f"  Active paths: {num_active_paths}/4")

        path_percentages = [stats['percentage'] for stats in path_data.values()]
        if path_percentages:
            balance_score = 100 - (max(path_percentages) - min(path_percentages))
            print(f"  Path balance score: {balance_score:.1f}/100 (higher = more balanced)")

    print("="*60)


if __name__ == "__main__":
    main()
