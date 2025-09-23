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
ppl_dual_path.py  (dual-path GPT-2 with gated mixture + usage statistics)

Enhanced version that tracks:
- Dual path usage percentages 
- Per-path statistics and competition
- Gate entropy and balance metrics

- Reads evaluator inputs.json (texts + meta with tokenizer/max_length/stride)
- Instantiates IndependentDualPathGPT2 from dual_path_model.py
- Loads `model_state_dict` from a training checkpoint
- Computes gold next-token logprobs with HF stride (concat mode)
- Computes accuracy (top-1) and writes model_outputs.json with usage stats

Usage:
  python ppl_dual_path.py \
    --inputs /path/inputs.json \
    --checkpoint /path/ckpt.pt \
    --out /path/model_outputs.json \
    --pretrained_model gpt2 \
    --split_at_layer 6 \
    --path_selection gate_soft
"""
import os, json, argparse, math
import torch
import torch.nn.functional as F
from transformers import GPT2Config, AutoTokenizer
from collections import defaultdict, Counter
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


from torch.serialization import add_safe_globals
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
add_safe_globals([GPT2Config])

# ✅ Use your dual-path model
from dual_path_multi_headers.dual_path_model import IndependentDualPathGPT2

def _extract_logits(output):
    """Works with dict-style model outputs."""
    if isinstance(output, dict):
        return output["logits"]
    return getattr(output, "logits")

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
    is_logprob = torch.allclose(torch.logsumexp(sample, dim=-1),
                                torch.tensor(0.0, device=sample.device),
                                atol=1e-3)
    return t if is_logprob else torch.log_softmax(t, dim=-1)

def _extract_path_probs(output):
    """Extract path probabilities from dual-path model output if available."""
    if isinstance(output, dict) and "gate" in output:
        gate = output["gate"]
        if isinstance(gate, dict):
            # Check for gate_soft or gate_hard (soft/hard gating weights)
            if "gate_soft" in gate:
                return gate["gate_soft"]  # [B, S, 2]
            elif "gate_hard" in gate:
                return gate["gate_hard"]  # [B, S, 2]
            # Check for gate_logits (convert to probabilities)
            elif "gate_logits" in gate:
                gate_logits = gate["gate_logits"]  # [B, S, 2]
                return torch.softmax(gate_logits, dim=-1)
        elif torch.is_tensor(gate):
            return gate  # [B, S, 2]
    elif isinstance(output, dict) and "gate_weights" in output:
        return output["gate_weights"]  # [B, S, 2] for dual paths
    return None

def _get_path_selection_from_probs(path_probs):
    """
    Convert path probabilities to path selections.
    For dual-path: [left, right]
    """
    if path_probs is None:
        return None
    
    # path_probs shape: [B, S, 2] for dual paths
    # Return the path index with highest probability (0=left, 1=right)
    return path_probs.argmax(dim=-1)  # [B, S]

def _extract_gate_stats(output):
    """Extract gate statistics from dual-path model output."""
    gate_stats = {}
    
    if isinstance(output, dict) and "gate" in output:
        gate = output["gate"]
        if isinstance(gate, dict):
            # Look for gate probabilities in various formats
            gate_probs = None
            
            if "gate_soft" in gate:
                gate_probs = gate["gate_soft"]  # [B, S, 2]
            elif "gate_hard" in gate:
                gate_probs = gate["gate_hard"]  # [B, S, 2]
            elif "gate_logits" in gate:
                gate_logits = gate["gate_logits"]  # [B, S, 2]
                gate_probs = torch.softmax(gate_logits, dim=-1)
            
            if gate_probs is not None and gate_probs.dim() == 3 and gate_probs.size(-1) == 2:
                # Usage percentages
                usage = gate_probs.mean(dim=(0, 1))  # [2]
                gate_stats["left_pct"] = float(usage[0] * 100)
                gate_stats["right_pct"] = float(usage[1] * 100)
                
                # Entropy (measure of uncertainty/balance)
                entropy = -(gate_probs * (gate_probs.clamp_min(1e-8)).log()).sum(dim=-1).mean()
                gate_stats["entropy"] = float(entropy)
                
                # Balance score (how close to 50/50)
                balance = 100 - abs(usage[0] - usage[1]) * 100
                gate_stats["balance"] = float(balance)
        elif torch.is_tensor(gate) and gate.dim() == 3 and gate.size(-1) == 2:
            # Direct tensor gate weights
            usage = gate.mean(dim=(0, 1))  # [2]
            gate_stats["left_pct"] = float(usage[0] * 100)
            gate_stats["right_pct"] = float(usage[1] * 100)
            
            entropy = -(gate * (gate.clamp_min(1e-8)).log()).sum(dim=-1).mean()
            gate_stats["entropy"] = float(entropy)
            
            balance = 100 - abs(usage[0] - usage[1]) * 100
            gate_stats["balance"] = float(balance)
    
    return gate_stats

@torch.no_grad()
def score_texts_hf_stride_concat_with_stats(model, tokenizer, texts, device, *,
                                           max_length=1024, stride=512, 
                                           path_selection="gate_soft"):
    """
    Enhanced scoring function that tracks path usage statistics for dual-path model.
    """
    sequences = []
    total_scored = 0
    total_logprob = 0.0
    total_correct = 0

    # Statistics tracking
    path_usage_counts = Counter()                           # Track which paths are used (0=left, 1=right)
    path_probs_all = []                                    # Store path probabilities for analysis
    gate_stats_accumulated = defaultdict(list)             # Accumulate gate statistics
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
        window = token_ids[begin:end].unsqueeze(0)            # [1, W]
        attn   = torch.ones_like(window, device=device)       # [1, W]

        # IMPORTANT: use the trained policy (gate_soft by default)
        out = model(input_ids=window, attention_mask=attn, labels=None,
                    path_selection=path_selection)
        
        # DEBUG (first window only)
        if begin == 0:
            print(f"DEBUG: Model output type: {type(out)}")
            if isinstance(out, dict):
                print(f"DEBUG: Model output keys: {list(out.keys())}")
                if "gate" in out:
                    gate = out["gate"]
                    print(f"DEBUG: Gate type: {type(gate)}")
                    if isinstance(gate, dict):
                        print(f"DEBUG: Gate keys: {list(gate.keys())}")
                        for k, v in gate.items():
                            if torch.is_tensor(v):
                                print(f"DEBUG: Gate[{k}] shape: {v.shape}")
                            else:
                                print(f"DEBUG: Gate[{k}] type: {type(v)}")
                    elif torch.is_tensor(gate):
                        print(f"DEBUG: Gate tensor shape: {gate.shape}")
        
        logits_full = _extract_logits(out)                    # [1, W, V] (logits or log-probs)
        path_probs  = _extract_path_probs(out)                # [1, W, 2] or None
        gate_stats  = _extract_gate_stats(out)                # dict of gate statistics
        
        # Additional debug for first window
        if begin == 0:
            print(f"DEBUG: path_probs is None: {path_probs is None}")
            if path_probs is not None:
                print(f"DEBUG: path_probs shape: {path_probs.shape}")
                print(f"DEBUG: path_probs sample: {path_probs[0, 0, :].tolist()}")
            print(f"DEBUG: gate_stats: {gate_stats}")
        
        # Accumulate gate statistics
        for key, value in gate_stats.items():
            gate_stats_accumulated[key].append(value)
        
        logits_full = logits_full[:, :-1, :]                  # [1, W-1, V] predicts next token
        labels      = window[:, 1:]                           # [1, W-1]

        # Align path probabilities to [1, W-1, ...]
        if path_probs is not None:
            path_probs = path_probs[:, :-1, :]

        Wm1 = logits_full.size(1)
        window_len = end - begin
        eff_stride = min(stride, max(0, window_len - 1))
        target_len = max(0, window_len - eff_stride)

        if target_len > 0 and Wm1 > 0:
            ignore = Wm1 - target_len
            labels_scored = labels.clone()
            if ignore > 0:
                labels_scored[:, :ignore] = -100

            logp = _ensure_log_probs(logits_full)             # [1, W-1, V]
            V = logp.size(-1)

            flat_logp   = logp.reshape(-1, V)                 # [W-1, V]
            flat_labels = labels_scored.reshape(-1)           # [W-1]
            keep_mask   = (flat_labels != -100)
            kept_labels = flat_labels[keep_mask]              # [K]
            kept_logp   = flat_logp[keep_mask, :]             # [K, V]

            # ----- PATH SELECTIONS -----
            if path_probs is not None:
                flat_path_probs = path_probs.reshape(-1, path_probs.size(-1))  # [W-1, 2]
                kept_path_probs = flat_path_probs[keep_mask, :]                # [K, 2]
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

            # Top-1 predictions (accuracy) – use same distribution as scoring
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
        for i in range(2):  # Dual paths: 0=left, 1=right
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
    # dual-path specifics
    ap.add_argument("--pretrained_model", default="gpt2")
    ap.add_argument("--split_at_layer", type=int, default=6)
    ap.add_argument("--path_selection", default="gate_soft",
                    choices=["gate_soft","gate_hard","left_only","right_only","max_prob","soft_weighted"])
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
    # Optional: be explicit about attention backend (your model also enforces this)
    if not hasattr(config, "_attn_implementation") or config._attn_implementation is None:
        config._attn_implementation = "eager"

    # Instantiate dual-path model (loads HF weights internally)
    model = IndependentDualPathGPT2(
        config=config,
        pretrained_model=tok_name,
        split_at_layer=args.split_at_layer
    ).to(device).eval()

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
    missing, unexpected = model.load_state_dict(state, strict=False)
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
            "model": "dual_path",
            "path_selection": args.path_selection,
            "split_at_layer": args.split_at_layer
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
        
        # Gate (left vs right split)
        if "left_pct" in gate_stats and "right_pct" in gate_stats:
            left_pct = gate_stats["left_pct"]["mean"]
            right_pct = gate_stats["right_pct"]["mean"]
            entropy = gate_stats.get("entropy", {}).get("mean", 0)
            balance = gate_stats.get("balance", {}).get("mean", 0)
            print(f"  Gate (Left/Right): {left_pct:.1f}% / {right_pct:.1f}% | Entropy: {entropy:.3f} | Balance: {balance:.1f}")
    
    if "path_probability_stats" in usage_stats:
        print("\nPATH PROBABILITY ANALYSIS:")
        prob_stats = usage_stats["path_probability_stats"]
        mean_probs = prob_stats["mean_probs"]
        entropy = prob_stats["entropy_mean"]
        
        # Define path names for dual paths
        path_names_local = ["left", "right"]
        
        print(f"  Average path probabilities:")
        for i, (path_name, prob) in enumerate(zip(path_names_local, mean_probs)):
            print(f"    {path_name:12}: {prob:.3f}")
        print(f"  Mean path entropy: {entropy:.3f}")
    
    # Overall summary
    if "path_usage" in usage_stats:
        path_data = usage_stats["path_usage"]["usage_by_path"]
        total_tokens = usage_stats["path_usage"]["total_selections"]
        num_active_paths = len([p for p in path_data.values() if p['count'] > 0])

        print("\nOVERALL:")
        print(f"  Total analyzed tokens: {total_tokens:,}")
        print(f"  Active paths: {num_active_paths}/2")

        path_percentages = [stats['percentage'] for stats in path_data.values()]
        if path_percentages:
            balance_score = 100 - abs(path_percentages[0] - path_percentages[1])
            print(f"  Path balance score: {balance_score:.1f}/100 (higher = more balanced)")

    print("="*60)

if __name__ == "__main__":
    main()
