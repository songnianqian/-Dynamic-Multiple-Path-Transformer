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
ppl_multi_headers.py  (gated multi-headers version)

- Reads evaluator inputs.json (texts + meta with tokenizer/max_length/stride)
- Instantiates CustomMultiHeaderGPT2Model from multi_headers_model.py
- Loads `model_state_dict` from a training checkpoint
- Computes gold next-token logprobs with HF stride method (concat mode)
- Computes accuracy (top-1) and writes model_outputs.json

Usage:
  python ppl_multi_headers.py \
    --inputs /path/inputs.json \
    --checkpoint /path/ckpt.pt \
    --out /path/model_outputs.json
"""
import os, json, argparse
import torch
import torch.nn.functional as F
from transformers import GPT2Config, AutoTokenizer
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# ✅ Use the gated multi-headers model
from multi_headers.multi_headers_model import CustomMultiHeaderGPT2Model

def _extract_logits(output):
    """Works with HF dict-style or object-with-.logits."""
    if isinstance(output, dict):
        return output["logits"]
    return getattr(output, "logits")

def _ensure_log_probs(t):
    """
    Your model already returns log-probs over vocab.
    If a different model returns raw logits, convert to log-probs.
    Heuristic: logsumexp over vocab ≈ 0.0 for log-probs.
    """
    # sample a small slice that's definitely present
    sample = t[..., :1, :].squeeze(-2) if t.dim() >= 3 else t
    # When using [B,S,V], take [0,0,:]; when [B,V], take [0,:]
    if t.dim() >= 3:
        sample = t[0, 0]
    else:
        sample = t[0]
    is_logprob = torch.allclose(torch.logsumexp(sample, dim=-1),
                                torch.tensor(0.0, device=sample.device),
                                atol=1e-3)
    return t if is_logprob else torch.log_softmax(t, dim=-1)

@torch.no_grad()
def score_texts_hf_stride_concat(model, tokenizer, texts, device, *, max_length=1024, stride=512):
    """
    Concatenate all texts with an EOS between them and score with HF stride.
    Produces a single 'sequence' entry with global token positions.
    """
    sequences = []
    total_scored = 0
    total_logprob = 0.0
    total_correct = 0

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

        out = model(input_ids=window, attention_mask=attn)    # logits are LOG-PROBS in your model
        logits_full = _extract_logits(out)                    # [1, W, V] (log-probs or raw)
        logits_full = logits_full[:, :-1, :]                  # [1, W-1, V] predicts next token
        labels      = window[:, 1:]                           # [1, W-1]

        Wm1 = logits_full.size(1)
        window_len = end - begin
        # Avoid zero scoring when window is short
        eff_stride = min(stride, max(0, window_len - 1))
        target_len = max(0, window_len - eff_stride)

        if target_len > 0 and Wm1 > 0:
            ignore = Wm1 - target_len
            if ignore > 0:
                labels_scored = labels.clone()
                labels_scored[:, :ignore] = -100
            else:
                labels_scored = labels

            # Convert to log-probs if needed (your model already is)
            logp = _ensure_log_probs(logits_full)             # [1, W-1, V]
            V = logp.size(-1)

            flat_logp   = logp.reshape(-1, V)                 # [W-1, V]
            flat_labels = labels_scored.reshape(-1)           # [W-1]
            keep_mask   = (flat_labels != -100)
            kept_labels = flat_labels[keep_mask]              # [K]
            kept_logp   = flat_logp[keep_mask, :]             # [K, V]

            # Gold-token log-probs
            token_logprobs = kept_logp.gather(1, kept_labels.view(-1, 1)).squeeze(1)  # [K]

            # Top-1 predictions (accuracy)
            kept_logits_for_pred = logits_full.reshape(-1, V)[keep_mask, :]           # [K, V]
            preds = kept_logits_for_pred.argmax(dim=-1)                               # [K]
            correct = (preds == kept_labels).sum().item()
            total_correct += correct

            # Map back to global positions
            ptr = 0
            for k in range(Wm1):
                if k < ignore:
                    continue
                global_pos = begin + 1 + k
                tok_id     = int(window[0, k+1].item())
                lp         = float(token_logprobs[ptr].item())
                pred_id    = int(preds[ptr].item())
                is_correct = (pred_id == tok_id)

                sequences[0]["scores"].append({
                    "pos": int(global_pos),
                    "token_id": tok_id,
                    "logprob": lp,
                    "predicted_id": pred_id,
                    "correct": is_correct
                })
                total_scored += 1
                total_logprob += lp
                ptr += 1

        if end == N:
            break
        begin = end - eff_stride

    return sequences, total_scored, total_logprob, total_correct

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.inputs, "r", encoding="utf-8") as f:
        inp = json.load(f)
    meta   = inp.get("meta", {})
    texts  = inp.get("texts", [])
    if not texts:
        raise SystemExit("No texts found in inputs.json")

    tok_name   = meta.get("tokenizer", "gpt2")
    max_length = int(meta.get("max_length", 1024))
    stride     = int(meta.get("stride", 512))
    n_heads    = int(meta.get("n_lm_perceptrons", 4))

    device    = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build config consistent with training
    config = GPT2Config.from_pretrained(tok_name)
    config.n_positions = max_length
    config.n_lm_perceptrons = n_heads
    config.force_identical_output = False  # keep gated behavior

    # Instantiate gated model
    model = CustomMultiHeaderGPT2Model(config, use_pretrained=True, pretrained_model=tok_name)
    model.to(device).eval()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint. missing={len(missing)}, unexpected={len(unexpected)}")

    # Score
    sequences, num_scored, sum_logprob, num_correct = score_texts_hf_stride_concat(
        model, tokenizer, texts, device, max_length=max_length, stride=stride
    )

    out = {
        "meta": {
            "tokenizer": tok_name,
            "max_length": max_length,
            "stride": stride,
            "n_lm_perceptrons": n_heads,
            "log_base": "e"
        },
        "totals": {
            "num_scored_tokens": int(num_scored),
            "sum_logprob": float(sum_logprob),
            "num_correct_predictions": int(num_correct)
        },
        "sequences": sequences
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out} with {num_scored} scored tokens, {num_correct} correct predictions.")

if __name__ == "__main__":
    main()
