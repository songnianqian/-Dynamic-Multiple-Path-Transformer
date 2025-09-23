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
ppl_evaluator.py

Reads model_outputs.json (from ppl_model_runner_custom.py) and reports perplexity and accuracy.
This script is model-agnostic; it trusts only the output JSON with per-token logprobs.

Usage:
  python ppl_evaluator.py --pred model_outputs.json
"""
import json, math, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Path to model_outputs.json produced by the model runner")
    args = ap.parse_args()

    with open(args.pred, "r", encoding="utf-8") as f:
        data = json.load(f)

    totals = data.get("totals", {})
    num_scored = int(totals.get("num_scored_tokens", 0))
    sum_logprob = float(totals.get("sum_logprob", 0.0))
    num_correct = int(totals.get("num_correct_predictions", 0))

    if num_scored <= 0:
        print("No scored tokens found; cannot compute perplexity or accuracy.")
        return

    # Calculate perplexity
    ppl = math.exp(-sum_logprob / num_scored)
    
    # Calculate accuracy
    accuracy = (num_correct / num_scored) * 100.0

    print(f"Perplexity: {ppl:.6f}")
    print(f"Accuracy: {accuracy:.2f}% ({num_correct}/{num_scored})")
    print(f"(Sum logprob: {sum_logprob:.6f})")

if __name__ == "__main__":
    main()