# Dynamic Multiple Path Transformer (DMPT)

A GPT-2–compatible, **multi-path** transformer that adds capacity with **~10% better perplexity and accuracy** over the HuggingFace GPT-2 baseline **at ~the same inference speed**. DMPT achieves this by **hard path splits** (dual/quad) and **multiple LM headers** with **fast-k routing**.

---

## 🔎 Introduction

The standard Transformer architecture is built from a **serial stack of attention + MLP blocks**, each layer refining the representation before passing it to the next.  
While powerful, this design lacks **parallelism** in representation flow — every token must follow the same path.  

The **Dynamic Multiple Path Transformer (DMPT)** introduces *multiple expert paths* inside the backbone. By splitting the model at specific layers (e.g., 6 or 9), DMPT enables **parallel representation learning**, where different paths can specialize in different aspects of language.

Another key observation is that **next-word prediction depends heavily on the distribution of output vectors**.  
Clusters and neighborhood structures of these vectors make the transformer effective. With **multiple LM headers**, DMPT can support *multiple clusters* and *diverse neighborhood distributions*, enabling the model to handle **different domain knowledge** simultaneously.

Together, multi-path routing and multi-header outputs give DMPT:
- **Extra capacity** without heavy speed penalties  
- **Domain specialization** across paths/headers  
- **Better perplexity and accuracy** (+10% over GPT-2 baseline)  

---

## 🔥 Results (WikiText-103, stride-concat eval)

| Model               | Perplexity ↓ | Accuracy ↑ | Notes |
|---------------------|--------------|------------|-------|
| HF GPT-2 Baseline   | **38.306**   | **35.04%** | reference |
| **DMPT (Dual Path)**| **30.442**   | **39.47%** | +~10% on both metrics |

- Tokens evaluated: 280,232  
- Baseline sum logprob: −1,021,618.53  
- DMPT sum logprob: −957,220.16

> You should see consistent, monotonic perplexity improvement during training with the provided schedules.

---

## ✨ Key Ideas

- **Multi-Path architecture**: split the backbone at selected layers (e.g., 6 for *dual*, 9 for *quad*) and route tokens **hard** to a single path.
- **Multiple LM headers** per path with **fast-k**: score one header fully to get top-K tokens, then compute remaining headers only on those candidates.
- **Stable training** thanks to: load-balance loss, gold-aux routing signal, and a small **tether** KL to a frozen baseline head.
- **Scalable**: add more splits / heads to grow capacity, or specialize paths for domains.

---

## 📦 Repository Layout

```
.
├─ src                             # Python source
  ├─ multi_headers                 # multi_headers model and training scripts
  ├─ dual_path_multi_headers        # dual path multi_headers model and training scripts
  ├─ quad_path_multi_headers        # quad path multi_headers model and training scripts
  ├─ utils                         # benchmark scripts
├─ benchmark                       # benchmark scripts
├─ docs                            # commands and documents
└─ README.md
└─ LICENSE
```

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/DMPT.git
cd DMPT
# Create/activate a venv if you like
pip install -r requirements.txt  # ensure torch + transformers are installed
```

**Requirements (tested):**
- Python 3.10+
- PyTorch 2.x (CUDA recommended)
- transformers 4.40+
- tqdm, numpy

---

## 🚀 Quick Start

### 1) Train — Dual Path (split at layer 6)
```bash
python training.py   --pretrained_model gpt2   --split_at_layer 6   --epochs 1   --batch_size 1   --grad_accum_steps 12   --max_length 256   --max_steps 8200   --train_path_selection gate_soft   --eval_path_selection gate_hard   --checkpoint_dir checkpoints/dual_path   --head_allocation head_allocation.json   --lb_coef 0.02   --gold_aux_coef 0.01   --head_lr 3e-6   --gate_lr 2e-6   --use_head_mixture   --head_gate_temp 0.9
```

### 2) (Optional) Resume from a checkpoint
```bash
python training.py   --resume checkpoints/dual_path/checkpoint_epoch_0_step_120000.pt   --eval_every 500 --save_every 2050
```

### 3) Evaluate perplexity & accuracy
```bash
# Uses a safe eval that computes logprobs, ppl, and token accuracy
python -c "from training import eval_ppl_safe; print('Use the provided scripts in your pipeline')"  # placeholder
```

> Tip: You can also integrate your own stride-concat evaluator; the model returns either logits or log-probs depending on flags.

### 4) Quick greedy preview
```bash
python -c "from training import quick_preview_greedy; print('See training.py for quick_preview_greedy() usage')"  # placeholder
```

---

## ⚙️ Configuration

### LM-Head Allocation
Map heads to paths. Example (`head_allocation.json`):
```json
{
  "left":  [0, 1, 2, 3],
  "right": [4, 5, 6, 7]
}
```

### Common Flags
- `--split_at_layer {6|9}`: dual vs quad (by adding another split in your model variant)
- `--train_path_selection {gate_soft|gate_hard|left_only|right_only|max_prob}`
- `--use_head_mixture`: enable differentiable mixture of LM heads
- `--head_topk K`: enable **fast-k** mixing over top-K heads per path
- `--lb_coef`, `--gold_aux_coef`, `--tether_coef`: regularizers for stability
- `--freeze_split_gates`, `--freeze_all_transformer`: for staged training

---

## 🔬 Repro Notes

- **Dataset**: WikiText-103 (train for a few thousand steps to reproduce the improvement quickly)
- **Hardware**: single A100 / T4 works (adjust batch/grad-accum)
- **Seeds**: set `--seed 42` for determinism where possible
- **Logging**: checkpoints and metrics under `--checkpoint_dir`

---

## 🛠️ Roadmap

This project is ongoing — current and planned work includes:

- ✅ Dual path and quad path models with multiple LM headers
- ✅ Fast-k header selection for efficient inference
- 🚧 Quantization of multiple layers to **reduce model size** while keeping accuracy
- 🚧 Domain-specific specialization experiments
- 🚧 **GPT-3 quality** with GPT-2 DMPT
- 🚧 **Selective path training**: search for the best-performing path and update only that path
- 🚧 **Next-N token training**: extend beyond single-token prediction to improve context modeling

---

## 📄 License

MIT (see LICENSE).

---

## 📚 Citation

```bibtex
@article{dmpt2025,
  title   = {Dynamic Multiple Path Transformer: Efficient Multi-Expert Language Modeling},
  author  = {Songnian Qian},
  year    = {2025},
  journal = {GitHub Repository},
  url     = {https://github.com/yourusername/DMPT}
}
```

---

## 🙌 Acknowledgments

- Built on top of HuggingFace Transformers and GPT-2.
- Thanks to the open-source ML community.
