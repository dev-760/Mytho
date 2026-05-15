<p align="center">
  <h1 align="center">Mytho</h1>
  <p align="center">
    <strong>Recurrent-Depth Transformer</strong><br>
    A research-first PyTorch language model with latent scratchpad reasoning,<br>
    adaptive computation, and mixture-of-experts routing.
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License">
    <img src="https://img.shields.io/badge/version-0.1.0-orange?style=flat-square" alt="Version">
  </p>
</p>

---

## Highlights

- **Recurrent Depth** - A weight-tied transformer block applied for up to N steps with ACT halting
- **Multi-Latent Attention** - Low-rank KV compression to reduce cache size
- **Mixture of Experts** - Top-k routing with load-balancing loss and SwiGLU experts
- **Latent Scratchpad** - Persistent internal workspace that attention reads and experts write
- **Verifier-Guided ACT** - Confidence and uncertainty signals guide adaptive depth
- **Single-GPU Pretraining** - FSDP + AdamW script with activation checkpointing

---

## Architecture

Mytho follows a recurrent depth decoder-only design:

```
  Input tokens
      |
      v
+------------------+
| Token Embedding  |  + RoPE
+--------+---------+
         |
         v
  Recurrent Depth Loop (t = 1..T)
         |
         +--> MythoBlock (MLA + MoE) [weight shared]
         +--> ACT halting (adaptive per token)
         |
         v
+------------------+
| RMSNorm          |
| LM Head (tied)   |
+--------+---------+
         |
         v
       logits
```

### Key Components

| Component | Description | Reference |
|-----------|-------------|-----------|
| **RMSNorm** | Root Mean Square normalization | Zhang and Sennrich, 2019 |
| **RoPE** | Rotary Position Embeddings | Su et al., 2021 |
| **Multi-Latent Attention** | Compressed KV via latent projection | DeepSeek-V2 inspired |
| **MoE** | Top-k routing with load-balancing loss | Shazeer et al., 2017 |
| **ACT** | Adaptive Computation Time for dynamic depth | Graves, 2016 |
| **Latent Scratchpad** | Internal workspace for recurrent reasoning | This repo |
| **Verifier Head** | Confidence and uncertainty signals | This repo |

---

## Project Structure

```
Mytho/
├── mytho_model/
│   ├── __init__.py            # Package exports
│   ├── config.py              # MythoConfig dataclass
│   ├── components.py          # RMSNorm, RoPE, SwiGLU
│   ├── attention.py           # Multi-Latent Attention
│   ├── experts.py             # MoE + Switch routing
│   ├── model.py               # MythoBlock, ACT, MythoModel
│   ├── scratchpad.py          # Latent scratchpad
│   ├── verifier.py            # Verifier head
│   ├── branching.py           # Branching recurrence
│   ├── expert_growth.py       # Expert metrics + growth
│   ├── memory.py              # Hierarchical memory
│   ├── quantized_cache.py     # Quantized KV cache
│   ├── self_consistency.py    # Self-consistency decoding
│   ├── uncertainty.py         # MC-Dropout + Ensemble heads
│   ├── reflexion.py           # Optional Reflexion loop
│   └── react.py               # Optional ReAct loop
├── data.py                    # FineWeb-Edu streaming data pipeline
├── model_configs.py           # Predefined model configs (10M → 7B)
├── pretrain.py                # FSDP + AdamW pretraining (single/multi-GPU)
├── pretrain_t4.py             # T4-optimised single-GPU pretraining
├── train.py                   # Lightweight training (synthetic data)
├── generate.py                # Text generation demo
├── export.py                  # Export to Safetensors / ONNX
├── test_model.py              # Full test suite
├── test_pretrain.py           # Data + FSDP validation
├── mytho_kaggle.ipynb         # Kaggle notebook (2× T4)
├── mytho_colab.ipynb          # Colab notebook (1× T4)
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (recommended for GPU runs)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Smoke Tests

```bash
python test_model.py
python test_pretrain.py
```

### 3. Lightweight Training (synthetic data)

```bash
python train.py --epochs 2 --steps_per_epoch 20 --batch_size 2
```

### 4. Pretrain on FineWeb-Edu (streaming)

Mytho ships with 8 predefined configs via `--model_size`:

| Config | Params | Active/tok | VRAM (est.) | Hardware |
|--------|--------|-----------|-------------|----------|
| **10M** | ~10M | ~8M | <1 GB | Any GPU |
| **50M** | ~50M | ~30M | ~1-2 GB | Any GPU |
| **100M** | ~100M | ~65M | ~2-3 GB | T4 / free Colab |
| **150M** | ~150M | ~90M | ~3-4 GB | T4 |
| **500M** | ~500M | ~160M | ~5-6 GB | T4 |
| **1B** | ~1.05B | ~300M | ~8-10 GB | T4 |
| **3B** | ~3.2B | ~570M | ~13-15 GB | T4 (tight) |
| **7B** | ~7B | ~1.2B | ~40+ GB | A100 / multi-GPU |

```bash
# Kaggle 2× T4 (recommended — uses FSDP across both GPUs)
torchrun --nproc_per_node=2 pretrain.py --model_size 500M --dtype fp16

# Colab / single T4
python pretrain_t4.py --model_size 100M --grad_checkpoint --optim_8bit

# FSDP single-GPU with CPU offload
python pretrain.py --model_size 1B --dtype fp16 --cpu_offload

# Quick smoke-test
python pretrain_t4.py --model_size 10M --max_docs 50 --max_steps 20
```

### 5. Generate Text

```bash
python generate.py --checkpoint checkpoints_pretrain/step_1000.pt --prompt "Once upon"
```

### 6. Export Model

Checkpoints are saved as both **PyTorch** (`.pt`) and **Safetensors** (`.safetensors`) during training.

```bash
# Export to safetensors
python export.py --checkpoint checkpoints_t4/step_2000.pt --format safetensors

# Export to ONNX (inference-only, fixed depth)
python export.py --checkpoint checkpoints_t4/step_2000.pt --format onnx

# Both
python export.py --checkpoint checkpoints_t4/step_2000.pt --format all
```

| Format | Contains | Use Case |
|--------|----------|----------|
| `.pt` | Weights + optimizer + config | Resume training |
| `.safetensors` | Weights only | Portable, fast loading |
| `.onnx` | Full inference graph | Deployment, cross-platform |

---

## Reasoning Mode

Mytho does not emit explicit <think> tokens. Instead, it uses latent reasoning:

1. A **latent scratchpad** stores internal workspace tensors.
2. The **verifier head** estimates confidence and uncertainty.
3. **Uncertainty-driven ACT** halts computation when signals stabilize.

This enables chain-of-thought style computation without exposing hidden steps.

---

## Training Data

The pretraining pipeline streams FineWeb-Edu from HuggingFace and tokenizes
on the fly using GPT-2 BPE via tiktoken. See data.py for details.

---

## API Reference

### Model

```python
from mytho_model import MythoConfig, MythoModel

config = MythoConfig(d_model=768, n_experts=8, max_depth=12)
model = MythoModel(
    config,
    use_scratchpad=True,
    d_scratch=128,
    use_branching=True,
    n_branches=2,
    use_memory=True,
)

out = model(input_ids, labels=labels)
print(out["loss"], out["mean_depth"])

generated = model.generate(input_ids, max_new_tokens=128)
```

### Expert Monitoring

```python
from mytho_model import ExpertMetrics, DynamicExpertGrowth

metrics = ExpertMetrics(n_experts=8)
metrics.update(router_logits, topk_indices)
report = metrics.compute(model.blocks[0].moe)
grower = DynamicExpertGrowth()
actions = grower.step(model.blocks[0].moe, metrics)
```

---

## References

- Adaptive Computation Time (Graves, 2016)
- Mixture of Experts (Shazeer et al., 2017)
- Switch Transformer (Fedus et al., 2022)
- Self-Consistency Decoding (Wang et al., 2023)
- MemGPT (Packer et al., 2023)

---

<p align="center">
  <sub>Built for research and rapid prototyping</sub>
</p>

