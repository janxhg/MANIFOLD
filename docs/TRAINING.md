# Training Guide

## Quick Start

```bash
# Math Oracle (fast demo)
python scripts/train.py \
    --model configs/model/gfn_medium.yaml \
    --training configs/training/math_oracle.yaml \
    --hardware configs/hardware/rtx_4090.yaml
```

## Configuration

### Model Configs (`configs/model/`)
| File | Params | Use Case |
|------|--------|----------|
| `gfn_small.yaml` | ~2M | Low VRAM, quick tests |
| `gfn_medium.yaml` | ~13M | Balanced training |
| `gfn_large.yaml` | ~56M | Benchmark, scaling |

### Training Configs (`configs/training/`)
| File | Dataset | Purpose |
|------|---------|---------|
| `math_oracle.yaml` | Math | Fast architecture demo |

### Hardware Configs (`configs/hardware/`)
| File | GPU | Notes |
|------|-----|-------|
| `rtx_4090.yaml` | RTX 4090 | 24GB, TF32 enabled |
| `gtx_1650.yaml` | GTX 1650 | 4GB, conservative |

## Monitoring

- **Console**: Loss, speed (ex/s), live demo
- **TensorBoard**: `tensorboard --logdir logs/`
- **Checkpoints**: `checkpoints/<run>/epoch_N.pt`

## Common Issues

### NaN Loss
- Lower learning rate (8e-4 → 5e-4)
- Increase weight decay (0.1 → 0.2)
- Check `geometry.py` clamp values

### OOM
- Reduce batch_size in training config
- Use smaller model config
- Disable AMP (set `use_amp: false`)
