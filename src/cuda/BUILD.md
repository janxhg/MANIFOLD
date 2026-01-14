# GFN CUDA Kernels Build Instructions

## Option 1: SCons (Recommended for Development)

### Prerequisites
- SCons: `pip install scons`
- CUDA Toolkit (nvcc)
- PyTorch with CUDA

### Build Commands
```bash
# Release build (optimized)
scons

# Debug build (with symbols)
scons debug=1

# Clean
scons -c
```

### Output
- Creates `gfn_cuda.pyd` in `src/cuda/`
- Can be loaded directly by PyTorch

### Advantages
- **AOT compilation**: No delay on first import
- **Explicit control**: Custom optimization flags
- **Debugging**: Can add `-g -G` for CUDA debugging
- **Incremental builds**: Only recompiles changed files

---

## Option 2: PyTorch JIT (Current Default)

### How it works
- Automatic compilation on first `import src.cuda.ops`
- Uses `torch.utils.cpp_extension.load()`
- Caches in `~/.cache/torch_extensions/`

### Advantages
- **Zero setup**: Works out of the box
- **Portable**: No manual build step required

### Disadvantages
- **First import delay**: ~30-60s compilation
- **Less control**: Uses PyTorch's default flags

---

## Switching Between Methods

### Use SCons build:
1. Run `scons` to build
2. Replace import in `src/cuda/__init__.py`:
   ```python
   from .ops_scons import christoffel_fused, leapfrog_fused
   ```

### Use JIT build (default):
- Just use the existing `ops.py`
- No changes needed

---

## Troubleshooting

### "nvcc not found"
- Ensure CUDA bin directory is in PATH
- Windows: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`

### "Cannot find torch/extension.h"
- PyTorch installation issue
- Reinstall: `pip install --force-reinstall torch`

### "Unsupported architecture"
- Update `-arch=sm_75` in `SConstruct` to match your GPU
- GTX 1650: `sm_75`
- RTX 30xx: `sm_86`
- RTX 40xx: `sm_89`
