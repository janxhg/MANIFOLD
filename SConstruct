"""
SConstruct for GFN CUDA Kernels
================================
Compiles CUDA kernels as a shared library loadable by PyTorch.

Usage:
    scons              # Build release
    scons debug=1      # Build with debug symbols
    scons clean        # Clean build artifacts
"""

import os
import sys

# Detect PyTorch paths
try:
    import torch
    torch_root = os.path.dirname(torch.__file__)
    torch_include = os.path.join(torch_root, 'include')
    torch_lib = os.path.join(torch_root, 'lib')
    print(f"Found PyTorch at: {torch_root}")
except ImportError:
    print("ERROR: PyTorch not found. Install with: pip install torch")
    Exit(1)

# CUDA paths (Windows)
cuda_root = os.environ.get('CUDA_PATH', 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0')
if not os.path.exists(cuda_root):
    print(f"ERROR: CUDA not found at {cuda_root}")
    print("Set CUDA_PATH environment variable to your CUDA installation")
    Exit(1)

cuda_include = os.path.join(cuda_root, 'include')
cuda_lib = os.path.join(cuda_root, 'lib', 'x64')
nvcc = os.path.join(cuda_root, 'bin', 'nvcc.exe')

print(f"Found CUDA at: {cuda_root}")

# Build configuration
debug = int(ARGUMENTS.get('debug', 0))

# Base environment (MSVC for C++)
env = Environment()

# CUDA compilation flags
if debug:
    cuda_flags = ['-g', '-G', '-O0']
    cpp_flags = ['/Zi', '/Od', '/MDd', '/std:c++17']
    print("Building in DEBUG mode")
else:
    cuda_flags = ['-O3', '--use_fast_math', '-DNDEBUG']
    cpp_flags = ['/O2', '/MD', '/DNDEBUG', '/std:c++17']
    print("Building in RELEASE mode")

# CUDA arch for GTX 1650 (Turing, compute capability 7.5)
cuda_flags += [
    '-arch=sm_75',
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda',
    '-Xcompiler=/MD' + ('d' if debug else ''),
]

# Include paths
env.Append(
    CPPPATH=[
        torch_include,
        os.path.join(torch_include, 'torch', 'csrc', 'api', 'include'),
        cuda_include,
        'src/cuda',
    ],
    LIBPATH=[torch_lib, cuda_lib],
    CPPFLAGS=cpp_flags,
)

# Define CUDA builder
def cuda_builder(target, source, env):
    """Custom builder for .cu files"""
    src = str(source[0])
    obj = str(target[0])
    
    # NVCC command
    cmd = [nvcc]
    cmd += cuda_flags
    cmd += ['-c', src]
    cmd += ['-o', obj]
    
    # Add include paths
    for path in env['CPPPATH']:
        cmd += [f'-I{path}']
    
    print(f"Compiling CUDA: {src}")
    ret = os.system(' '.join(cmd))
    if ret != 0:
        print(f"ERROR: CUDA compilation failed for {src}")
        Exit(1)
    return None

# Register CUDA builder
cuda_bld = Builder(action=cuda_builder, suffix='.obj', src_suffix='.cu')
env.Append(BUILDERS={'CUDA': cuda_bld})

# Compile sources
cpp_obj = env.Object('src/cuda/cuda_kernels.cpp')
cu_obj1 = env.CUDA('src/cuda/kernels/christoffel_fused.cu')
cu_obj2 = env.CUDA('src/cuda/kernels/leapfrog_fused.cu')

# Link shared library
target = 'gfn_cuda' + ('_debug' if debug else '')
lib = env.SharedLibrary(
    target,
    [cpp_obj, cu_obj1, cu_obj2],
    LIBS=['c10', 'torch_cpu', 'torch_cuda', 'cudart'],
    SHLIBPREFIX='',
    SHLIBSUFFIX='.pyd'
)

# Install to src/cuda/
install = env.Install('src/cuda/', lib)

Default(install)

# Clean
env.Clean(install, ['*.obj', '*.exp', '*.lib', 'src/cuda/*.pyd', 'src/cuda/*.exp', 'src/cuda/*.lib'])
