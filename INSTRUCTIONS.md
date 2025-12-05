# CUDA GEMM Examples - Testing Instructions

## Quick Start

```bash
cd <path-to-repo>/examples

# Build all examples
make all

# Run all examples with benchmarks
make run

# Run a single example
./07_async_copy_gemm
```

## Comparing to cuBLAS Baseline

To compare your results against cuBLAS (the "gold standard"), you can use the CUDA-L2 repo's benchmarking tools.

### Method 1: Using CUDA-L2's Benchmark Scripts

```bash
cd <path-to-cuda-l2>/CUDA-L2

# Set up environment
export CUTLASS_DIR=/path/to/cutlass  # Clone v4.2.1 first if needed
export TORCH_CUDA_ARCH_LIST="8.0"    # Your GPU arch

# Run benchmark for a specific matrix size
./eval_one_file.sh \
    --mnk 1024_1024_1024 \
    --warmup_seconds 5 \
    --benchmark_seconds 10 \
    --base_dir ./results \
    --gpu_device_id 0 \
    --mode offline
```

### Method 2: Quick cuBLAS Comparison Script

Create and run this script to compare against cuBLAS directly:

```bash
# Save as: compare_cublas.py
# Run with: python compare_cublas.py

import torch
import time

def benchmark_cublas(m, n, k, warmup=10, iterations=100):
    """Benchmark torch.matmul (uses cuBLAS internally)"""
    a = torch.randn(m, k, dtype=torch.float16, device='cuda')
    b = torch.randn(k, n, dtype=torch.float16, device='cuda')

    # Warmup
    for _ in range(warmup):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        c = torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iterations
    tflops = (2 * m * n * k) / (ms / 1000) / 1e12
    return ms, tflops

if __name__ == "__main__":
    sizes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    print("cuBLAS (torch.matmul) Baseline Performance")
    print("=" * 50)
    print(f"{'Size':^20} {'Time (ms)':^12} {'TFLOPS':^12}")
    print("-" * 50)

    for m, n, k in sizes:
        ms, tflops = benchmark_cublas(m, n, k)
        print(f"{m}x{n}x{k:^10} {ms:^12.3f} {tflops:^12.2f}")
```

Run it:
```bash
python compare_cublas.py
```

## Expected Results Comparison

### Your GPU Results (from make run)

| Example | Size | TFLOPS |
|---------|------|--------|
| 01 Naive | 1024³ | ~7 |
| 03 Double-buffer | 1024³ | ~17 |
| 06 Block swizzle | 4096³ | ~71 |
| 07 Async pipeline | 1024³ | ~95 |

### Typical cuBLAS Performance

| GPU | Size | cuBLAS TFLOPS |
|-----|------|---------------|
| RTX 3090 | 4096³ | ~120-140 |
| RTX 4090 | 4096³ | ~180-220 |
| A100 | 4096³ | ~250-280 |

### CUDA-L2 Claims (on A100)

| Baseline | CUDA-L2 Speedup |
|----------|-----------------|
| torch.matmul | up to 2.74x |
| cuBLAS | up to 1.23x |
| cuBLASLt-AutoTuning | up to 1.17x |

## Profiling Your Kernels

Use NVIDIA Nsight Compute for detailed analysis:

```bash
# Profile a specific example
ncu --set full -o profile_07 ./07_async_copy_gemm

# View the report
ncu-ui profile_07.ncu-rep

# Quick metrics on command line
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
./07_async_copy_gemm
```

Key metrics to check:
- `sm__throughput` - SM utilization (target: >80%)
- `dram__throughput` - Memory bandwidth (target: >70%)
- `smsp__sass_thread_inst_executed_op_hmma` - Tensor core usage

## Modifying Examples for Different Sizes

Edit the `#define` at the top of each example:

```cpp
// In any example .cu file
#define M 2048  // Change these
#define N 2048
#define K 2048
```

Then rebuild:
```bash
make clean && make all
```

## Running on Different GPUs

Change the architecture in the Makefile:

```bash
# RTX 30xx (Ampere)
make ARCH=sm_80 all

# RTX 40xx (Ada Lovelace)
make ARCH=sm_89 all

# A100
make ARCH=sm_80 all

# H100 (Hopper)
make ARCH=sm_90 all

# RTX 50xx (Blackwell) - when CUDA supports it
make ARCH=sm_100 all
```

## Troubleshooting

### "misaligned address" error
- Matrix dimensions must be multiples of tile sizes (usually 64 or 128)
- cp.async requires 16-byte aligned addresses

### Low performance
- Check GPU is not thermal throttling: `nvidia-smi -q -d PERFORMANCE`
- Ensure no other processes using GPU: `nvidia-smi`
- Verify correct architecture: `nvcc --version` and check ARCH flag

### Compilation errors
- Ensure CUDA toolkit installed: `nvcc --version`
- Check compute capability matches your GPU
- For CUDA-L2: verify CUTLASS_DIR is set correctly

## File Organization

```
examples/
├── 01_naive_gemm.cu           # Baseline implementation
├── 02_tiled_gemm.cu           # Shared memory tiling
├── 03_double_buffered_gemm.cu # Software pipelining
├── 04_wmma_tensor_core_gemm.cu # Tensor cores via WMMA
├── 05_swizzled_gemm.cu        # Bank conflict elimination
├── 06_block_swizzle_l2_gemm.cu # L2 cache optimization
├── 07_async_copy_gemm.cu      # Async memory pipeline
├── Makefile                   # Build system
├── README.md                  # Example descriptions
├── INSTRUCTIONS.md            # This file
├── claudeAnalysis.md          # CUDA-L2 repo analysis
└── claude-5090.md             # RTX 5090 adaptation guide
```
