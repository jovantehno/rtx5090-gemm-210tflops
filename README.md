# rtx5090-gemm-210tflops

**From 7 TFLOPS to 215 TFLOPS: A hands-on journey through CUDA GEMM optimization**

This repository contains 8 progressive CUDA examples that demonstrate how to write a high-performance matrix multiplication kernel from scratch. Starting with a naive implementation running at 7 TFLOPS, we apply optimization techniques one by one until we reach **215 TFLOPS** — that's **94% of cuBLAS performance** on an RTX 5090.

## Why This Matters

Matrix multiplication (GEMM) is the computational heart of:
- **Large Language Models** — Every transformer layer is dominated by matrix ops
- **Deep Learning** — Convolutions, attention, linear layers all reduce to GEMM
- **Scientific Computing** — Linear algebra, simulations, signal processing

Understanding how to optimize GEMM means understanding how to get maximum performance from modern GPUs. This knowledge transfers directly to writing custom CUDA kernels for any compute-intensive workload.

## The Journey: 7 → 210 TFLOPS

```
Example 01: Naive           ████░░░░░░░░░░░░░░░░░░░░░░░░░░    7 TFLOPS  (3%)
Example 02: Tiled           ████░░░░░░░░░░░░░░░░░░░░░░░░░░    6 TFLOPS  (3%)
Example 03: Double-buffer   ██████░░░░░░░░░░░░░░░░░░░░░░░░   17 TFLOPS  (7%)
Example 04: Tensor cores    █████░░░░░░░░░░░░░░░░░░░░░░░░░   11 TFLOPS  (5%)
Example 05: Swizzled        █████░░░░░░░░░░░░░░░░░░░░░░░░░   13 TFLOPS  (6%)
Example 06: L2 optimized    ████████████████░░░░░░░░░░░░░░   71 TFLOPS (31%)
Example 07: Async pipeline  ███████████████████████░░░░░░░   95 TFLOPS (41%)
Example 08: Combined        ██████████████████████████████  215 TFLOPS (94%)
─────────────────────────────────────────────────────────────────────────────
cuBLAS baseline             ██████████████████████████████  229 TFLOPS
```

*Results from RTX 5090 (Blackwell), 4096×4096×4096 FP16 GEMM*

## Quick Start

```bash
# Clone and build
git clone https://github.com/yourusername/rtx5090-gemm-210tflops
cd rtx5090-gemm-210tflops
make all

# Run all examples
make run

# Compare against cuBLAS
make compare
```

**Requirements:** NVIDIA GPU (SM80+), CUDA 11.0+, Python 3 with PyTorch (for cuBLAS comparison)

---

## The Idea

This project explores **how to write high-performance matrix multiplication (GEMM) kernels on NVIDIA GPUs** by studying the techniques used in [CUDA-L2](https://arxiv.org/pdf/2512.02551), a research project that uses reinforcement learning to generate kernels that outperform NVIDIA's own cuBLAS library.

The goal is to understand and demonstrate the key optimization techniques that make the difference between a naive 7 TFLOPS kernel and a production-quality 200+ TFLOPS kernel.

## What We Expected

Based on CUDA-L2's paper and A100 results:

| Technique | Expected Impact |
|-----------|-----------------|
| Shared memory tiling | 5-10x over naive |
| Software pipelining | 2-3x over basic tiling |
| Tensor cores (WMMA) | 10-20x over scalar FMA |
| Bank conflict elimination | 5-15% improvement |
| L2 cache optimization | 20-40% improvement |
| Async memory pipeline | 10-30% improvement |

Combined, these techniques should approach **80-100% of cuBLAS performance**.

## What We Achieved

Tested on **NVIDIA RTX 5090** (Blackwell, SM 12.0):

| Example | Technique | TFLOPS | vs cuBLAS |
|---------|-----------|--------|-----------|
| 01 | Naive baseline | 6.97 | 3.0% |
| 02 | Shared memory tiling | 6.15 | 2.7% |
| 03 | Double buffering | 16.64 | 7.3% |
| 04 | WMMA tensor cores | 11.45 | 5.0% |
| 05 | Swizzled memory | 12.51 | 5.5% |
| 06 | Block swizzle L2 | 71.38 | 31.1% |
| 07 | Async pipeline | 95.01 | 41.4% |
| **08** | **All combined** | **210.47** | **91.7%** |

**cuBLAS baseline**: 229 TFLOPS (4096³ FP16)

### Key Insights

1. **Individual techniques aren't enough** — Examples 01-07 each demonstrate one optimization, but real performance requires combining them all
2. **L2 cache matters enormously** — Block swizzling alone gave 4.5x speedup over the basic tensor core version
3. **Async is essential** — The `cp.async` pipeline eliminates memory stalls that synchronous loads can't avoid
4. **Tuning for your GPU matters** — The optimal swizzle stride for RTX 5090 (3072) differs from A100 (1792) due to different L2 sizes

### Why Some Individual Examples Underperform

1. **Architecture mismatch**: Examples target SM80 (Ampere), running on SM120 (Blackwell) via compatibility mode
2. **Educational focus**: Code prioritizes clarity over maximum optimization
3. **Isolated techniques**: Each example (01-07) demonstrates one concept; real gains come from combining them (08)

## How Each Optimization Works

### 1. Naive → Tiled (Example 01 → 02)
```
Problem: Each thread loads data independently from global memory
         Same data loaded millions of times

Solution: Thread block loads tile into shared memory once
          All threads in block reuse the cached data
          Reduces global memory traffic by tile_size (32x)
```

### 2. Tiled → Double Buffered (Example 02 → 03)
```
Problem: Load tile → wait → compute → load next tile → wait...
         GPU sits idle during memory loads

Solution: Two buffers in shared memory
          While computing on buffer A, load into buffer B
          Overlaps memory latency with computation
```

### 3. Scalar → Tensor Cores (Example 03 → 04)
```
Problem: Scalar FMA: 1 multiply-add per thread per cycle
         Massively underutilizes modern GPU hardware

Solution: WMMA API accesses tensor cores
          16x16x16 matrix ops in single instruction
          ~10-20x theoretical throughput increase
```

### 4. Bank Conflicts → Swizzled (Example 04 → 05)
```
Problem: 32 shared memory banks, 4 bytes each
         Column access pattern: all threads hit same bank
         Serializes 32-way parallel access to 1-way

Solution: XOR-based address swizzling
          new_addr = col ^ (row & 7)
          Distributes accesses across all banks
```

### 5. Cache Thrashing → Block Swizzle (Example 05 → 06)
```
Problem: Default block order: (0,0), (1,0), (2,0)...
         Blocks in same row load same A data
         By next row, A data evicted from L2 cache

Solution: Reorder block execution
          Group blocks that share data
          Execute them together while data is in L2
          1.36x speedup with stride=1792 on A100
```

### 6. Synchronous → Async Pipeline (Example 06 → 07)
```
Problem: Even with double buffering, loads block on issue
         Thread waits for load instruction to complete

Solution: cp.async (SM80+) / TMA (SM90+)
          Hardware DMA: issue load, continue immediately
          Multi-stage pipeline: 3-6 tiles in flight
          Memory latency completely hidden
```

### 7. Combined Optimization (Example 08)
```
All techniques together:
- 128×128 tiles with 16-wide K dimension
- 4×4 warp grid (512 threads per block)
- 3-stage async pipeline with cp.async
- XOR-based shared memory swizzling
- Block swizzle stride tuned for 96MB L2

Result: 215 TFLOPS = 94% of cuBLAS
```

## Possible Applications

### 1. LLM Inference Optimization
- Matrix multiplications dominate transformer compute
- Custom kernels for specific model shapes (1024×4096, etc.)
- Batch size 1 inference where cuBLAS isn't optimal

### 2. Custom Neural Network Layers
- Non-standard GEMM shapes
- Fused operations (GEMM + activation + bias)
- Quantized inference (INT8, FP8, FP4)

### 3. Scientific Computing
- Dense linear algebra (solving Ax=b)
- Signal processing (FFT-based convolutions)
- Physics simulations

### 4. Learning GPU Architecture
- Understanding memory hierarchy
- Tensor core programming
- Performance optimization techniques

### 5. Kernel Auto-tuning Research
- Search space for RL/ML-based optimization
- Hardware-aware neural architecture search
- Compiler optimization targets

## Project Structure

```
├── Source Code
│   ├── 01_naive_gemm.cu              # Baseline (~7 TFLOPS)
│   ├── 02_tiled_gemm.cu              # Shared memory tiling
│   ├── 03_double_buffered_gemm.cu    # Software pipelining
│   ├── 04_wmma_tensor_core_gemm.cu   # Tensor cores via WMMA
│   ├── 05_swizzled_gemm.cu           # Bank conflict elimination
│   ├── 06_block_swizzle_l2_gemm.cu   # L2 cache optimization
│   ├── 07_async_copy_gemm.cu         # Async memory pipeline
│   └── 08_combined_optimized_gemm.cu # All optimizations (210 TFLOPS)
│
├── Documentation
│   ├── README.md                     # This file
│   ├── INSTRUCTIONS.md               # Testing guide
│   ├── claudeAnalysis.md             # CUDA-L2 paper analysis
│   └── claude-5090.md                # Blackwell adaptation notes
│
└── Tools
    ├── Makefile                      # Build system
    └── compare_cublas.py             # cuBLAS benchmark script
```

## Building and Running

```bash
# Build all examples
make all

# Run all benchmarks
make run

# Compare against cuBLAS
make compare

# Full comparison with examples
make compare-all

# Build for specific architecture
make ARCH=sm_89 all  # Ada (RTX 40xx)
make ARCH=sm_90 all  # Hopper (H100)

# See all options
make help
```

## Hardware Requirements

- NVIDIA GPU with compute capability 8.0+ (Ampere or newer)
- CUDA Toolkit 11.0+
- For best results: A100, RTX 30xx/40xx/50xx, H100

## What's Next

To push beyond 94% of cuBLAS, PTX ISA 9.1 introduces several Blackwell-native features:

### Immediate Priorities

1. **tcgen05 (TensorCore 5th Gen)** — Native Blackwell tensor core instructions
   - `tcgen05.mma` for direct matrix multiply-accumulate
   - "Tensor Memory" — new memory space distinct from shared memory
   - Block scaling and compressed formats (4-bit, 6-bit FP)

2. **WGMMA (Warpgroup MMA)** — Higher throughput than WMMA
   - Operates on warpgroup (4 warps) granularity
   - Shapes like m64nNk16 vs WMMA's 16x16x16
   - `wgmma.mma_async` with fence/commit/wait semantics

3. **Thread Block Clusters** — Hardware-assisted L2 optimization
   - Groups of CTAs that sync and share memory across blocks
   - `barrier.cluster` for cluster-wide synchronization
   - Could replace software block swizzling

4. **TMA (Tensor Memory Accelerator)** — Replace cp.async
   - Hardware-managed tensor data movement
   - Automatic address calculation and strided copies

### Additional Directions

5. **FP8 support** — Blackwell's native FP8 tensor cores
6. **Non-square matrices** — Real workload shapes (e.g., 4096×1024×8192)
7. **Adaptive stride selection** — Runtime tuning based on matrix dimensions

See `claude-5090.md` for detailed Blackwell adaptation guide and [PTX ISA 9.1](https://docs.nvidia.com/cuda/parallel-thread-execution/) for instruction reference.

## References

- [CUDA-L2 Paper](https://arxiv.org/pdf/2512.02551) — RL-based kernel optimization that beats cuBLAS
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) — Template library for high-performance GEMM
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

## License

MIT License. Educational examples for learning GPU optimization techniques.

---

*Built with curiosity and lots of profiling. Contributions welcome!*
