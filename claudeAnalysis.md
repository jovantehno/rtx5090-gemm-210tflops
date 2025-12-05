# CUDA-L2 Repository Analysis

## Overview

**CUDA-L2** is a research project that combines large language models (LLMs) and reinforcement learning (RL) to automatically generate optimized CUDA kernels for half-precision general matrix multiplication (HGEMM). The system produces kernels that outperform NVIDIA's closed-source cuBLAS library.

Paper: https://arxiv.org/pdf/2512.02551

## Core Concept

Traditional CUDA kernel optimization requires expert knowledge of GPU architecture, memory hierarchies, and low-level programming. CUDA-L2 automates this process by:

1. Using an LLM to generate candidate kernel configurations
2. Applying reinforcement learning to iteratively improve kernel parameters
3. Training kernels specific to each (M, N, K) matrix dimension

## Repository Structure

```
CUDA-L2/
├── kernels/
│   └── a100_F16F16F16F16/     # 1000 pre-trained kernels for A100
│       ├── 64_64_64.cu
│       ├── 1024_1024_1024.cu
│       └── ...
├── cublas/                    # Reference cuBLAS implementations
│   ├── hgemm_cublas.cu
│   ├── hgemm_cublaslt_heuristic.cu
│   └── hgemm_cublaslt_auto_tuning.cu
├── tools/
│   └── utils.py               # Build and utility functions
├── pybind/
│   └── hgemm.cc               # Python bindings
├── benchmarking_offline.py    # Batch benchmarking mode
├── benchmarking_server.py     # Request-based benchmarking mode
├── benchmarking_utils.py      # Shared benchmarking utilities
├── compile.py                 # JIT compilation script
└── eval_one_file.sh           # Main evaluation entry point
```

## Technical Implementation

### Kernel Architecture

Each generated kernel uses NVIDIA CUTLASS CuTe abstractions and implements several optimization techniques:

#### 1. Multi-Stage Pipelining (6 stages)
```cpp
auto KStage = Int<Stages>{};  // Default: 6 stages
```
Hides global memory latency by prefetching data while computing on previously loaded data.

#### 2. Tensor Core Operations
```cpp
using mma_op = SM80_16x8x16_F16F16F16F16_TN;
```
Uses A100's Tensor Core matrix-multiply-accumulate instructions for maximum throughput.

#### 3. Shared Memory Swizzling
```cpp
using SmemLayoutAtom = decltype(composition(
    Swizzle<3, 3, 3>{},
    make_layout(make_shape(Int<8>{}, Int<BK>{}),
                make_stride(Int<BK>{}, Int<1>{}))));
```
Eliminates shared memory bank conflicts through intelligent address mapping.

#### 4. Block Swizzling for L2 Cache
```cpp
template <..., const bool BlockSwizzle = true>
void launch_hgemm_mma_stages_block_swizzle_tn_cute(...)
```
Reorders thread block execution to improve L2 cache hit rates.

#### 5. Asynchronous Memory Copies
```cpp
using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
```
Uses hardware-accelerated async copy from global to shared memory.

### Tunable Parameters Per Kernel

Each kernel configuration optimizes these parameters:
- **BM, BN, BK**: Thread block tile dimensions
- **kStage**: Number of pipeline stages
- **kMmaEURepeatM/N/K**: MMA warp tile repetitions
- **swizzle_stride**: Block swizzling stride for cache optimization
- **Thread block layout**: Distribution of threads for memory copies

### Build System

The project uses PyTorch's JIT compilation:

```python
def build_from_sources(mnk, base_dir, verbose):
    return load(
        name="hgemm_lib",
        sources=get_build_sources(mnk),
        extra_cuda_cflags=get_build_cuda_cflags(),
        ...
    )
```

Key compilation flags:
- `-O3`: Maximum optimization
- `--use_fast_math`: Fast math approximations
- Links against CUTLASS v4.2.1 headers
- Links against cuBLAS for baseline comparison

## Benchmarking Methodology

### Offline Mode
Continuous execution measuring throughput (TFLOPS):
```bash
./eval_one_file.sh --mnk 1024_1024_1024 --mode offline \
    --warmup_seconds 5 --benchmark_seconds 10
```

### Server Mode
Simulates request-based inference with target QPS:
```bash
./eval_one_file.sh --mnk 1024_1024_1024 --mode server \
    --target_qps 100
```

### Metrics Collected
- TFLOPS (teraflops per second)
- Latency in milliseconds
- Speedup ratio vs baselines

## Performance Claims

Tested across 1000 (M, N, K) configurations on A100:

| Baseline | Max Speedup | Configurations with Speedup |
|----------|-------------|----------------------------|
| torch.matmul | 2.74x | Most configurations |
| cuBLAS | 1.23x | Significant portion |
| cuBLASLt-heuristic | 1.20x | Significant portion |
| cuBLASLt-AutoTuning | 1.17x | Subset of configurations |

## Dependencies

- **Python**: 3.x
- **PyTorch**: >= 2.6.0
- **CUTLASS**: v4.2.1 (specific version required)
- **CUDA Architecture**: SM80 (A100/Ampere)

## Limitations

1. **Hardware-Specific**: Kernels trained on A100 only work optimally on A100
2. **Fixed Configurations**: Only 1000 pre-defined (M, N, K) dimensions supported
3. **16-bit Accumulator Only**: F16F16F16F16 (no F32 accumulator option yet)
4. **Dimension Constraints**: Unsupported dimensions require zero-padding to nearest larger configuration

## Future Roadmap

- [ ] 32-bit accumulator support (F16F16F16F32)
- [ ] Denser matrix configurations
- [ ] Support for Ada Lovelace, Hopper, Blackwell architectures
- [ ] Easy deployment integration for open-source LLMs

## Usage Example

```bash
# Set environment
export CUTLASS_DIR=/path/to/cutlass
export TORCH_CUDA_ARCH_LIST="8.0"

# Clone CUTLASS
git clone -b v4.2.1 https://github.com/NVIDIA/cutlass.git cutlass

# Run benchmark
./eval_one_file.sh --mnk 1024_4096_1024 \
    --warmup_seconds 5 \
    --benchmark_seconds 10 \
    --base_dir ./results \
    --gpu_device_id 0 \
    --mode offline
```

## Key Insights

1. **RL + LLM Synergy**: The combination allows exploration of a vast parameter space that would be infeasible for manual tuning.

2. **Per-Configuration Optimization**: Unlike cuBLAS which uses heuristics, CUDA-L2 trains a specific kernel for each matrix size.

3. **Cache-Aware Design**: Block swizzling specifically targets L2 cache behavior, a often-overlooked optimization.

4. **Trade-off**: The approach requires pre-training for each configuration, making it less flexible but more performant for known workloads.
