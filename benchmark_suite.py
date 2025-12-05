#!/usr/bin/env python3
"""
Comprehensive GEMM Benchmarking Suite for Blackwell Characterization

Collects performance data across:
- Matrix sizes (512 to 16384)
- Swizzle strides (512 to 16384)
- Tile configurations
- Pipeline depths

Outputs CSV for analysis and paper figures.

Usage:
    python benchmark_suite.py                    # Quick run
    python benchmark_suite.py --full             # Full sweep
    python benchmark_suite.py --ncu              # With Nsight Compute metrics
"""

import argparse
import subprocess
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not found. cuBLAS comparison disabled.")


def get_gpu_info():
    """Get GPU specifications"""
    info = {
        "name": "Unknown",
        "compute_cap": "Unknown",
        "memory_gb": 0,
        "l2_cache_mb": 0,
        "sm_count": 0
    }

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        parts = result.stdout.strip().split(", ")
        info["name"] = parts[0]
        info["memory_gb"] = int(parts[1]) // 1024
    except:
        pass

    if HAS_TORCH and torch.cuda.is_available():
        info["name"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        info["compute_cap"] = f"{cap[0]}.{cap[1]}"
        props = torch.cuda.get_device_properties(0)
        info["sm_count"] = props.multi_processor_count
        info["l2_cache_mb"] = props.l2_cache_size // (1024 * 1024) if hasattr(props, 'l2_cache_size') else 0

    # Known L2 sizes (not always exposed via API)
    if "5090" in info["name"]:
        info["l2_cache_mb"] = 96
    elif "4090" in info["name"]:
        info["l2_cache_mb"] = 72
    elif "A100" in info["name"]:
        info["l2_cache_mb"] = 40
    elif "H100" in info["name"]:
        info["l2_cache_mb"] = 50

    return info


def benchmark_cublas(m, n, k, warmup=10, iterations=50):
    """Benchmark cuBLAS via PyTorch"""
    if not HAS_TORCH:
        return None, None

    try:
        a = torch.randn(m, k, dtype=torch.float16, device='cuda')
        b = torch.randn(k, n, dtype=torch.float16, device='cuda')

        for _ in range(warmup):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()

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
    except RuntimeError:
        return None, None


def compile_benchmark_kernel(output_name="benchmark_gemm"):
    """Compile the configurable benchmark kernel"""

    kernel_source = '''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32

#define CUDA_CHECK(call) do { \\
    cudaError_t err = call; \\
    if (err != cudaSuccess) { \\
        fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err)); \\
        exit(1); \\
    } \\
} while(0)

// Configurable parameters passed at runtime
struct GemmConfig {
    int M, N, K;
    int BM, BN, BK;
    int num_stages;
    int swizzle_stride;
    int warps_m, warps_n;
    bool use_swizzle;
};

template<int BM, int BN, int BK, int NUM_STAGES, int WARPS_M, int WARPS_N>
__global__ void gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K,
    int swizzle_stride,
    bool use_swizzle
) {
    constexpr int THREADS = WARPS_M * WARPS_N * WARP_SIZE;

    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = As + NUM_STAGES * BM * BK;

    int bx, by;
    if (use_swizzle && swizzle_stride > 0) {
        int blocks_n = (N + BN - 1) / BN;
        int blocks_per_group = swizzle_stride / BN;
        if (blocks_per_group < 1) blocks_per_group = 1;

        int group_id = blockIdx.x / blocks_per_group;
        int local_bx = blockIdx.x % blocks_per_group;

        bx = blockIdx.x;
        by = blockIdx.y;

        // Swizzle pattern
        int total_blocks = blocks_n * ((M + BM - 1) / BM);
        int linear_id = blockIdx.y * blocks_n + blockIdx.x;
        int swizzle_width = blocks_per_group;
        int group_row = linear_id / (swizzle_width * ((M + BM - 1) / BM));

        bx = (group_row * blocks_per_group + (linear_id % blocks_per_group)) % blocks_n;
        by = (linear_id / blocks_per_group) % ((M + BM - 1) / BM);
    } else {
        bx = blockIdx.x;
        by = blockIdx.y;
    }

    if (bx * BN >= N || by * BM >= M) return;

    int warpId = threadIdx.x / WARP_SIZE;
    int warp_row = warpId / WARPS_N;
    int warp_col = warpId % WARPS_N;

    int block_row = by * BM;
    int block_col = bx * BN;

    constexpr int WMMA_TILES_M = BM / WMMA_M / WARPS_M;
    constexpr int WMMA_TILES_N = BN / WMMA_N / WARPS_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WMMA_TILES_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WMMA_TILES_N];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag[WMMA_TILES_M][WMMA_TILES_N];

    #pragma unroll
    for (int i = 0; i < WMMA_TILES_M; i++) {
        #pragma unroll
        for (int j = 0; j < WMMA_TILES_N; j++) {
            wmma::fill_fragment(c_frag[i][j], __float2half(0.0f));
        }
    }

    int tid = threadIdx.x;
    int num_tiles_k = (K + BK - 1) / BK;

    // Prologue: fill pipeline
    #pragma unroll
    for (int stage = 0; stage < NUM_STAGES - 1 && stage < num_tiles_k; stage++) {
        int tile_k = stage * BK;
        half* As_stage = As + stage * BM * BK;
        half* Bs_stage = Bs + stage * BK * BN;

        for (int i = tid * 8; i < BM * BK; i += THREADS * 8) {
            int row = i / BK;
            int col = i % BK;
            if (col + 8 <= BK) {
                int global_row = block_row + row;
                int global_col = tile_k + col;
                if (global_row < M && global_col + 8 <= K) {
                    __pipeline_memcpy_async(
                        As_stage + row * BK + col,
                        A + global_row * K + global_col,
                        16
                    );
                }
            }
        }

        for (int i = tid * 8; i < BK * BN; i += THREADS * 8) {
            int row = i / BN;
            int col = i % BN;
            if (col + 8 <= BN) {
                int global_row = tile_k + row;
                int global_col = block_col + col;
                if (global_row < K && global_col + 8 <= N) {
                    __pipeline_memcpy_async(
                        Bs_stage + row * BN + col,
                        B + global_row * N + global_col,
                        16
                    );
                }
            }
        }
        __pipeline_commit();
    }

    // Main loop
    for (int tile_idx = 0; tile_idx < num_tiles_k; tile_idx++) {
        int compute_stage = tile_idx % NUM_STAGES;
        int load_stage = (tile_idx + NUM_STAGES - 1) % NUM_STAGES;

        __pipeline_wait_prior(NUM_STAGES - 2);
        __syncthreads();

        half* As_compute = As + compute_stage * BM * BK;
        half* Bs_compute = Bs + compute_stage * BK * BN;

        int warp_tile_row = warp_row * WMMA_TILES_M * WMMA_M;
        int warp_tile_col = warp_col * WMMA_TILES_N * WMMA_N;

        #pragma unroll
        for (int i = 0; i < WMMA_TILES_M; i++) {
            wmma::load_matrix_sync(a_frag[i], As_compute + (warp_tile_row + i * WMMA_M) * BK, BK);
        }

        #pragma unroll
        for (int j = 0; j < WMMA_TILES_N; j++) {
            wmma::load_matrix_sync(b_frag[j], Bs_compute + warp_tile_col + j * WMMA_N, BN);
        }

        #pragma unroll
        for (int i = 0; i < WMMA_TILES_M; i++) {
            #pragma unroll
            for (int j = 0; j < WMMA_TILES_N; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }

        // Load next tile
        int next_tile = tile_idx + NUM_STAGES - 1;
        if (next_tile < num_tiles_k) {
            int tile_k = next_tile * BK;
            half* As_load = As + load_stage * BM * BK;
            half* Bs_load = Bs + load_stage * BK * BN;

            for (int i = tid * 8; i < BM * BK; i += THREADS * 8) {
                int row = i / BK;
                int col = i % BK;
                if (col + 8 <= BK) {
                    int global_row = block_row + row;
                    int global_col = tile_k + col;
                    if (global_row < M && global_col + 8 <= K) {
                        __pipeline_memcpy_async(
                            As_load + row * BK + col,
                            A + global_row * K + global_col,
                            16
                        );
                    }
                }
            }

            for (int i = tid * 8; i < BK * BN; i += THREADS * 8) {
                int row = i / BN;
                int col = i % BN;
                if (col + 8 <= BN) {
                    int global_row = tile_k + row;
                    int global_col = block_col + col;
                    if (global_row < K && global_col + 8 <= N) {
                        __pipeline_memcpy_async(
                            Bs_load + row * BN + col,
                            B + global_row * N + global_col,
                            16
                        );
                    }
                }
            }
            __pipeline_commit();
        }
    }

    // Store results
    int warp_out_row = block_row + warp_row * WMMA_TILES_M * WMMA_M;
    int warp_out_col = block_col + warp_col * WMMA_TILES_N * WMMA_N;

    #pragma unroll
    for (int i = 0; i < WMMA_TILES_M; i++) {
        #pragma unroll
        for (int j = 0; j < WMMA_TILES_N; j++) {
            int out_row = warp_out_row + i * WMMA_M;
            int out_col = warp_out_col + j * WMMA_N;
            if (out_row < M && out_col < N) {
                wmma::store_matrix_sync(C + out_row * N + out_col, c_frag[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

float benchmark_config(
    const half* d_A, const half* d_B, half* d_C,
    const GemmConfig& cfg,
    int warmup, int iterations
) {
    int blocks_m = (cfg.M + cfg.BM - 1) / cfg.BM;
    int blocks_n = (cfg.N + cfg.BN - 1) / cfg.BN;

    dim3 grid(blocks_n, blocks_m);
    int threads = cfg.warps_m * cfg.warps_n * WARP_SIZE;

    size_t smem_size = cfg.num_stages * (cfg.BM * cfg.BK + cfg.BK * cfg.BN) * sizeof(half);

    cudaFuncSetAttribute(
        gemm_kernel<128, 128, 16, 3, 4, 4>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );

    // Warmup
    for (int i = 0; i < warmup; i++) {
        gemm_kernel<128, 128, 16, 3, 4, 4><<<grid, threads, smem_size>>>(
            d_A, d_B, d_C, cfg.M, cfg.N, cfg.K,
            cfg.swizzle_stride, cfg.use_swizzle
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        gemm_kernel<128, 128, 16, 3, 4, 4><<<grid, threads, smem_size>>>(
            d_A, d_B, d_C, cfg.M, cfg.N, cfg.K,
            cfg.swizzle_stride, cfg.use_swizzle
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iterations;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: %s M N K swizzle_stride [use_swizzle]\\n", argv[0]);
        return 1;
    }

    GemmConfig cfg;
    cfg.M = atoi(argv[1]);
    cfg.N = atoi(argv[2]);
    cfg.K = atoi(argv[3]);
    cfg.swizzle_stride = atoi(argv[4]);
    cfg.use_swizzle = (argc > 5) ? atoi(argv[5]) : 1;
    cfg.BM = 128;
    cfg.BN = 128;
    cfg.BK = 16;
    cfg.num_stages = 3;
    cfg.warps_m = 4;
    cfg.warps_n = 4;

    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, cfg.M * cfg.K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, cfg.K * cfg.N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, cfg.M * cfg.N * sizeof(half)));

    // Initialize with random data
    half* h_A = (half*)malloc(cfg.M * cfg.K * sizeof(half));
    half* h_B = (half*)malloc(cfg.K * cfg.N * sizeof(half));
    srand(42);
    for (int i = 0; i < cfg.M * cfg.K; i++) h_A[i] = __float2half((float)(rand() % 10) / 10.0f);
    for (int i = 0; i < cfg.K * cfg.N; i++) h_B[i] = __float2half((float)(rand() % 10) / 10.0f);
    CUDA_CHECK(cudaMemcpy(d_A, h_A, cfg.M * cfg.K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, cfg.K * cfg.N * sizeof(half), cudaMemcpyHostToDevice));

    float ms = benchmark_config(d_A, d_B, d_C, cfg, 5, 20);
    double tflops = (2.0 * cfg.M * cfg.N * cfg.K / (ms / 1000.0)) / 1e12;

    // JSON output for easy parsing
    printf("{\\"m\\": %d, \\"n\\": %d, \\"k\\": %d, \\"swizzle_stride\\": %d, \\"use_swizzle\\": %d, \\"time_ms\\": %.4f, \\"tflops\\": %.2f}\\n",
           cfg.M, cfg.N, cfg.K, cfg.swizzle_stride, cfg.use_swizzle, ms, tflops);

    free(h_A);
    free(h_B);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
'''

    with open("benchmark_gemm.cu", "w") as f:
        f.write(kernel_source)

    result = subprocess.run(
        ["nvcc", "-O3", "-std=c++17", "-arch=sm_80", "--use_fast_math",
         "benchmark_gemm.cu", "-o", output_name],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"Compilation failed: {result.stderr}")
        return False
    return True


def run_benchmark(m, n, k, swizzle_stride, use_swizzle=1):
    """Run a single benchmark configuration"""
    try:
        result = subprocess.run(
            ["./benchmark_gemm", str(m), str(n), str(k), str(swizzle_stride), str(use_swizzle)],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            return json.loads(result.stdout.strip())
    except Exception as e:
        print(f"Error: {e}")
    return None


def run_ncu_benchmark(m, n, k, swizzle_stride, use_swizzle=1, metrics=None):
    """Run benchmark with Nsight Compute metrics"""
    if metrics is None:
        metrics = [
            "lts__t_sectors_srcunit_tex_op_read.sum",  # L2 read sectors
            "lts__t_sectors_srcunit_tex_op_write.sum", # L2 write sectors
            "lts__t_sector_hit_rate.pct",              # L2 hit rate
            "dram__bytes_read.sum",                    # DRAM reads
            "dram__bytes_write.sum",                   # DRAM writes
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",  # SM utilization
        ]

    metrics_str = ",".join(metrics)

    try:
        result = subprocess.run(
            ["ncu", "--metrics", metrics_str, "--csv",
             "./benchmark_gemm", str(m), str(n), str(k), str(swizzle_stride), str(use_swizzle)],
            capture_output=True, text=True, timeout=120
        )
        return result.stdout
    except Exception as e:
        print(f"NCU Error: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description='GEMM Benchmark Suite')
    parser.add_argument('--full', action='store_true', help='Full parameter sweep')
    parser.add_argument('--ncu', action='store_true', help='Collect Nsight Compute metrics')
    parser.add_argument('--output', type=str, default='benchmark_results', help='Output file prefix')
    args = parser.parse_args()

    # Get GPU info
    gpu_info = get_gpu_info()
    print("=" * 70)
    print("GEMM Benchmark Suite for Blackwell Characterization")
    print("=" * 70)
    print(f"GPU: {gpu_info['name']}")
    print(f"Compute Capability: {gpu_info['compute_cap']}")
    print(f"L2 Cache: {gpu_info['l2_cache_mb']} MB")
    print(f"SMs: {gpu_info['sm_count']}")
    print()

    # Compile benchmark kernel
    print("Compiling benchmark kernel...")
    if not compile_benchmark_kernel():
        print("Failed to compile. Exiting.")
        return
    print("Done.\n")

    # Define sweep parameters
    if args.full:
        matrix_sizes = [512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384]
        swizzle_strides = [0, 512, 1024, 1536, 1792, 2048, 2560, 3072, 4096, 6144, 8192]
    else:
        matrix_sizes = [1024, 2048, 4096, 8192]
        swizzle_strides = [0, 1792, 2048, 3072, 4096]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{args.output}_{timestamp}.csv"

    results = []

    # Main benchmark loop
    print("Running benchmarks...")
    print("-" * 70)
    print(f"{'Size':^12} {'Stride':^10} {'Time (ms)':^12} {'TFLOPS':^12} {'vs cuBLAS':^12}")
    print("-" * 70)

    for size in matrix_sizes:
        # Get cuBLAS baseline
        cublas_ms, cublas_tflops = benchmark_cublas(size, size, size)

        for stride in swizzle_strides:
            use_swizzle = 1 if stride > 0 else 0
            result = run_benchmark(size, size, size, stride, use_swizzle)

            if result:
                result['cublas_tflops'] = cublas_tflops if cublas_tflops else 0
                result['cublas_ms'] = cublas_ms if cublas_ms else 0
                result['vs_cublas'] = result['tflops'] / cublas_tflops if cublas_tflops else 0
                result['gpu'] = gpu_info['name']
                result['l2_cache_mb'] = gpu_info['l2_cache_mb']
                results.append(result)

                vs_cublas_str = f"{result['vs_cublas']:.1%}" if cublas_tflops else "N/A"
                print(f"{size:^12} {stride:^10} {result['time_ms']:^12.3f} {result['tflops']:^12.2f} {vs_cublas_str:^12}")

        print()

    # Save results to CSV
    if results:
        with open(results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {results_file}")

    # Find optimal strides per size
    print("\n" + "=" * 70)
    print("Optimal Swizzle Stride per Matrix Size")
    print("=" * 70)
    print(f"{'Size':^12} {'Best Stride':^12} {'TFLOPS':^12} {'vs cuBLAS':^12}")
    print("-" * 70)

    for size in matrix_sizes:
        size_results = [r for r in results if r['m'] == size]
        if size_results:
            best = max(size_results, key=lambda x: x['tflops'])
            vs_cublas_str = f"{best['vs_cublas']:.1%}" if best['cublas_tflops'] else "N/A"
            print(f"{size:^12} {best['swizzle_stride']:^12} {best['tflops']:^12.2f} {vs_cublas_str:^12}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary for Paper")
    print("=" * 70)
    print(f"GPU: {gpu_info['name']}")
    print(f"L2 Cache Size: {gpu_info['l2_cache_mb']} MB")

    # Find overall optimal stride
    stride_performance = {}
    for r in results:
        s = r['swizzle_stride']
        if s not in stride_performance:
            stride_performance[s] = []
        stride_performance[s].append(r['tflops'])

    avg_by_stride = {s: sum(v)/len(v) for s, v in stride_performance.items()}
    best_stride = max(avg_by_stride, key=avg_by_stride.get)
    print(f"Best Average Swizzle Stride: {best_stride}")
    print(f"Average TFLOPS at Best Stride: {avg_by_stride[best_stride]:.2f}")

    if 0 in avg_by_stride:
        improvement = avg_by_stride[best_stride] / avg_by_stride[0]
        print(f"Improvement over No Swizzle: {improvement:.2f}x")

    # Compare to CUDA-L2 A100 numbers
    print("\n" + "-" * 70)
    print("Comparison to CUDA-L2 A100 Results (from paper)")
    print("-" * 70)
    print("A100 (40MB L2): Optimal stride = 1792")
    print(f"RTX 5090 ({gpu_info['l2_cache_mb']}MB L2): Optimal stride = {best_stride}")
    print(f"L2 ratio: {gpu_info['l2_cache_mb']}/40 = {gpu_info['l2_cache_mb']/40:.2f}x")
    print(f"Stride ratio: {best_stride}/1792 = {best_stride/1792:.2f}x")

    # NCU profiling if requested
    if args.ncu:
        print("\n" + "=" * 70)
        print("Nsight Compute Profiling")
        print("=" * 70)

        ncu_results_file = f"{args.output}_ncu_{timestamp}.csv"

        # Profile at 4096 size with different strides
        test_size = 4096
        print(f"Profiling {test_size}x{test_size}x{test_size} GEMM...")

        for stride in [0, 1792, 3072]:
            print(f"\nStride {stride}:")
            ncu_output = run_ncu_benchmark(test_size, test_size, test_size, stride)
            if ncu_output:
                print(ncu_output[:500])  # Print first 500 chars


if __name__ == "__main__":
    main()
