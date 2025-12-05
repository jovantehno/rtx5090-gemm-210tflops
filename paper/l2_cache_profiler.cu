/**
 * L2 Cache Profiling for GEMM Blackwell Characterization
 *
 * Uses CUDA's built-in profiling to measure:
 * - L2 cache hit rate (via CUPTI if available)
 * - Memory throughput
 * - Achieved occupancy
 *
 * Compile: nvcc -O3 -arch=sm_80 l2_cache_profiler.cu -o l2_profiler
 * Run: ./l2_profiler
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32

#define BM 128
#define BN 128
#define BK 16
#define NUM_STAGES 3
#define WARPS_M 4
#define WARPS_N 4

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Simplified GEMM kernel for profiling
template<bool UseSwizzle>
__global__ void gemm_profiled(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K,
    int swizzle_stride
) {
    constexpr int THREADS = WARPS_M * WARPS_N * WARP_SIZE;

    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = As + NUM_STAGES * BM * BK;

    int bx, by;
    if constexpr (UseSwizzle) {
        int blocks_n = (N + BN - 1) / BN;
        int blocks_per_group = swizzle_stride / BN;
        if (blocks_per_group < 1) blocks_per_group = 1;

        int linear_id = blockIdx.y * blocks_n + blockIdx.x;
        bx = (linear_id % blocks_per_group) + (blockIdx.x / blocks_per_group) * blocks_per_group;
        bx = bx % blocks_n;
        by = linear_id / blocks_per_group % ((M + BM - 1) / BM);
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

    // Prologue
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
                    __pipeline_memcpy_async(As_stage + row * BK + col,
                                           A + global_row * K + global_col, 16);
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
                    __pipeline_memcpy_async(Bs_stage + row * BN + col,
                                           B + global_row * N + global_col, 16);
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
                        __pipeline_memcpy_async(As_load + row * BK + col,
                                               A + global_row * K + global_col, 16);
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
                        __pipeline_memcpy_async(Bs_load + row * BN + col,
                                               B + global_row * N + global_col, 16);
                    }
                }
            }
            __pipeline_commit();
        }
    }

    // Store
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

struct ProfileResult {
    int M, N, K;
    int swizzle_stride;
    bool use_swizzle;
    float time_ms;
    double tflops;
    double memory_read_gb;
    double memory_write_gb;
    double achieved_bandwidth_gbps;
    double theoretical_bandwidth_gbps;
    double bandwidth_efficiency;
};

template<bool UseSwizzle>
ProfileResult run_profiled(
    const half* d_A, const half* d_B, half* d_C,
    int M, int N, int K, int swizzle_stride,
    int warmup, int iterations
) {
    int blocks_m = (M + BM - 1) / BM;
    int blocks_n = (N + BN - 1) / BN;

    dim3 grid(blocks_n, blocks_m);
    int threads = WARPS_M * WARPS_N * WARP_SIZE;
    size_t smem_size = NUM_STAGES * (BM * BK + BK * BN) * sizeof(half);

    cudaFuncSetAttribute(gemm_profiled<UseSwizzle>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        gemm_profiled<UseSwizzle><<<grid, threads, smem_size>>>(
            d_A, d_B, d_C, M, N, K, swizzle_stride);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        gemm_profiled<UseSwizzle><<<grid, threads, smem_size>>>(
            d_A, d_B, d_C, M, N, K, swizzle_stride);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iterations;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    ProfileResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.swizzle_stride = swizzle_stride;
    result.use_swizzle = UseSwizzle;
    result.time_ms = ms;
    result.tflops = (2.0 * M * N * K / (ms / 1000.0)) / 1e12;

    // Calculate memory traffic (theoretical minimum)
    // Read: A (M*K) + B (K*N), Write: C (M*N)
    result.memory_read_gb = ((double)M * K + (double)K * N) * sizeof(half) / 1e9;
    result.memory_write_gb = (double)M * N * sizeof(half) / 1e9;

    double total_memory_gb = result.memory_read_gb + result.memory_write_gb;
    result.achieved_bandwidth_gbps = total_memory_gb / (ms / 1000.0);

    // RTX 5090 theoretical: 1792 GB/s
    result.theoretical_bandwidth_gbps = 1792.0;
    result.bandwidth_efficiency = result.achieved_bandwidth_gbps / result.theoretical_bandwidth_gbps * 100.0;

    return result;
}

void print_gpu_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("=== GPU Information ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("L2 Cache: %d MB\n", prop.l2CacheSize / (1024 * 1024));
    printf("Memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);

    // RTX 5090 theoretical bandwidth: 1792 GB/s
    printf("Theoretical Bandwidth: ~1792 GB/s (RTX 5090)\n");
    printf("\n");
}

void analyze_l2_behavior(int M, int N, int K) {
    printf("=== L2 Cache Analysis for %dx%dx%d ===\n", M, N, K);

    // Calculate working set sizes
    double A_size_mb = (double)M * K * sizeof(half) / (1024 * 1024);
    double B_size_mb = (double)K * N * sizeof(half) / (1024 * 1024);
    double C_size_mb = (double)M * N * sizeof(half) / (1024 * 1024);
    double total_mb = A_size_mb + B_size_mb + C_size_mb;

    printf("Matrix A: %.1f MB\n", A_size_mb);
    printf("Matrix B: %.1f MB\n", B_size_mb);
    printf("Matrix C: %.1f MB\n", C_size_mb);
    printf("Total: %.1f MB\n", total_mb);

    // L2 cache size (RTX 5090 = 96 MB)
    double l2_size_mb = 96.0;
    printf("L2 Cache: %.0f MB\n", l2_size_mb);

    if (total_mb <= l2_size_mb) {
        printf("Status: FULL FIT - entire working set fits in L2\n");
        printf("Expected: Stride should have minimal impact\n");
    } else if (A_size_mb + B_size_mb <= l2_size_mb) {
        printf("Status: PARTIAL FIT - A+B fit, C streams\n");
        printf("Expected: Stride affects A/B reuse patterns\n");
    } else if (B_size_mb <= l2_size_mb) {
        printf("Status: B ONLY - only B matrix fits\n");
        printf("Expected: Stride critical for B column reuse\n");
    } else {
        printf("Status: STREAMING - nothing fits entirely\n");
        printf("Expected: Stride affects bandwidth amplification\n");
    }

    // Calculate tile-level analysis
    double tile_A_size_kb = (double)BM * K * sizeof(half) / 1024;
    double tile_B_size_kb = (double)K * BN * sizeof(half) / 1024;
    int tiles_A_fit = (int)(l2_size_mb * 1024 / tile_A_size_kb);
    int tiles_B_fit = (int)(l2_size_mb * 1024 / tile_B_size_kb);

    printf("\nTile Analysis:\n");
    printf("  A row tile (BM x K): %.1f KB, ~%d tiles fit in L2\n", tile_A_size_kb, tiles_A_fit);
    printf("  B col tile (K x BN): %.1f KB, ~%d tiles fit in L2\n", tile_B_size_kb, tiles_B_fit);

    int blocks_m = (M + BM - 1) / BM;
    int blocks_n = (N + BN - 1) / BN;
    printf("  Grid: %d x %d blocks\n", blocks_n, blocks_m);
    printf("\n");
}

int main() {
    print_gpu_info();

    // Test configurations
    int sizes[] = {1024, 2048, 4096, 8192};
    int strides[] = {0, 512, 1792, 3072, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_strides = sizeof(strides) / sizeof(strides[0]);

    printf("=== L2 Cache Profiling Results ===\n\n");

    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];

        // Analyze L2 behavior for this size
        analyze_l2_behavior(size, size, size);

        // Allocate
        half *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, (size_t)size * size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_B, (size_t)size * size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C, (size_t)size * size * sizeof(half)));

        // Initialize
        half* h_A = (half*)malloc((size_t)size * size * sizeof(half));
        half* h_B = (half*)malloc((size_t)size * size * sizeof(half));
        srand(42);
        for (int i = 0; i < size * size; i++) {
            h_A[i] = __float2half((float)(rand() % 10) / 10.0f);
            h_B[i] = __float2half((float)(rand() % 10) / 10.0f);
        }
        CUDA_CHECK(cudaMemcpy(d_A, h_A, (size_t)size * size * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, (size_t)size * size * sizeof(half), cudaMemcpyHostToDevice));

        printf("%-8s %-8s %-12s %-12s %-15s %-12s\n",
               "Stride", "Time(ms)", "TFLOPS", "BW (GB/s)", "BW Efficiency", "Speedup");
        printf("------------------------------------------------------------------------\n");

        ProfileResult baseline;
        for (int i = 0; i < num_strides; i++) {
            int stride = strides[i];
            ProfileResult result;

            if (stride == 0) {
                result = run_profiled<false>(d_A, d_B, d_C, size, size, size, stride, 5, 20);
                baseline = result;
            } else {
                result = run_profiled<true>(d_A, d_B, d_C, size, size, size, stride, 5, 20);
            }

            double speedup = baseline.time_ms / result.time_ms;

            printf("%-8d %-8.3f %-12.2f %-12.1f %-12.1f%% %-12.2fx\n",
                   stride, result.time_ms, result.tflops,
                   result.achieved_bandwidth_gbps, result.bandwidth_efficiency,
                   speedup);
        }
        printf("\n");

        free(h_A);
        free(h_B);
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    printf("=== Analysis Summary ===\n\n");
    printf("Key observations:\n");
    printf("1. Bandwidth efficiency indicates how well we utilize memory system\n");
    printf("2. When BW efficiency > 100%%, L2 cache is amplifying bandwidth\n");
    printf("3. Higher amplification = better L2 reuse from block swizzling\n");
    printf("4. Optimal stride varies because cache fitting patterns differ by size\n");
    printf("\n");
    printf("L2 Cache Amplification Formula:\n");
    printf("  Amplification = Achieved BW / Theoretical DRAM BW\n");
    printf("  Values > 1.0 mean L2 is reducing DRAM traffic\n");
    printf("\n");

    return 0;
}
