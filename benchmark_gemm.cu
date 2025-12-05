
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

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
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
        printf("Usage: %s M N K swizzle_stride [use_swizzle]\n", argv[0]);
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
    printf("{\"m\": %d, \"n\": %d, \"k\": %d, \"swizzle_stride\": %d, \"use_swizzle\": %d, \"time_ms\": %.4f, \"tflops\": %.2f}\n",
           cfg.M, cfg.N, cfg.K, cfg.swizzle_stride, cfg.use_swizzle, ms, tflops);

    free(h_A);
    free(h_B);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
