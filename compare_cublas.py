#!/usr/bin/env python3
"""
Compare custom GEMM implementations against cuBLAS baseline.

This script benchmarks torch.matmul (which uses cuBLAS internally)
to establish a baseline for comparison with the CUDA examples.

Usage:
    python compare_cublas.py
    python compare_cublas.py --size 2048
    python compare_cublas.py --all-sizes
"""

import argparse
import subprocess
import sys

try:
    import torch
except ImportError:
    print("PyTorch not found. Install with: pip install torch")
    sys.exit(1)


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


def get_gpu_info():
    """Get GPU name and compute capability"""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        return name, f"{cap[0]}.{cap[1]}"
    return "Unknown", "Unknown"


def run_example(example_name):
    """Run a compiled CUDA example and parse its output"""
    try:
        result = subprocess.run(
            [f"./{example_name}"],
            capture_output=True,
            text=True,
            timeout=60
        )
        # Extract TFLOPS from output
        for line in result.stdout.split('\n'):
            if 'TFLOPS' in line and 'Performance:' in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == 'TFLOPS':
                        return float(parts[i-1])
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def main():
    parser = argparse.ArgumentParser(description='Compare GEMM implementations')
    parser.add_argument('--size', type=int, default=1024,
                        help='Matrix size (MxNxK will all be this value)')
    parser.add_argument('--all-sizes', action='store_true',
                        help='Test multiple sizes')
    parser.add_argument('--compare-examples', action='store_true',
                        help='Also run and compare CUDA examples')
    args = parser.parse_args()

    gpu_name, compute_cap = get_gpu_info()

    print("=" * 60)
    print("GEMM Performance Comparison")
    print("=" * 60)
    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: {compute_cap}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    if args.all_sizes:
        sizes = [
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
        ]
    else:
        sizes = [(args.size, args.size, args.size)]

    # cuBLAS baseline
    print("cuBLAS Baseline (torch.matmul)")
    print("-" * 60)
    print(f"{'Size':^20} {'Time (ms)':^15} {'TFLOPS':^15}")
    print("-" * 60)

    cublas_results = {}
    for m, n, k in sizes:
        try:
            ms, tflops = benchmark_cublas(m, n, k)
            cublas_results[(m, n, k)] = tflops
            print(f"{m}x{n}x{k:^10} {ms:^15.3f} {tflops:^15.2f}")
        except RuntimeError as e:
            print(f"{m}x{n}x{k:^10} {'OOM':^15} {'N/A':^15}")

    print()

    # Compare with examples if requested
    if args.compare_examples:
        print("Custom CUDA Examples (1024x1024x1024)")
        print("-" * 60)
        print(f"{'Example':^30} {'TFLOPS':^15} {'vs cuBLAS':^15}")
        print("-" * 60)

        examples = [
            ("01_naive_gemm", "01 Naive"),
            ("02_tiled_gemm", "02 Tiled"),
            ("03_double_buffered_gemm", "03 Double-buffer"),
            ("04_wmma_tensor_core_gemm", "04 WMMA"),
            ("05_swizzled_gemm", "05 Swizzled"),
            ("07_async_copy_gemm", "07 Async"),
        ]

        baseline = cublas_results.get((1024, 1024, 1024), 100)

        for exe, name in examples:
            tflops = run_example(exe)
            if tflops:
                ratio = tflops / baseline
                print(f"{name:^30} {tflops:^15.2f} {ratio:^15.2%}")
            else:
                print(f"{name:^30} {'N/A':^15} {'N/A':^15}")

    print()
    print("=" * 60)
    print("Notes:")
    print("- cuBLAS is highly optimized by NVIDIA")
    print("- CUDA-L2 claims 1.17-1.23x speedup over cuBLAS on A100")
    print("- Our examples are educational, not production-optimized")
    print("=" * 60)


if __name__ == "__main__":
    main()
