# Makefile for CUDA GEMM Learning Examples
#
# Usage:
#   make all      - Build all examples
#   make run      - Build and run all examples
#   make clean    - Remove compiled binaries
#   make example_01  - Build specific example
#
# Requirements:
#   - CUDA Toolkit 11.0+ (for SM80 support)
#   - GPU with compute capability 8.0+ (A100, RTX 30xx, etc.)
#
# For RTX 5090 (SM100), change ARCH to sm_100 when CUDA supports it

# CUDA compiler
NVCC = nvcc

# Target architecture
# SM80 = Ampere (A100, RTX 30xx)
# SM89 = Ada Lovelace (RTX 40xx)
# SM90 = Hopper (H100)
# SM100 = Blackwell (RTX 50xx) - future
ARCH ?= sm_80

# Compiler flags
NVCC_FLAGS = -O3 -std=c++17 -arch=$(ARCH)
NVCC_FLAGS += -Xcompiler -Wall
NVCC_FLAGS += --use_fast_math
NVCC_FLAGS += -lineinfo  # For profiling

# All examples
EXAMPLES = 01_naive_gemm \
           02_tiled_gemm \
           03_double_buffered_gemm \
           04_wmma_tensor_core_gemm \
           05_swizzled_gemm \
           06_block_swizzle_l2_gemm \
           07_async_copy_gemm \
           08_combined_optimized_gemm \
           09_ptx_optimized_gemm

# Build all
all: $(EXAMPLES)

# Individual examples
01_naive_gemm: 01_naive_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

02_tiled_gemm: 02_tiled_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

03_double_buffered_gemm: 03_double_buffered_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

04_wmma_tensor_core_gemm: 04_wmma_tensor_core_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

05_swizzled_gemm: 05_swizzled_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

06_block_swizzle_l2_gemm: 06_block_swizzle_l2_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

07_async_copy_gemm: 07_async_copy_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

08_combined_optimized_gemm: 08_combined_optimized_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

09_ptx_optimized_gemm: 09_ptx_optimized_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Run all examples
run: all
	@echo "=========================================="
	@echo "Running all GEMM examples"
	@echo "=========================================="
	@for example in $(EXAMPLES); do \
		echo ""; \
		echo "------------------------------------------"; \
		./$$example; \
	done

# Run individual examples
run_%: %
	./$<

# Profile with Nsight Compute (if available)
profile_%: %
	ncu --set full -o $<_profile ./$<

# Clean
clean:
	rm -f $(EXAMPLES) *.o *.ncu-rep

# Compare against cuBLAS baseline
compare: all
	@echo "=========================================="
	@echo "Comparing against cuBLAS baseline"
	@echo "=========================================="
	python3 compare_cublas.py --all-sizes

# Full comparison including examples
compare-all: all
	@echo "=========================================="
	@echo "Full comparison with examples"
	@echo "=========================================="
	python3 compare_cublas.py --all-sizes --compare-examples

# Help
help:
	@echo "CUDA GEMM Learning Examples"
	@echo ""
	@echo "Examples (in order of complexity):"
	@echo "  01_naive_gemm           - Baseline implementation"
	@echo "  02_tiled_gemm           - Shared memory tiling"
	@echo "  03_double_buffered_gemm - Software pipelining"
	@echo "  04_wmma_tensor_core_gemm - Tensor cores via WMMA"
	@echo "  05_swizzled_gemm        - Bank conflict elimination"
	@echo "  06_block_swizzle_l2_gemm - L2 cache optimization"
	@echo "  07_async_copy_gemm      - Async memory pipeline"
	@echo "  08_combined_optimized   - All optimizations combined"
	@echo "  09_ptx_optimized_gemm   - PTX-level optimizations"
	@echo ""
	@echo "Targets:"
	@echo "  make all         - Build all examples"
	@echo "  make run         - Build and run all"
	@echo "  make compare     - Compare against cuBLAS baseline"
	@echo "  make compare-all - Compare examples + cuBLAS"
	@echo "  make <example>   - Build specific example"
	@echo "  make run_<ex>    - Run specific example"
	@echo "  make profile_<ex>- Profile with Nsight Compute"
	@echo "  make clean       - Remove binaries"
	@echo ""
	@echo "Variables:"
	@echo "  ARCH=sm_XX       - Target GPU architecture (default: sm_80)"
	@echo ""
	@echo "Documentation:"
	@echo "  README.md         - Example descriptions"
	@echo "  INSTRUCTIONS.md   - How to test and compare"
	@echo "  claudeAnalysis.md - CUDA-L2 repo analysis"
	@echo "  claude-5090.md    - RTX 5090 adaptation guide"

.PHONY: all run clean help compare compare-all
