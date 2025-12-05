#!/usr/bin/env python3
"""
Visualization script for Blackwell GEMM characterization paper.

Generates publication-quality figures from benchmark data.

Usage:
    python visualize.py benchmark_results_*.csv
    python visualize.py --all  # Use most recent results file
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
except ImportError:
    print("Required packages not found. Install with:")
    print("  pip install matplotlib seaborn pandas")
    sys.exit(1)

# Publication-quality settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def load_data(csv_path):
    """Load benchmark results CSV"""
    df = pd.read_csv(csv_path)
    return df


def fig1_optimization_progression(output_dir):
    """
    Figure 1: Bar chart showing TFLOPS progression through examples 01-09
    """
    # Data from our experiments (2000-iteration benchmarks)
    examples = {
        '01\nNaive': 6.97,
        '02\nTiled': 6.15,
        '03\nDouble\nBuffer': 16.64,
        '04\nTensor\nCores': 11.45,
        '05\nSwizzled': 12.51,
        '06\nL2\nOptim': 71.38,
        '07\nAsync': 95.01,
        '08\nCombined': 215,
        '09\nPTX\nOptim': 380,
    }
    cublas = 354  # cublasHgemm, 2000 iterations

    fig, ax = plt.subplots(figsize=(11, 5))

    names = list(examples.keys())
    values = list(examples.values())
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(names)))

    bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=0.5)

    # Highlight the best one (Example 09)
    bars[-1].set_color('#2ecc71')
    bars[-1].set_edgecolor('black')

    # Add cuBLAS reference line
    ax.axhline(y=cublas, color='#e74c3c', linestyle='--', linewidth=2, label=f'cuBLAS ({cublas} TFLOPS)')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('TFLOPS')
    ax.set_xlabel('Optimization Stage')
    ax.set_title('GEMM Optimization Progression on RTX 5090 (4096³ FP16, 2000 iterations)')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 420)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_optimization_progression.png')
    plt.savefig(output_dir / 'fig1_optimization_progression.pdf')
    plt.close()
    print("Generated: fig1_optimization_progression.png/pdf")


def fig2_stride_heatmap(df, output_dir):
    """
    Figure 2: Heatmap of TFLOPS across matrix sizes and swizzle strides
    """
    # Pivot the data
    pivot = df.pivot_table(values='tflops', index='m', columns='swizzle_stride', aggfunc='mean')

    # Sort index and columns
    pivot = pivot.sort_index()
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Custom colormap
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    sns.heatmap(pivot, annot=True, fmt='.0f', cmap=cmap, ax=ax,
                cbar_kws={'label': 'TFLOPS'},
                linewidths=0.5, linecolor='white')

    # Mark optimal stride for each size
    for i, size in enumerate(pivot.index):
        row = pivot.loc[size]
        best_stride = row.idxmax()
        j = list(pivot.columns).index(best_stride)
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='blue', linewidth=2))

    ax.set_xlabel('Swizzle Stride')
    ax.set_ylabel('Matrix Size (M=N=K)')
    ax.set_title('GEMM Performance Heatmap: Matrix Size vs Swizzle Stride\n(Blue boxes = optimal stride per size)')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_stride_heatmap.png')
    plt.savefig(output_dir / 'fig2_stride_heatmap.pdf')
    plt.close()
    print("Generated: fig2_stride_heatmap.png/pdf")


def fig3_tflops_vs_size(df, output_dir):
    """
    Figure 3: Line plot of TFLOPS vs matrix size for different strides
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Select interesting strides to show
    key_strides = [0, 1792, 2048, 3072, 4096]
    colors = plt.cm.tab10(np.linspace(0, 1, len(key_strides)))

    for stride, color in zip(key_strides, colors):
        subset = df[df['swizzle_stride'] == stride].sort_values('m')
        label = 'No swizzle' if stride == 0 else f'Stride {stride}'
        ax.plot(subset['m'], subset['tflops'], 'o-', color=color, label=label, linewidth=2, markersize=6)

    # Add cuBLAS line if available
    if 'cublas_tflops' in df.columns:
        cublas = df.groupby('m')['cublas_tflops'].first().sort_index()
        ax.plot(cublas.index, cublas.values, 's--', color='red', label='cuBLAS', linewidth=2, markersize=6)

    ax.set_xlabel('Matrix Size (M=N=K)')
    ax.set_ylabel('TFLOPS')
    ax.set_title('GEMM Performance vs Matrix Size (RTX 5090)')
    ax.legend(loc='lower right')
    ax.set_xscale('log', base=2)

    # Format x-axis
    sizes = sorted(df['m'].unique())
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_tflops_vs_size.png')
    plt.savefig(output_dir / 'fig3_tflops_vs_size.pdf')
    plt.close()
    print("Generated: fig3_tflops_vs_size.png/pdf")


def fig4_vs_cublas(df, output_dir):
    """
    Figure 4: Performance relative to cuBLAS across sizes
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get best performance per size
    best_per_size = df.loc[df.groupby('m')['tflops'].idxmax()]

    # Also get no-swizzle performance
    no_swizzle = df[df['swizzle_stride'] == 0].sort_values('m')

    sizes = best_per_size['m'].values
    best_ratio = best_per_size['vs_cublas'].values * 100
    no_swizzle_ratio = no_swizzle['vs_cublas'].values * 100

    x = np.arange(len(sizes))
    width = 0.35

    bars1 = ax.bar(x - width/2, no_swizzle_ratio, width, label='No Swizzle', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, best_ratio, width, label='Best Stride', color='#2ecc71', edgecolor='black')

    # Add 100% reference line
    ax.axhline(y=100, color='#e74c3c', linestyle='--', linewidth=2, label='cuBLAS (100%)')

    ax.set_xlabel('Matrix Size (M=N=K)')
    ax.set_ylabel('Performance vs cuBLAS (%)')
    ax.set_title('Custom Kernel Performance Relative to cuBLAS')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, rotation=45)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 110)

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_vs_cublas.png')
    plt.savefig(output_dir / 'fig4_vs_cublas.pdf')
    plt.close()
    print("Generated: fig4_vs_cublas.png/pdf")


def fig5_optimal_stride_by_size(df, output_dir):
    """
    Figure 5: Optimal stride for each matrix size (key finding)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Get optimal stride per size
    best_per_size = df.loc[df.groupby('m')['tflops'].idxmax()].sort_values('m')

    sizes = best_per_size['m'].values
    optimal_strides = best_per_size['swizzle_stride'].values
    tflops = best_per_size['tflops'].values

    # Left plot: Optimal stride
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sizes)))
    bars = ax1.bar(range(len(sizes)), optimal_strides, color=colors, edgecolor='black')

    ax1.set_xlabel('Matrix Size (M=N=K)')
    ax1.set_ylabel('Optimal Swizzle Stride')
    ax1.set_title('Optimal Stride Varies by Matrix Size')
    ax1.set_xticks(range(len(sizes)))
    ax1.set_xticklabels(sizes, rotation=45)

    # Add A100 reference line
    ax1.axhline(y=1792, color='#e74c3c', linestyle='--', linewidth=2, label='A100 optimal (1792)')
    ax1.legend()

    # Add value labels
    for bar, stride in zip(bars, optimal_strides):
        height = bar.get_height()
        ax1.annotate(f'{int(stride)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # Right plot: TFLOPS at optimal stride
    ax2.bar(range(len(sizes)), tflops, color='#2ecc71', edgecolor='black')
    ax2.set_xlabel('Matrix Size (M=N=K)')
    ax2.set_ylabel('TFLOPS')
    ax2.set_title('Peak Performance at Optimal Stride')
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels(sizes, rotation=45)

    for i, (s, t) in enumerate(zip(sizes, tflops)):
        ax2.annotate(f'{t:.0f}',
                    xy=(i, t),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_optimal_stride_by_size.png')
    plt.savefig(output_dir / 'fig5_optimal_stride_by_size.pdf')
    plt.close()
    print("Generated: fig5_optimal_stride_by_size.png/pdf")


def fig6_stride_sensitivity(df, output_dir):
    """
    Figure 6: How sensitive is performance to stride choice at different sizes?
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    sizes = sorted(df['m'].unique())

    # Calculate sensitivity: (max - min) / max for each size
    sensitivity = []
    for size in sizes:
        subset = df[df['m'] == size]
        max_tflops = subset['tflops'].max()
        min_tflops = subset['tflops'].min()
        sens = (max_tflops - min_tflops) / max_tflops * 100
        sensitivity.append(sens)

    colors = plt.cm.RdYlGn_r(np.array(sensitivity) / max(sensitivity))
    bars = ax.bar(range(len(sizes)), sensitivity, color=colors, edgecolor='black')

    ax.set_xlabel('Matrix Size (M=N=K)')
    ax.set_ylabel('Performance Sensitivity (%)')
    ax.set_title('Sensitivity to Stride Choice\n(Higher = more important to choose correct stride)')
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes, rotation=45)

    for bar, sens in zip(bars, sensitivity):
        height = bar.get_height()
        ax.annotate(f'{sens:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_stride_sensitivity.png')
    plt.savefig(output_dir / 'fig6_stride_sensitivity.pdf')
    plt.close()
    print("Generated: fig6_stride_sensitivity.png/pdf")


def fig7_comparison_table(df, output_dir):
    """
    Figure 7: Create a comparison table as an image (for paper)
    """
    # Get optimal per size
    best = df.loc[df.groupby('m')['tflops'].idxmax()].sort_values('m')

    # A100 reference data (from CUDA-L2 paper)
    a100_optimal_stride = 1792  # Consistent for A100

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    # Table data
    table_data = [
        ['Matrix Size', 'RTX 5090\nOptimal Stride', 'RTX 5090\nTFLOPS', 'RTX 5090\nvs cuBLAS', 'A100\nOptimal Stride'],
    ]

    for _, row in best.iterrows():
        table_data.append([
            f"{int(row['m'])}³",
            str(int(row['swizzle_stride'])),
            f"{row['tflops']:.1f}",
            f"{row['vs_cublas']*100:.1f}%",
            str(a100_optimal_stride)
        ])

    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    ax.set_title('RTX 5090 (Blackwell) vs A100 (Ampere) Optimal Parameters',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_comparison_table.png')
    plt.savefig(output_dir / 'fig7_comparison_table.pdf')
    plt.close()
    print("Generated: fig7_comparison_table.png/pdf")


def fig8_l2_scaling_hypothesis(output_dir):
    """
    Figure 8: Illustrate the L2 cache scaling hypothesis
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: L2 cache size comparison
    gpus = ['A100', 'H100', 'RTX 5090']
    l2_sizes = [40, 50, 96]
    colors = ['#3498db', '#9b59b6', '#2ecc71']

    ax1.barh(gpus, l2_sizes, color=colors, edgecolor='black')
    ax1.set_xlabel('L2 Cache Size (MB)')
    ax1.set_title('L2 Cache Size by GPU')

    for i, (gpu, size) in enumerate(zip(gpus, l2_sizes)):
        ax1.annotate(f'{size} MB', xy=(size + 2, i), va='center', fontweight='bold')

    ax1.set_xlim(0, 110)

    # Right: Conceptual diagram of tile fitting
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('L2 Cache Tile Fitting (Conceptual)')

    # A100: fewer tiles fit
    ax2.add_patch(plt.Rectangle((0.5, 5.5), 4, 4, fill=True, facecolor='#3498db', edgecolor='black', alpha=0.7))
    ax2.text(2.5, 7.5, 'A100\n40 MB\n~40 tiles', ha='center', va='center', fontsize=10, fontweight='bold')

    # RTX 5090: more tiles fit
    ax2.add_patch(plt.Rectangle((5.5, 5.5), 4, 4, fill=True, facecolor='#2ecc71', edgecolor='black', alpha=0.7))
    ax2.text(7.5, 7.5, 'RTX 5090\n96 MB\n~96 tiles', ha='center', va='center', fontsize=10, fontweight='bold')

    # Annotations
    ax2.annotate('Consistent\nstride works', xy=(2.5, 5.2), ha='center', fontsize=9)
    ax2.annotate('Stride must\nadapt to size', xy=(7.5, 5.2), ha='center', fontsize=9)

    # Add caption box
    ax2.text(5, 1.5,
             'Hypothesis: Larger L2 creates more complex\ninteraction between block scheduling and cache,\nrequiring size-dependent stride tuning.',
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig8_l2_scaling_hypothesis.png')
    plt.savefig(output_dir / 'fig8_l2_scaling_hypothesis.pdf')
    plt.close()
    print("Generated: fig8_l2_scaling_hypothesis.png/pdf")


def generate_all_figures(df, output_dir):
    """Generate all figures"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\nGenerating figures in {output_dir}/\n")
    print("=" * 50)

    fig1_optimization_progression(output_dir)
    fig2_stride_heatmap(df, output_dir)
    fig3_tflops_vs_size(df, output_dir)
    fig4_vs_cublas(df, output_dir)
    fig5_optimal_stride_by_size(df, output_dir)
    fig6_stride_sensitivity(df, output_dir)
    fig7_comparison_table(df, output_dir)
    fig8_l2_scaling_hypothesis(output_dir)

    print("=" * 50)
    print(f"\nAll figures saved to {output_dir}/")
    print("\nFigures for paper:")
    print("  - fig1: Optimization progression (intro)")
    print("  - fig2: Heatmap (main result)")
    print("  - fig3: TFLOPS vs size (performance scaling)")
    print("  - fig4: vs cuBLAS comparison")
    print("  - fig5: Optimal stride by size (key finding)")
    print("  - fig6: Stride sensitivity (implications)")
    print("  - fig7: Comparison table")
    print("  - fig8: L2 scaling hypothesis (discussion)")


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('csv_file', nargs='?', help='Benchmark results CSV file')
    parser.add_argument('--all', action='store_true', help='Use most recent results file')
    parser.add_argument('--output', type=str, default='figures', help='Output directory')
    args = parser.parse_args()

    # Find CSV file
    if args.csv_file:
        csv_path = args.csv_file
    elif args.all:
        csv_files = sorted(glob.glob('../benchmark_results_*.csv'))
        if not csv_files:
            csv_files = sorted(glob.glob('benchmark_results_*.csv'))
        if not csv_files:
            print("No benchmark_results_*.csv files found")
            sys.exit(1)
        csv_path = csv_files[-1]  # Most recent
        print(f"Using: {csv_path}")
    else:
        print("Usage: python visualize.py <csv_file> or python visualize.py --all")
        sys.exit(1)

    # Load data
    df = load_data(csv_path)
    print(f"Loaded {len(df)} data points")
    print(f"Matrix sizes: {sorted(df['m'].unique())}")
    print(f"Strides: {sorted(df['swizzle_stride'].unique())}")

    # Generate figures
    generate_all_figures(df, args.output)


if __name__ == "__main__":
    main()
