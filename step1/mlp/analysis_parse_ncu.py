import sys
import argparse
from collections import defaultdict

# Add NVIDIA Nsight Compute Python API path
ncu_python_path = "/packages/apps/spack/21/opt/spack/linux-rocky8-zen3/gcc-12.1.0/cuda-12.6.1-cf4xlcbcfpwchqwo5bktxyhjagryzcx6/nsight-compute-2024.3.1/extras/python/"
sys.path.append(ncu_python_path)
import ncu_report

def parseArgs():
    parser = argparse.ArgumentParser(
        description="Parse NCU report and analyze all metrics"
    )
    parser.add_argument(
        "--ncu_rep",
        dest="ncu_rep_file",
        required=True,
        help="Path to .ncu-rep file"
    )
    return parser.parse_args()

def get_all_metrics(report):
    """Extract all unique metric names from the report"""
    metric_set = set()
    for range_idx in range(report.num_ranges()):
        current_range = report.range_by_idx(range_idx)
        for action_idx in range(current_range.num_actions()):
            action = current_range.action_by_idx(action_idx)
            metric_set.update(action.metric_names())
    return sorted(metric_set)

def analyze_metrics(report):
    """Collect metrics across all kernels and categorize them"""
    metrics_sum = defaultdict(lambda: 0)
    metric_categories = {
        'dram': ['dram__'],
        'l1/l2_cache': ['l1tex__', 'lts__'],
        'tensor_ops': ['sm__ops_path_tensor'],
        'flops': ['sm__sass_thread_inst_executed_op', 'smsp__sass_thread_inst_executed_op'],
        'memory': ['bytes', 't_bytes'],
        'cycles': ['cycles_elapsed']
    }

    # First pass: Collect all metrics
    for range_idx in range(report.num_ranges()):
        current_range = report.range_by_idx(range_idx)
        for action_idx in range(current_range.num_actions()):
            action = current_range.action_by_idx(action_idx)
            for metric_name in action.metric_names():
                try:
                    value = action.metric_by_name(metric_name).as_uint64()
                    metrics_sum[metric_name] += value
                except Exception as e:
                    print(f"Error processing {metric_name}: {str(e)}")
                    continue

    # Categorize metrics
    categorized = defaultdict(dict)
    for metric, total in metrics_sum.items():
        found = False
        for cat, patterns in metric_categories.items():
            if any(pattern in metric for pattern in patterns):
                categorized[cat][metric] = total
                found = True
                break
        if not found:
            categorized['other'][metric] = total

    return categorized

def print_analysis(categorized_metrics):
    """Print formatted analysis with metric categories"""
    for category, metrics in categorized_metrics.items():
        print(f"\n=== {category.upper()} METRICS ===")
        for metric, value in metrics.items():
            unit = "FLOPs" if "inst_executed" in metric else "bytes" if "bytes" in metric else "cycles" if "cycles" in metric else "count"
            print(f"{metric:<80} {value:>15} {unit}")

def generate_latex_table(categorized_metrics):
    """Generate LaTeX table from categorized metrics"""
    latex = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{NCU Metrics Summary}",
        "\\label{tab:ncu_metrics}",
        "\\begin{tabular}{|l|r|r|}",
        "\\hline",
        "\\textbf{Metric Category} & \\textbf{Metric Name} & \\textbf{Value} \\\\",
        "\\hline"
    ]

    for category, metrics in categorized_metrics.items():
        for metric, value in metrics.items():
            latex.append(f"{category} & \\texttt{{{metric}}} & {value:,} \\\\")
        latex.append("\\hline")

    latex.extend([
        "\\end{tabular}",
        "\\end{table}"
    ])

    return '\n'.join(latex)

def main():
    args = parseArgs()
    report = ncu_report.load_report(args.ncu_rep_file)

    # Analyze and categorize metrics
    categorized_metrics = analyze_metrics(report)

    # Print human-readable analysis
    print_analysis(categorized_metrics)

    # Generate LaTeX table
    print("\n\\nLaTeX Table Output:")
    print(generate_latex_table(categorized_metrics))

if __name__ == "__main__":
    main()
