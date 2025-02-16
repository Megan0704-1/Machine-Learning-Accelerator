import sqlite3
import argparse
import sys

# Import the ncu_report
ncu_python_path = "/packages/apps/spack/21/opt/spack/linux-rocky8-zen3/gcc-12.1.0/cuda-12.6.1-cf4xlcbcfpwchqwo5bktxyhjagryzcx6/nsight-compute-2024.3.1/extras/python/"
sys.path.append(ncu_python_path)
import ncu_report


def parseArgs():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0], description="Parse ncu_rep generator by the ncu."
    )
    parser.add_argument(
        "--ncu_rep",
        dest="ncu_rep_file",
        default=False,
        # type=text,
        help="Pass the ncu_rep",
    )

    args = parser.parse_args()

    return args


def main():
    args = parseArgs()

    ncu_rep = args.ncu_rep_file
    report = ncu_report.load_report(ncu_rep)
    # Define the metric which you want to extract from the ncu-rep file
    metric_to_extract = [
        "sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.sum",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
    ]

    raw_metrics = {}
    for range_idx in range(report.num_ranges()):
        current_range = report.range_by_idx(range_idx)
        for action_idx in range(current_range.num_actions()):
            action = current_range.action_by_idx(action_idx)
            # print(action.name())
            raw_metrics[action_idx] = {}
            for metric in metric_to_extract:
                # print the metric value. Use for debug
                # print(
                #     f"Metric {metric} value {action.metric_by_name(metric).as_uint64()}"
                # )

                raw_metrics[action_idx][metric] = action.metric_by_name(
                    metric
                ).as_uint64()

    # print(raw_metrics)
    # print fo
    print(
        "Total operations in {} : {} ".format(
            "sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.sum",
            sum(
                kernel.get(
                    "sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.sum"
                )
                for kernel in raw_metrics.values()
            ),
        )
    )

    print(
        "Total DRAM read bytes: {} ".format(
            sum(kernel.get("dram__bytes_read.sum") for kernel in raw_metrics.values())
        )
    )

    print(
        "Total DRAM write bytes: {} ".format(
            sum(kernel.get("dram__bytes_write.sum") for kernel in raw_metrics.values())
        )
    )


if __name__ == "__main__":
    main()
