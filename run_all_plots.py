import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable or "python")


PLOT_TASKS = [
    # Backfill each args block from the best coarse-search row produced by
    # run_sensitivity.py before running this fine-grained sweep.
    {
        "dataset": "All_Beauty",
        "model": "USTv2_SASRec",
        "args": {
            "vocab_size": 128,
            "beta1": 0.05,
            "beta2": 0.0,
            "beta3": 0.05,
            "beta4": 0.05,
            "gumbel_tau": 1.0,
            "tau_min": 0.1,
        },
    },
    {
        "dataset": "All_Beauty",
        "model": "USTv2_GRU4Rec",
        "args": {
            "vocab_size": 256,
            "beta1": 0.01,
            "beta2": 0.0,
            "beta3": 0.05,
            "beta4": 0.1,
            "gumbel_tau": 1.0,
            "tau_min": 0.1,
        },
    },
    {
        "dataset": "Cross_Sport_Beauty",
        "model": "USTv2_SASRec",
        "args": {
            "vocab_size": 512,
            "beta1": 0.01,
            "beta2": 0.05,
            "beta3": 0.05,
            "beta4": 0.1,
            "gumbel_tau": 1.0,
            "tau_min": 0.2,
        },
    },
    {
        "dataset": "Cross_Sport_Beauty",
        "model": "USTv2_GRU4Rec",
        "args": {
            "vocab_size": 512,
            "beta1": 0.01,
            "beta2": 0.05,
            "beta3": 0.05,
            "beta4": 0.1,
            "gumbel_tau": 1.0,
            "tau_min": 0.2,
        },
    },
]


def dataset_exists(dataset_name):
    return os.path.isdir(os.path.join(DATA_ROOT, f"Processed_{dataset_name}"))


def run_command(command):
    print("[Exec]", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def build_plot_command(task):
    command = [
        PYTHON_BIN,
        os.path.join(PROJECT_ROOT, "run_plot_sensitivity.py"),
        "--dataset",
        task["dataset"],
        "--model",
        task["model"],
    ]
    for key, value in task["args"].items():
        command.extend([f"--{key}", str(value)])
    return command


def main():
    print("=================================================================")
    print("USTv2 Plot Runner")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data root:    {DATA_ROOT}")
    print(f"Python:       {PYTHON_BIN}")
    print("=================================================================")

    executed = 0
    for task in PLOT_TASKS:
        if not dataset_exists(task["dataset"]):
            print(f"[Skip] Missing dataset directory: Processed_{task['dataset']}")
            continue

        print("")
        print("-----------------------------------------------------------------")
        print(f"[Run] dataset={task['dataset']} model={task['model']}")
        print("-----------------------------------------------------------------")
        run_command(build_plot_command(task))
        executed += 1

    print("")
    print("=================================================================")
    if executed == 0:
        print("No plotting tasks were executed because no matching datasets were found.")
    else:
        print(f"Finished {executed} plotting task(s). Building comparison reports...")
        run_command([PYTHON_BIN, os.path.join(PROJECT_ROOT, "build_ust_comparison_reports.py")])
    print("Results directory: results_sensitivity_plot")
    print("Compare directory: results_compare")
    print("=================================================================")


if __name__ == "__main__":
    main()
