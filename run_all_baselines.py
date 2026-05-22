import gc
import argparse
import json
import os
import subprocess
import sys

import ust_bootstrap  # noqa: F401
import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender.bpr import BPR
from recbole.model.sequential_recommender.gru4rec import GRU4Rec
from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.utils import get_trainer, init_logger, init_seed

from USTv2_GRU4Rec import USTv2_GRU4Rec
from USTv2_SASRec import USTv2_SASRec
from ust_reporting import parse_model_metadata
from ust_utils import (
    apply_model_runtime_compat,
    cleanup_checkpoint_file,
    get_model_config_files,
    prepare_data_path_config,
    resolve_checkpoint_dir,
    resolve_config_file_list,
)


MODEL_DICT = {
    "SASRec": SASRec,
    "GRU4Rec": GRU4Rec,
    "BPR": BPR,
    "USTv2_SASRec": USTv2_SASRec,
    "USTv2_GRU4Rec": USTv2_GRU4Rec,
}

EXTERNAL_MODEL_SCRIPTS = {
    "VBPR": "run_external_vbpr.py",
    "BM3": "run_external_bm3.py",
    "UniSRec": "run_external_unisrec.py",
    "VQRec": "run_external_vqrec.py",
    "C2DSR": "run_external_c2dsr.py",
#    "SASRec_PTFT_10pct": "run_pretrain_finetune_baseline.py",
#    "SASRec_PTFT_20pct": "run_pretrain_finetune_baseline.py",
#    "GRU4Rec_PTFT_10pct": "run_pretrain_finetune_baseline.py",
#    "GRU4Rec_PTFT_20pct": "run_pretrain_finetune_baseline.py",
}

ACTIVE_RESULT_MODELS = set(MODEL_DICT) | set(EXTERNAL_MODEL_SCRIPTS)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

single_domain_models = [
    "SASRec",
    "GRU4Rec",
    "BPR",
    "VBPR",
    "BM3",
    "UniSRec",
    "VQRec",
    "USTv2_SASRec",
    "USTv2_GRU4Rec",
]

cross_domain_models = [
    "SASRec",
    "GRU4Rec",
#    "SASRec_PTFT_10pct",
#    "SASRec_PTFT_20pct",
#    "GRU4Rec_PTFT_10pct",
#    "GRU4Rec_PTFT_20pct",
    "UniSRec",
    "VQRec",
    "C2DSR",
    "USTv2_SASRec",
    "USTv2_GRU4Rec",
]

datasets = [
    "All_Beauty",
    "Electronics",
    "Sports_and_Outdoors",
#    "Tenrec_SBR",
    "Cross_Sport_Beauty",
    "Cross_Elec_Sport",
]

SINGLE_DOMAIN_DATASETS = [dataset_name for dataset_name in datasets if not dataset_name.startswith("Cross_")]
CROSS_DOMAIN_DATASETS = [dataset_name for dataset_name in datasets if dataset_name.startswith("Cross_")]

BEST_PARAMS_MAP = {
    "All_Beauty": {
        "USTv2_SASRec": {
            "vocab_size": 128,
            "beta1": 0.05,
            "beta2": 0.0,
            "beta3": 0.05,
            "beta4": 0.05,
            "gumbel_tau": 1.0,
            "tau_min": 0.1,
        },
        "USTv2_GRU4Rec": {
            "vocab_size": 64,
            "beta1": 0.1,
            "beta2": 0.0,
            "beta3": 0.05,
            "beta4": 0.05,
            "gumbel_tau": 1.0,
            "tau_min": 0.1,
        },
    },
    # Manual backfill target after run_plot_sensitivity.py / run_all_plots.py.
    # If a dataset-model pair is missing here, run_all_baselines.py will fall
    # back to DEFAULT_V2_PARAMS and print a warning with the exact map key.
    "Electronics": {
        "USTv2_SASRec": {
            "vocab_size": 128,
            "beta1": 0.05,
            "beta2": 0.0,
            "beta3": 0.05,
            "beta4": 0.05,
            "gumbel_tau": 1.0,
            "tau_min": 0.1,
        },
        "USTv2_GRU4Rec": {
            "vocab_size": 64,
            "beta1": 0.1,
            "beta2": 0.0,
            "beta3": 0.05,
            "beta4": 0.05,
            "gumbel_tau": 1.0,
            "tau_min": 0.1,
        },
    },
    "Sports_and_Outdoors": {
        "USTv2_SASRec": {
            "vocab_size": 128,
            "beta1": 0.05,
            "beta2": 0.0,
            "beta3": 0.05,
            "beta4": 0.05,
            "gumbel_tau": 1.0,
            "tau_min": 0.1,
        },
        "USTv2_GRU4Rec": {
            "vocab_size": 64,
            "beta1": 0.1,
            "beta2": 0.0,
            "beta3": 0.05,
            "beta4": 0.05,
            "gumbel_tau": 1.0,
            "tau_min": 0.1,
        },
    },
    "Tenrec_SBR": {},
    "Cross_Sport_Beauty": {
        "USTv2_SASRec": {
            "vocab_size": 512,
            "beta1": 0.01,
            "beta2": 0.05,
            "beta3": 0.05,
            "beta4": 0.1,
            "gumbel_tau": 1.0,
            "tau_min": 0.2,
        },
        "USTv2_GRU4Rec": {
            "vocab_size": 512,
            "beta1": 0.01,
            "beta2": 0.05,
            "beta3": 0.05,
            "beta4": 0.1,
            "gumbel_tau": 1.0,
            "tau_min": 0.2,
        },
    },
    "Cross_Elec_Sport": {
        "USTv2_SASRec": {
            "vocab_size": 512,
            "beta1": 0.01,
            "beta2": 0.05,
            "beta3": 0.05,
            "beta4": 0.1,
            "gumbel_tau": 1.0,
            "tau_min": 0.2,
        },
        "USTv2_GRU4Rec": {
            "vocab_size": 512,
            "beta1": 0.01,
            "beta2": 0.05,
            "beta3": 0.05,
            "beta4": 0.1,
            "gumbel_tau": 1.0,
            "tau_min": 0.2,
        },
    },
}

DEFAULT_V2_PARAMS = {
    "vocab_size": 256,
    "beta1": 0.01,
    "beta2": 0.0,
    "beta3": 0.05,
    "beta4": 0.1,
    "gumbel_tau": 1.0,
    "tau_min": 0.2,
    "mm_loss_weight": 0.1,
}

ABLATION_RESULT_DIR = os.path.join(PROJECT_ROOT, "results_ablation")
CONFIG_SNAPSHOT_KEYS = [
    "vocab_size",
    "beta1",
    "beta2",
    "beta3",
    "beta4",
    "gumbel_tau",
    "tau_min",
    "mm_loss_weight",
    "use_ust",
]


def is_cross_domain_dataset(dataset_name):
    return str(dataset_name).startswith("Cross_")


def resolve_external_script(model_name, dataset_name):
    if model_name == "UniSRec":
        return (
            "run_external_unisrec_transfer.py"
            if is_cross_domain_dataset(dataset_name)
            else "run_external_unisrec.py"
        )
    if model_name == "VQRec":
        return (
            "run_external_vqrec_transfer.py"
            if is_cross_domain_dataset(dataset_name)
            else "run_external_vqrec.py"
        )
    return EXTERNAL_MODEL_SCRIPTS[model_name]


def resolve_baseline_protocol(model_name, dataset_name):
    if model_name in {"UniSRec", "VQRec"}:
        if is_cross_domain_dataset(dataset_name):
            return "source_pretrain_target_finetune"
        return "scratch"
    if is_cross_domain_dataset(dataset_name) and model_name in {
        "SASRec",
        "GRU4Rec",
        "C2DSR",
        "USTv2_SASRec",
        "USTv2_GRU4Rec",
    }:
        return "cross_dataset_direct"
    return None


def resolve_cross_dataset_metadata(processed_dir, dataset_name):
    if not is_cross_domain_dataset(dataset_name):
        return {
            "Source_Dataset": dataset_name,
            "Target_Dataset": dataset_name,
        }
    meta_path = os.path.join(processed_dir, "ust_meta.json")
    source_dataset = dataset_name
    target_dataset = dataset_name
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as file_obj:
                meta_info = json.load(file_obj)
            source_dataset = str(meta_info.get("source_domain") or source_dataset)
            target_dataset = str(meta_info.get("target_domain") or target_dataset)
        except Exception:
            pass
    return {
        "Source_Dataset": source_dataset,
        "Target_Dataset": target_dataset,
    }


def values_close(left_value, right_value, tol=1e-12):
    try:
        return abs(float(left_value) - float(right_value)) <= tol
    except (TypeError, ValueError):
        return str(left_value) == str(right_value)


def load_ablation_table(project_root, dataset_name, backbone_name):
    ablation_path = os.path.join(
        project_root,
        os.path.basename(ABLATION_RESULT_DIR),
        f"Ablation_USTv2_{backbone_name}_{dataset_name}.xlsx",
    )
    if not os.path.exists(ablation_path):
        return pd.DataFrame(), ablation_path

    try:
        return pd.read_excel(ablation_path), ablation_path
    except Exception as exc:
        print(f"[Warn] Failed to load ablation cache from {ablation_path}: {exc}")
        return pd.DataFrame(), ablation_path


def match_ablation_record(ablation_df, model_name, dataset_name, target_config, variant_name):
    if ablation_df is None or ablation_df.empty:
        return None

    mask = pd.Series(True, index=ablation_df.index)
    if "Model" in ablation_df.columns:
        mask &= ablation_df["Model"].astype(str) == str(model_name)
    if "Dataset" in ablation_df.columns:
        mask &= ablation_df["Dataset"].astype(str) == str(dataset_name)
    if "Variant" in ablation_df.columns:
        mask &= ablation_df["Variant"].astype(str) == str(variant_name)
    if "Status" in ablation_df.columns:
        mask &= ablation_df["Status"].astype(str) == "success"

    compare_keys = [key for key in CONFIG_SNAPSHOT_KEYS if key in target_config]
    if model_name.startswith("USTv2_"):
        missing_columns = [key for key in compare_keys if key not in ablation_df.columns]
        if missing_columns:
            print(
                f"[Warn] Ablation cache for {model_name} lacks config columns "
                f"{missing_columns}. Skip reuse."
            )
            return None
    for config_key in compare_keys:
        if config_key not in ablation_df.columns:
            continue
        mask &= ablation_df[config_key].apply(
            lambda value: values_close(value, target_config[config_key])
        )

    matches = ablation_df[mask]
    if matches.empty:
        return None

    if "Variant" in matches.columns and variant_name:
        exact_matches = matches[matches["Variant"].astype(str) == str(variant_name)]
        if not exact_matches.empty:
            matches = exact_matches

    return matches.iloc[0].to_dict()


def build_reused_result_row(model_name, dataset_name, ablation_record, variant_name=None):
    family_name, backbone_name = parse_model_metadata(model_name)
    reused_row = {
        "Model": model_name,
        "Dataset": dataset_name,
        "Family": family_name,
        "Backbone": backbone_name,
        "Source": "ablation_reuse",
    }
    if variant_name is not None:
        reused_row["Reused_Variant"] = variant_name
    for metric_name in [
        "recall@5",
        "recall@10",
        "recall@20",
        "mrr@5",
        "mrr@10",
        "mrr@20",
        "ndcg@5",
        "ndcg@10",
        "ndcg@20",
        "hit@5",
        "hit@10",
        "hit@20",
    ]:
        if metric_name in ablation_record:
            reused_row[metric_name] = ablation_record[metric_name]
    for key in CONFIG_SNAPSHOT_KEYS:
        if key in ablation_record:
            reused_row[key] = ablation_record[key]
    return reused_row


def result_file_matches_config(excel_filepath, model_name, target_config):
    if not os.path.exists(excel_filepath):
        return False

    try:
        result_df = pd.read_excel(excel_filepath)
    except Exception:
        return False
    if result_df.empty:
        return False

    record = result_df.iloc[0].to_dict()
    expected_protocol = resolve_baseline_protocol(model_name, target_config.get("dataset", ""))
    if expected_protocol is not None:
        if "Transfer_Protocol" not in record:
            return False
        if str(record["Transfer_Protocol"]) != str(expected_protocol):
            return False

    if not model_name.startswith("USTv2_"):
        return True

    compare_keys = [key for key in CONFIG_SNAPSHOT_KEYS if key in target_config]
    missing_columns = [key for key in compare_keys if key not in record]
    if missing_columns:
        return False

    return all(values_close(record[key], target_config[key]) for key in compare_keys)


def run_custom_model(model_class, dataset_name, config_dict, config_file_list):
    runtime_config = apply_model_runtime_compat(config_dict, model_class)
    runtime_config = prepare_data_path_config(runtime_config, dataset_name)
    resolved_config_files = resolve_config_file_list(PROJECT_ROOT, config_file_list)

    config = Config(
        model=model_class,
        dataset=dataset_name,
        config_dict=runtime_config,
        config_file_list=resolved_config_files,
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = model_class(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.fit(
        train_data,
        valid_data,
        saved=config["saved"],
        show_progress=config["show_progress"],
    )
    test_result = trainer.evaluate(
        test_data,
        load_best_model=config["saved"],
        show_progress=config["show_progress"],
    )
    cleanup_checkpoint_file(trainer)
    return {"test_result": test_result}


def build_external_command(model_name, dataset_name, processed_dir, dynamic_config):
    script_path = os.path.join(PROJECT_ROOT, resolve_external_script(model_name, dataset_name))
    command = [
        sys.executable,
        script_path,
        "--project_root",
        PROJECT_ROOT,
        "--dataset",
        dataset_name,
        "--processed_dir",
        processed_dir,
        "--gpu_id",
        str(dynamic_config.get("gpu_id", 0)),
        "--seed",
        str(dynamic_config.get("seed", 2026)),
    ]

    if model_name == "C2DSR":
        command.extend(
            [
                "--batch_size",
                str(dynamic_config["train_batch_size"]),
                "--eval_batch_size",
                str(dynamic_config["eval_batch_size"]),
                "--num_workers",
                str(dynamic_config.get("worker", 0)),
                "--stopping_step",
                str(dynamic_config.get("stopping_step", 10)),
            ]
        )
    else:
        command.extend(
            [
                "--train_batch_size",
                str(dynamic_config["train_batch_size"]),
                "--eval_batch_size",
                str(dynamic_config["eval_batch_size"]),
            ]
        )

    if model_name in {"UniSRec", "VQRec"}:
        command.extend(["--worker", str(dynamic_config.get("worker", 0))])
        if is_cross_domain_dataset(dataset_name):
            command.extend(
                [
                    "--pretrain_batch_size",
                    str(dynamic_config.get("pretrain_batch_size", 128)),
                ]
            )
    elif "_PTFT_" in model_name:
        backbone_name, ratio_tag = model_name.split("_PTFT_", 1)
        ratio_text = ratio_tag.replace("pct", "")
        shot_ratio = float(ratio_text) / 100.0
        command.extend(
            [
                "--backbone",
                backbone_name,
                "--shot_ratio",
                str(shot_ratio),
                "--worker",
                str(dynamic_config.get("worker", 0)),
            ]
        )
    return command


def parse_external_result(stdout_text):
    for line in reversed(stdout_text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("No JSON result payload found in external baseline output.")


def run_external_model(model_name, dataset_name, processed_dir, dynamic_config):
    command = build_external_command(model_name, dataset_name, processed_dir, dynamic_config)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"External model runner failed ({model_name} x {dataset_name}).\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return {"test_result": parse_external_result(completed.stdout)}

def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline comparisons for UST experiments.")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "single_domain", "cross_domain"],
        help="Choose whether to run all baselines, only the single-domain main table, or only the cross-domain main table.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Optional comma-separated dataset override. Example: All_Beauty,Electronics",
    )
    return parser.parse_args()


def normalize_dataset_selection(mode, dataset_override_text):
    if dataset_override_text.strip():
        requested = [item.strip() for item in dataset_override_text.split(",") if item.strip()]
        unknown = [item for item in requested if item not in datasets]
        if unknown:
            raise ValueError(f"Unknown dataset(s): {unknown}")
        return requested

    if mode == "single_domain":
        return list(SINGLE_DOMAIN_DATASETS)
    if mode == "cross_domain":
        return list(CROSS_DOMAIN_DATASETS)
    return list(datasets)


def current_models_for_dataset(dataset_name):
    return cross_domain_models if str(dataset_name).startswith("Cross_") else single_domain_models


def summary_suffix(mode):
    if mode == "single_domain":
        return "SingleDomain"
    if mode == "cross_domain":
        return "CrossDomain"
    return "All"


def run_one_model(dataset_name, model_name):
    current_processed_dir = os.path.join(BASE_DATA_DIR, f"Processed_{dataset_name}")
    if not os.path.isdir(current_processed_dir):
        print(f"[Skip] Missing processed dataset directory: {current_processed_dir}")
        return

    is_cross_domain = dataset_name.startswith("Cross_")
    pair_metadata = resolve_cross_dataset_metadata(current_processed_dir, dataset_name)
    if (
        is_cross_domain
        and model_name in {"UniSRec", "VQRec"}
        and resolve_baseline_protocol(model_name, dataset_name) == "source_pretrain_target_finetune"
    ):
        source_processed_dir = os.path.join(
            BASE_DATA_DIR, f"Processed_{pair_metadata['Source_Dataset']}"
        )
        target_processed_dir = os.path.join(
            BASE_DATA_DIR, f"Processed_{pair_metadata['Target_Dataset']}"
        )
        missing_dirs = [
            path
            for path in (source_processed_dir, target_processed_dir)
            if not os.path.isdir(path)
        ]
        if missing_dirs:
            print(
                "[Skip] Transfer baseline requires source/target processed datasets: "
                + ", ".join(missing_dirs)
            )
            return

    print("\n" + "=" * 60)
    print(f"[Run] {model_name} x {dataset_name}")
    print("=" * 60)

    excel_filename = f"{model_name}_{dataset_name}.xlsx"
    excel_filepath = os.path.join(RESULT_DIR, excel_filename)

    if "Elec" in dataset_name or "Tenrec" in dataset_name:
        cur_train_batch, cur_eval_batch, cur_worker = 512, 64, 0
    else:
        cur_train_batch, cur_eval_batch, cur_worker = 2048, 256, 4

    cur_pretrain_batch = None

    if model_name == "C2DSR" and ("Elec" in dataset_name or "Tenrec" in dataset_name):
        cur_train_batch, cur_eval_batch, cur_worker = 256, 256, 0

    if is_cross_domain and model_name in {"UniSRec", "VQRec"}:
        cur_train_batch, cur_eval_batch, cur_worker = 256, 64, 0
        cur_pretrain_batch = 128

    dynamic_config = {
        "dataset": dataset_name,
        "data_path": current_processed_dir,
        "show_progress": False,
        "train_batch_size": cur_train_batch,
        "eval_batch_size": cur_eval_batch,
        "MAX_ITEM_LIST_LENGTH": 50,
        "worker": cur_worker,
        "checkpoint_dir": resolve_checkpoint_dir(PROJECT_ROOT),
        "save_dataset": False,
        "save_dataloaders": False,
        "saved": True,
    }
    if cur_pretrain_batch is not None:
        dynamic_config["pretrain_batch_size"] = cur_pretrain_batch
    protocol_name = resolve_baseline_protocol(model_name, dataset_name)

    using_default_params = False
    target_params = None
    if model_name.startswith("USTv2_"):
        dataset_params = BEST_PARAMS_MAP.get(dataset_name, {})
        using_default_params = model_name not in dataset_params
        target_params = dict(dataset_params.get(model_name, DEFAULT_V2_PARAMS))
        if not is_cross_domain:
            target_params["beta2"] = 0.0
        elif target_params.get("beta2", 0.0) <= 0:
            target_params["beta2"] = 0.05
        target_params.setdefault("mm_loss_weight", DEFAULT_V2_PARAMS["mm_loss_weight"])
        dynamic_config.update({"use_ust": True, **target_params})

    if os.path.exists(excel_filepath):
        if result_file_matches_config(excel_filepath, model_name, dynamic_config):
            print(f"[Skip] Existing result found: {excel_filename}")
            return
        print(f"[Refresh] Existing result has stale or missing params: {excel_filename}")

    ablation_df = pd.DataFrame()
    ablation_path = None
    reuse_variant_name = None
    if model_name in {"SASRec", "GRU4Rec", "USTv2_SASRec", "USTv2_GRU4Rec"}:
        model_backbone = (
            model_name if model_name in {"SASRec", "GRU4Rec"} else model_name.split("_", 1)[1]
        )
        ablation_df, ablation_path = load_ablation_table(PROJECT_ROOT, dataset_name, model_backbone)
        if model_name.startswith("USTv2_"):
            reuse_variant_name = "7_Full_UST_Cross" if is_cross_domain else "6_Full_UST_Single"
        else:
            reuse_variant_name = "1_Vanilla"

    if reuse_variant_name and not ablation_df.empty:
        ablation_record = match_ablation_record(
            ablation_df,
            model_name=model_name,
            dataset_name=dataset_name,
            target_config=dynamic_config,
            variant_name=reuse_variant_name,
        )
        if ablation_record is not None:
            print(
                f"[Reuse] Using ablation cache for {model_name} x {dataset_name} "
                f"from {os.path.basename(ablation_path)} ({reuse_variant_name})."
            )
            reused_row = build_reused_result_row(
                model_name=model_name,
                dataset_name=dataset_name,
                ablation_record=ablation_record,
                variant_name=reuse_variant_name,
            )
            reused_row.update(pair_metadata)
            if protocol_name is not None:
                reused_row["Transfer_Protocol"] = protocol_name
            df = pd.DataFrame([reused_row])
            columns = ["Model", "Dataset", "Family", "Backbone"] + [
                column
                for column in df.columns
                if column not in ["Model", "Dataset", "Family", "Backbone"]
            ]
            df[columns].to_excel(excel_filepath, index=False)
            print(f"[Save] Reused result written to {excel_filename}")
            return

    if model_name.startswith("USTv2_"):
        print(f"[USTv2] Injected params: {target_params}")
        if using_default_params:
            print(
                f"[Warn] {dataset_name} x {model_name} is using DEFAULT_V2_PARAMS. "
                f"Backfill the final tuned params into BEST_PARAMS_MAP['{dataset_name}']['{model_name}'] "
                "before the final run.",
            )
    elif model_name in EXTERNAL_MODEL_SCRIPTS:
        if protocol_name:
            print(
                "[External Baseline] Using the project-aligned adapter with unified metrics. "
                f"Protocol={protocol_name}."
            )
        else:
            print("[External Baseline] Using the project-aligned adapter with unified metrics.")
    else:
        print("[Baseline] Using the same evaluation protocol without USTv2-specific losses.")

    try:
        if model_name in MODEL_DICT:
            model_class = MODEL_DICT[model_name]
            result = run_custom_model(
                model_class=model_class,
                dataset_name=dataset_name,
                config_dict=dynamic_config,
                config_file_list=get_model_config_files(model_name),
            )
        else:
            result = run_external_model(
                model_name=model_name,
                dataset_name=dataset_name,
                processed_dir=current_processed_dir,
                dynamic_config=dynamic_config,
            )

        test_res = result["test_result"]
        family_name, backbone_name = parse_model_metadata(model_name)
        test_res["Model"] = model_name
        test_res["Dataset"] = dataset_name
        test_res["Family"] = family_name
        test_res["Backbone"] = backbone_name
        test_res["Source"] = "trained"
        test_res.update(pair_metadata)
        if protocol_name is not None:
            test_res["Transfer_Protocol"] = protocol_name
        for config_key in CONFIG_SNAPSHOT_KEYS:
            if config_key in dynamic_config:
                test_res[config_key] = dynamic_config[config_key]

        df = pd.DataFrame([test_res])
        columns = ["Model", "Dataset", "Family", "Backbone"] + [
            column
            for column in df.columns
            if column not in ["Model", "Dataset", "Family", "Backbone"]
        ]
        df[columns].to_excel(excel_filepath, index=False)
        print(f"[Save] Wrote result to {excel_filename}")
    except Exception as exc:
        print(f"[Error] Failed on {model_name} x {dataset_name}: {exc}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def collect_mode_summary(dataset_names):
    all_dfs = []
    for file_name in os.listdir(RESULT_DIR):
        if file_name.endswith(".xlsx") and not file_name.startswith("Final_"):
            try:
                frame = pd.read_excel(os.path.join(RESULT_DIR, file_name))
            except Exception:
                continue
            if frame.empty or "Dataset" not in frame.columns:
                continue
            frame = frame[frame["Dataset"].astype(str).isin(dataset_names)].copy()
            if not frame.empty:
                all_dfs.append(frame)

    if not all_dfs:
        return pd.DataFrame()

    final_df = pd.concat(all_dfs, ignore_index=True)
    if "Backbone" in final_df.columns:
        final_df = final_df[final_df["Backbone"] != "LightGCN"].copy()
    if "Model" in final_df.columns:
        final_df = final_df[final_df["Model"].isin(ACTIVE_RESULT_MODELS)].copy()
    if not final_df.empty:
        final_df.sort_values(by=["Dataset", "Model"], inplace=True)
    return final_df


def main():
    args = parse_args()
    selected_datasets = normalize_dataset_selection(args.mode, args.datasets)

    print(f"[USTv2] Running baseline comparison. Results directory: {RESULT_DIR}")
    print(f"[USTv2] Mode: {args.mode}")
    print(f"[USTv2] Selected datasets: {selected_datasets}")

    for dataset_name in selected_datasets:
        for model_name in current_models_for_dataset(dataset_name):
            run_one_model(dataset_name, model_name)

    print("\n" + "=" * 60)
    print("[USTv2] Merging single-run result files.")

    final_df = collect_mode_summary(selected_datasets)
    if final_df.empty:
        print("[Warn] No successful result files were found for the selected mode.")
        return

    if args.mode == "all":
        final_path = os.path.join(RESULT_DIR, "Final_Summary_Results.xlsx")
    else:
        suffix = summary_suffix(args.mode)
        final_path = os.path.join(RESULT_DIR, f"Final_Summary_Results_{suffix}.xlsx")
    final_df.to_excel(final_path, index=False)
    print(f"[Done] Summary saved to {final_path}")

    if args.mode == "all":
        try:
            from build_ust_comparison_reports import build_baseline_report

            compare_path = build_baseline_report(PROJECT_ROOT, os.path.join(PROJECT_ROOT, "results_compare"))
            if compare_path:
                print(f"[Done] Baseline comparison summary saved to {compare_path}")
        except Exception as exc:
            print(f"[Warn] Failed to build comparison summary: {exc}")


if __name__ == "__main__":
    main()
