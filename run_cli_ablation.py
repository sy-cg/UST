import argparse
import gc
import os
import sys
import traceback

import ust_bootstrap  # noqa: F401
import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.model.sequential_recommender.gru4rec import GRU4Rec
from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.utils import get_trainer, init_logger, init_seed

from USTv2_GRU4Rec import USTv2_GRU4Rec
from USTv2_LightGCN import USTv2_LightGCN
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
    "LightGCN": LightGCN,
    "USTv2_SASRec": USTv2_SASRec,
    "USTv2_GRU4Rec": USTv2_GRU4Rec,
    "USTv2_LightGCN": USTv2_LightGCN,
}

PLOT_RESULT_DIR_NAME = "results_sensitivity_plot"
SINGLE_DOMAIN_TAU_MIN = 0.1
CROSS_DOMAIN_TAU_MIN = 0.2
PARAM_COLUMN_CANDIDATES = {
    "vocab_size": ("vocab_size", "Vocab_Size"),
    "beta1": ("beta1", "Beta1"),
    "beta2": ("beta2", "Beta2"),
    "beta3": ("beta3", "Beta3"),
    "beta4": ("beta4", "Beta4"),
    "gumbel_tau": ("gumbel_tau", "Gumbel_Tau"),
    "tau_min": ("tau_min", "Tau_Min"),
    "mm_loss_weight": ("mm_loss_weight", "MM_Loss_Weight"),
}
METRIC_COLUMNS = [
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
]
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
    "ust_token_mode",
]


def run_custom_model(model_class, dataset_name, config_dict, config_file_list):
    project_root = os.path.dirname(os.path.abspath(__file__))
    runtime_config = apply_model_runtime_compat(config_dict, model_class)
    runtime_config = prepare_data_path_config(runtime_config, dataset_name)
    resolved_config_files = resolve_config_file_list(project_root, config_file_list)

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


def parse_args():
    parser = argparse.ArgumentParser(description="USTv2 ablation runner")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--model", type=str, required=True, choices=["SASRec", "GRU4Rec", "LightGCN"]
    )
    parser.add_argument("--beta1", type=float, default=0.01)
    parser.add_argument("--beta2", type=float, default=0.1)
    parser.add_argument("--beta3", type=float, default=0.1)
    parser.add_argument("--beta4", type=float, default=0.1)
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--gumbel_tau", type=float, default=1.0)
    parser.add_argument(
        "--tau_min",
        type=float,
        default=None,
        help="If omitted, use 0.1 for single-domain and 0.2 for cross-domain datasets.",
    )
    parser.add_argument("--mm_loss_weight", type=float, default=0.1)
    return parser.parse_args()


def get_ust_model_name(backbone_name):
    if backbone_name == "GRU4Rec":
        return "USTv2_GRU4Rec"
    return f"USTv2_{backbone_name}"


def load_existing_ablation_records(excel_filepath):
    if not os.path.exists(excel_filepath):
        return [], {}

    try:
        existing_df = pd.read_excel(excel_filepath)
    except Exception:
        return [], {}

    if existing_df.empty or "Variant" not in existing_df.columns:
        return [], {}

    records = existing_df.to_dict("records")
    latest_by_variant = {}
    for record in records:
        variant_name = record.get("Variant")
        if variant_name is not None:
            latest_by_variant[str(variant_name)] = record
    return records, latest_by_variant


def upsert_ablation_record(all_results, record):
    variant_name = record["Variant"]
    filtered_results = [
        existing_record
        for existing_record in all_results
        if str(existing_record.get("Variant")) != str(variant_name)
    ]
    filtered_results.append(record)
    return filtered_results


def values_close(left_value, right_value, tol=1e-12):
    try:
        return abs(float(left_value) - float(right_value)) <= tol
    except (TypeError, ValueError):
        return str(left_value) == str(right_value)


def load_plot_records(project_root, model_name, dataset_name):
    plot_path = os.path.join(
        project_root,
        PLOT_RESULT_DIR_NAME,
        f"PlotData_{model_name}_{dataset_name}.xlsx",
    )
    if not os.path.exists(plot_path):
        return pd.DataFrame()

    try:
        return pd.read_excel(plot_path)
    except Exception as exc:
        print(f"[Warn] Failed to load plot-data cache from {plot_path}: {exc}")
        return pd.DataFrame()


def find_matching_plot_record(plot_df, model_name, dataset_name, config):
    if plot_df is None or plot_df.empty:
        return None

    mask = pd.Series(True, index=plot_df.index)
    if "Model" in plot_df.columns:
        mask &= plot_df["Model"].astype(str) == str(model_name)
    if "Dataset" in plot_df.columns:
        mask &= plot_df["Dataset"].astype(str) == str(dataset_name)

    for config_key, column_candidates in PARAM_COLUMN_CANDIDATES.items():
        if config_key not in config:
            continue

        matching_column = next(
            (column for column in column_candidates if column in plot_df.columns),
            None,
        )
        if matching_column is None:
            continue

        mask &= plot_df[matching_column].apply(
            lambda value: values_close(value, config[config_key])
        )

    matches = plot_df[mask]
    if matches.empty:
        return None

    if "NDCG@10" in matches.columns:
        matches = matches.sort_values(by="NDCG@10", ascending=False)
    return matches.iloc[0].to_dict()


def get_metric_from_record(record, metric_name):
    candidates = [metric_name, metric_name.upper(), metric_name.replace("@", "@").upper()]
    if metric_name.startswith("ndcg"):
        candidates.append(metric_name.replace("ndcg", "NDCG"))
    elif metric_name.startswith("recall"):
        candidates.append(metric_name.replace("recall", "Recall"))
    elif metric_name.startswith("mrr"):
        candidates.append(metric_name.replace("mrr", "MRR"))
    elif metric_name.startswith("hit"):
        candidates.append(metric_name.replace("hit", "Hit"))

    lowercase_map = {str(key).lower(): value for key, value in record.items()}
    for candidate in candidates:
        if candidate.lower() in lowercase_map:
            return lowercase_map[candidate.lower()]
    return None


def build_ablation_record_from_plot(variant_name, model_name, dataset_name, plot_record):
    family_name, backbone_name = parse_model_metadata(model_name)
    res_dict = {
        "Variant": variant_name,
        "Model": model_name,
        "Dataset": dataset_name,
        "Family": family_name,
        "Backbone": backbone_name,
        "Status": "success",
        "Error_Message": "",
        "Source": "plot_reuse",
    }
    for metric_name in METRIC_COLUMNS:
        metric_value = get_metric_from_record(plot_record, metric_name)
        if metric_value is not None:
            res_dict[metric_name] = metric_value
    return res_dict


def attach_config_snapshot(record, config_dict):
    for key in CONFIG_SNAPSHOT_KEYS:
        if key in config_dict:
            record[key] = config_dict[key]
    return record


def ablation_record_matches_config(record, config_dict, model_name):
    compare_keys = [key for key in CONFIG_SNAPSHOT_KEYS if key in config_dict]
    if model_name.startswith("USTv2_"):
        missing_keys = [key for key in compare_keys if key not in record]
        if missing_keys:
            return False

    for key in compare_keys:
        if key not in record:
            continue
        if not values_close(record[key], config_dict[key]):
            return False
    return True


def main():
    args = parse_args()
    sys.argv = [sys.argv[0]]

    project_root = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.join(project_root, "data")
    result_dir = os.path.join(project_root, "results_ablation")
    os.makedirs(result_dir, exist_ok=True)

    target_dataset = args.dataset
    target_backbone = args.model
    target_ust_model = get_ust_model_name(target_backbone)
    is_cross_domain = target_dataset.startswith("Cross_")
    tau_min = (
        args.tau_min
        if args.tau_min is not None
        else (CROSS_DOMAIN_TAU_MIN if is_cross_domain else SINGLE_DOMAIN_TAU_MIN)
    )

    optimal_ust_config = {
        "use_ust": True,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "beta3": args.beta3,
        "beta4": args.beta4,
        "vocab_size": args.vocab_size,
        "gumbel_tau": args.gumbel_tau,
        "tau_min": tau_min,
        "mm_loss_weight": args.mm_loss_weight,
    }
    early_fusion_config = {**optimal_ust_config, "use_ust": False}
    if target_backbone == "LightGCN":
        early_fusion_config.update(
            {
                "beta1": 0.0,
                "beta2": 0.0,
                "beta3": 0.0,
                "beta4": 0.0,
                "mm_loss_weight": 0.0,
            }
        )

    ablation_variants = {
        "1_Vanilla": {
            "model_to_run": target_backbone,
            "config": {},
        },
        "2_EarlyFusion": {
            "model_to_run": target_ust_model,
            "config": early_fusion_config,
        },
        "3_wo_Compress": {
            "model_to_run": target_ust_model,
            "config": {**optimal_ust_config, "beta1": 0.0},
        },
        "4_wo_Orthogonal": {
            "model_to_run": target_ust_model,
            "config": {**optimal_ust_config, "beta3": 0.0},
        },
    }

    ablation_variants["5_wo_Commit"] = {
        "model_to_run": target_ust_model,
        "config": {**optimal_ust_config, "beta4": 0.0},
    }

    ablation_variants["6_Shared_Only"] = {
        "model_to_run": target_ust_model,
        "config": {
            **optimal_ust_config,
            "ust_token_mode": "shared_only",
            "beta3": 0.0,
        },
    }
    ablation_variants["7_Private_Only"] = {
        "model_to_run": target_ust_model,
        "config": {
            **optimal_ust_config,
            "ust_token_mode": "private_only",
            "beta2": 0.0,
            "beta3": 0.0,
        },
    }

    if is_cross_domain:
        next_index = len(ablation_variants) + 1
        ablation_variants[f"{next_index}_wo_Align"] = {
            "model_to_run": target_ust_model,
            "config": {**optimal_ust_config, "beta2": 0.0},
        }
        ablation_variants[f"{next_index + 1}_Full_UST_Cross"] = {
            "model_to_run": target_ust_model,
            "config": optimal_ust_config,
        }
    else:
        next_index = len(ablation_variants) + 1
        full_single_config = {**optimal_ust_config, "beta2": 0.0}
        ablation_variants[f"{next_index}_Full_UST_Single"] = {
            "model_to_run": target_ust_model,
            "config": full_single_config,
        }

    current_processed_dir = os.path.join(base_data_dir, f"Processed_{target_dataset}")
    if not os.path.isdir(current_processed_dir):
        raise FileNotFoundError(f"Processed dataset directory not found: {current_processed_dir}")

    excel_filepath = os.path.join(
        result_dir, f"Ablation_USTv2_{target_backbone}_{target_dataset}.xlsx"
    )
    all_results, latest_records = load_existing_ablation_records(excel_filepath)
    plot_df = load_plot_records(project_root, target_ust_model, target_dataset)
    if not plot_df.empty:
        print(f"[USTv2] Loaded plot-data cache for full-model reuse: {len(plot_df)} record(s).")

    for variant_name, setup in sorted(ablation_variants.items(), key=lambda item: item[0]):
        print("\n" + "=" * 60)
        print(f"[Run] {variant_name}")

        if "Elec" in target_dataset or "Tenrec" in target_dataset:
            cur_train_batch, cur_eval_batch, cur_worker = 512, 64, 0
        else:
            cur_train_batch, cur_eval_batch, cur_worker = 2048, 256, 4

        dynamic_config = {
            "dataset": target_dataset,
            "data_path": current_processed_dir,
            "show_progress": False,
            "train_batch_size": cur_train_batch,
            "eval_batch_size": cur_eval_batch,
            "MAX_ITEM_LIST_LENGTH": 50,
            "worker": cur_worker,
            "saved": True,
            "checkpoint_dir": resolve_checkpoint_dir(project_root),
            "save_dataset": False,
            "save_dataloaders": False,
        }
        dynamic_config.update(setup["config"])

        model_name = setup["model_to_run"]
        existing_record = latest_records.get(variant_name)
        if existing_record and existing_record.get("Status", "success") == "success":
            if ablation_record_matches_config(existing_record, dynamic_config, model_name):
                print(f"[Skip] Existing ablation record found for {variant_name}")
                continue
            print(f"[Refresh] Existing {variant_name} uses stale params. Rerun it.")

        if variant_name.endswith("Full_UST_Single") or variant_name.endswith("Full_UST_Cross"):
            plot_record = find_matching_plot_record(
                plot_df,
                model_name,
                target_dataset,
                dynamic_config,
            )
            if plot_record is not None:
                print("[Reuse] Matched fine-tuning plot data for full model. Skip training.")
                res_dict = build_ablation_record_from_plot(
                    variant_name,
                    model_name,
                    target_dataset,
                    plot_record,
                )
                res_dict = attach_config_snapshot(res_dict, dynamic_config)
                all_results = upsert_ablation_record(all_results, res_dict)
                latest_records[variant_name] = res_dict
                pd.DataFrame(all_results).to_excel(excel_filepath, index=False)
                continue

        try:
            model_class = MODEL_DICT[model_name]
            config_files = get_model_config_files(model_name)
            if variant_name == "1_Vanilla":
                config_files = get_model_config_files(target_backbone)

            result = run_custom_model(
                model_class=model_class,
                dataset_name=target_dataset,
                config_dict=dynamic_config,
                config_file_list=config_files,
            )

            res_dict = {
                "Variant": variant_name,
                "Model": model_name,
                "Dataset": target_dataset,
            }
            family_name, backbone_name = parse_model_metadata(model_name)
            res_dict["Family"] = family_name
            res_dict["Backbone"] = backbone_name
            res_dict["Status"] = "success"
            res_dict["Error_Message"] = ""
            res_dict = attach_config_snapshot(res_dict, dynamic_config)
            res_dict.update(result["test_result"])
            all_results = upsert_ablation_record(all_results, res_dict)
            latest_records[variant_name] = res_dict
            pd.DataFrame(all_results).to_excel(excel_filepath, index=False)
            print(f"[Save] Stored result for {variant_name}")
        except Exception as exc:
            print(f"[Error] Variant failed: {variant_name}")
            traceback.print_exc()
            failed_record = {
                "Variant": variant_name,
                "Model": model_name,
                "Dataset": target_dataset,
                "Family": parse_model_metadata(model_name)[0],
                "Backbone": parse_model_metadata(model_name)[1],
                "Status": "failed",
                "Error_Message": str(exc),
            }
            failed_record = attach_config_snapshot(failed_record, dynamic_config)
            all_results = upsert_ablation_record(all_results, failed_record)
            latest_records[variant_name] = failed_record
            pd.DataFrame(all_results).to_excel(excel_filepath, index=False)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n[Done] Ablation report written to {excel_filepath}")


if __name__ == "__main__":
    main()
