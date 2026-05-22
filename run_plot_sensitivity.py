import argparse
import gc
import os
import sys

import ust_bootstrap  # noqa: F401
import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_trainer, init_logger, init_seed

from USTv2_GRU4Rec import USTv2_GRU4Rec
from USTv2_SASRec import USTv2_SASRec
from ust_reporting import build_model_wide, parse_model_metadata, write_comparison_workbook
from ust_utils import (
    apply_model_runtime_compat,
    cleanup_checkpoint_file,
    get_model_config_files,
    prepare_data_path_config,
    resolve_checkpoint_dir,
    resolve_config_file_list,
)


MODEL_DICT = {
    "USTv2_SASRec": USTv2_SASRec,
    "USTv2_GRU4Rec": USTv2_GRU4Rec,
}

SINGLE_DOMAIN_TAU_MIN = 0.1
CROSS_DOMAIN_TAU_MIN = 0.2
COARSE_RESULT_DIR_NAME = "results_sensitivity"
COARSE_FIXED_DEFAULTS = {
    "gumbel_tau": 1.0,
    # Older coarse-search files did not store tau_min explicitly and were
    # generated with this historical default. Do not reuse them for the new
    # single-domain tau_min=0.1 setting.
    "tau_min": 0.2,
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
PARAM_COLUMN_CANDIDATES = {
    "vocab_size": ("Vocab_Size", "vocab_size"),
    "beta1": ("Beta1", "beta1"),
    "beta2": ("Beta2", "beta2"),
    "beta3": ("Beta3", "beta3"),
    "beta4": ("Beta4", "beta4"),
    "gumbel_tau": ("Gumbel_Tau", "gumbel_tau"),
    "tau_min": ("Tau_Min", "tau_min"),
}


def is_v2_model_name(model_name):
    return model_name.startswith("USTv2_")


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
    parser = argparse.ArgumentParser(description="USTv2 plot-data generator")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_DICT.keys()))
    parser.add_argument(
        "--compare_with",
        type=str,
        default=None,
        choices=list(MODEL_DICT.keys()),
        help="Optional extra model to export into the same comparison workbook.",
    )
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--beta1", type=float, default=0.01)
    parser.add_argument("--beta2", type=float, default=0.1)
    parser.add_argument("--beta3", type=float, default=0.1)
    parser.add_argument("--beta4", type=float, default=0.1)
    parser.add_argument("--gumbel_tau", type=float, default=1.0)
    parser.add_argument(
        "--tau_min",
        type=float,
        default=None,
        help="Ignored in fine-tuning mode. tau_min is fixed to 0.1 for single-domain and 0.2 for cross-domain datasets.",
    )
    parser.add_argument("--mm_loss_weight", type=float, default=0.1)
    return parser.parse_args()


def build_base_config(args, is_cross_domain):
    # This base config should come from the best coarse-search row before the
    # fine-grained plotting sweep starts. tau_min is fixed by domain setting
    # and therefore is not treated as a fine-grained sweep parameter.
    tau_min = CROSS_DOMAIN_TAU_MIN if is_cross_domain else SINGLE_DOMAIN_TAU_MIN
    return {
        "vocab_size": args.vocab_size,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "beta3": args.beta3,
        "beta4": args.beta4,
        "gumbel_tau": args.gumbel_tau,
        "tau_min": tau_min,
        "mm_loss_weight": args.mm_loss_weight,
        "use_ust": True,
    }


def build_sensitivity_ranges(model_name, is_cross_domain):
    sensitivity_ranges = {
        "vocab_size": [64, 128, 256, 1024],
        "beta1": [0.01, 0.05, 0.1, 0.2],
        "beta3": [0.01, 0.05, 0.1, 0.5],
    }
    if is_v2_model_name(model_name):
        sensitivity_ranges["beta4"] = [0.01, 0.05, 0.1, 0.2]
    if not is_cross_domain:
        # Keep 1.0 as a reference point in the plot, but reuse the coarse-search
        # result instead of retraining it.
        sensitivity_ranges["gumbel_tau"] = [0.1, 0.2, 0.5, 1.0]
    if is_cross_domain:
        sensitivity_ranges["beta2"] = [0.01, 0.05, 0.1, 0.5]
    return sensitivity_ranges


def values_close(left_value, right_value, tol=1e-12):
    try:
        return abs(float(left_value) - float(right_value)) <= tol
    except (TypeError, ValueError):
        return str(left_value) == str(right_value)


def load_coarse_search_results(project_root, model_name, dataset_name):
    coarse_path = os.path.join(
        project_root,
        COARSE_RESULT_DIR_NAME,
        f"Sensitivity_{model_name}_{dataset_name}.xlsx",
    )
    if not os.path.exists(coarse_path):
        return pd.DataFrame()

    try:
        return pd.read_excel(coarse_path)
    except Exception as exc:
        print(f"[Warn] Failed to load coarse-search results from {coarse_path}: {exc}")
        return pd.DataFrame()


def find_matching_coarse_record(coarse_df, model_name, dataset_name, config):
    if coarse_df is None or coarse_df.empty:
        return None

    mask = pd.Series(True, index=coarse_df.index)
    if "Model" in coarse_df.columns:
        mask &= coarse_df["Model"].astype(str) == str(model_name)
    if "Dataset" in coarse_df.columns:
        mask &= coarse_df["Dataset"].astype(str) == str(dataset_name)

    for config_key, column_candidates in PARAM_COLUMN_CANDIDATES.items():
        if config_key not in config:
            continue

        matching_column = next(
            (column for column in column_candidates if column in coarse_df.columns),
            None,
        )
        if matching_column is None:
            fixed_value = COARSE_FIXED_DEFAULTS.get(config_key)
            if fixed_value is not None and not values_close(config[config_key], fixed_value):
                return None
            continue

        mask &= coarse_df[matching_column].apply(
            lambda value: values_close(value, config[config_key])
        )

    matches = coarse_df[mask]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def plot_record_matches_config(record, config):
    for config_key, column_candidates in PARAM_COLUMN_CANDIDATES.items():
        if config_key not in config:
            continue

        matching_column = next(
            (column for column in column_candidates if column in record),
            None,
        )
        if matching_column is None:
            return False
        if not values_close(record[matching_column], config[config_key]):
            return False
    return True


def get_metric_value(metric_dict, *candidate_names):
    lowercase_map = {str(key).lower(): value for key, value in metric_dict.items()}
    for candidate_name in candidate_names:
        if candidate_name.lower() in lowercase_map:
            return lowercase_map[candidate_name.lower()]
    return 0.0


def build_plot_record(
    model_name,
    dataset_name,
    family_name,
    backbone_name,
    tuning_param_name,
    value,
    current_config,
    metric_dict,
    source,
):
    res_dict = {
        "Model": model_name,
        "Dataset": dataset_name,
        "Family": family_name,
        "Backbone": backbone_name,
        "Tuning_Param": tuning_param_name,
        "Test_Value": value,
        "Source": source,
        "NDCG@10": get_metric_value(metric_dict, "NDCG@10", "ndcg@10"),
        "Recall@10": get_metric_value(metric_dict, "Recall@10", "recall@10"),
    }
    for metric_name in METRIC_COLUMNS:
        metric_value = get_metric_value(metric_dict, metric_name)
        if metric_value != 0.0 or metric_name in metric_dict:
            res_dict[metric_name] = metric_value
    res_dict.update(current_config)
    return res_dict


def write_plot_records(all_results, excel_filepath):
    ordered_cols = [
        "Model",
        "Dataset",
        "Family",
        "Backbone",
        "Tuning_Param",
        "Test_Value",
        "Source",
        "NDCG@10",
        "Recall@10",
    ]
    df = pd.DataFrame(all_results)
    ordered_cols += [column for column in df.columns if column not in ordered_cols]
    df[ordered_cols].to_excel(excel_filepath, index=False)


def run_single_model_plotdata(model_name, args, project_root, result_dir):
    base_data_dir = os.path.join(project_root, "data")
    target_dataset = args.dataset
    current_processed_dir = os.path.join(base_data_dir, f"Processed_{target_dataset}")
    if not os.path.isdir(current_processed_dir):
        raise FileNotFoundError(f"Processed dataset directory not found: {current_processed_dir}")

    is_cross_domain = target_dataset.startswith("Cross_")
    optimal_config = build_base_config(args, is_cross_domain)
    if not is_cross_domain:
        optimal_config["beta2"] = 0.0

    sensitivity_ranges = build_sensitivity_ranges(model_name, is_cross_domain)
    family_name, backbone_name = parse_model_metadata(model_name)

    print(f"[USTv2] Plot data generation for {model_name} on {target_dataset}")
    print(f"[USTv2] Base config: {optimal_config}")
    print(
        f"[USTv2] Fixed tau_min={'0.2' if is_cross_domain else '0.1'} "
        f"({'cross-domain' if is_cross_domain else 'single-domain'} setting)"
    )

    all_results = []
    excel_filepath = os.path.join(result_dir, f"PlotData_{model_name}_{target_dataset}.xlsx")
    coarse_df = load_coarse_search_results(project_root, model_name, target_dataset)
    if not coarse_df.empty:
        print(
            f"[USTv2] Loaded coarse-search cache for reuse: "
            f"{len(coarse_df)} record(s)."
        )

    for tuning_param_name, tuning_values in sensitivity_ranges.items():
        print("\n" + "=" * 60)
        print(f"[Tune] {tuning_param_name}")

        for value in tuning_values:
            current_config = optimal_config.copy()
            current_config[tuning_param_name] = value
            print(f"[Value] {tuning_param_name} = {value}")

            if os.path.exists(excel_filepath):
                try:
                    df_exist = pd.read_excel(excel_filepath)
                    mask = (df_exist["Tuning_Param"] == tuning_param_name) & (
                        df_exist["Test_Value"] == value
                    )
                    if mask.any():
                        matched_records = [
                            record
                            for record in df_exist[mask].to_dict("records")
                            if plot_record_matches_config(record, current_config)
                        ]
                        if matched_records:
                            print("[Skip] Existing plot point found.")
                            all_results.append(matched_records[0])
                            continue
                except Exception:
                    pass

            coarse_record = find_matching_coarse_record(
                coarse_df,
                model_name,
                target_dataset,
                current_config,
            )
            if coarse_record is not None:
                print("[Reuse] Matched a coarse-search result. Skip training.")
                all_results.append(
                    build_plot_record(
                        model_name=model_name,
                        dataset_name=target_dataset,
                        family_name=family_name,
                        backbone_name=backbone_name,
                        tuning_param_name=tuning_param_name,
                        value=value,
                        current_config=current_config,
                        metric_dict=coarse_record,
                        source="coarse_reuse",
                    )
                )
                write_plot_records(all_results, excel_filepath)
                continue

            if "Elec" in target_dataset or "Tenrec" in target_dataset or "Cloth" in target_dataset:
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
                "checkpoint_dir": resolve_checkpoint_dir(project_root),
                "save_dataset": False,
                "save_dataloaders": False,
                "saved": True,
            }
            dynamic_config.update(current_config)

            try:
                model_class = MODEL_DICT[model_name]
                result = run_custom_model(
                    model_class=model_class,
                    dataset_name=target_dataset,
                    config_dict=dynamic_config,
                    config_file_list=get_model_config_files(model_name),
                )

                test_res = result["test_result"]
                res_dict = build_plot_record(
                    model_name=model_name,
                    dataset_name=target_dataset,
                    family_name=family_name,
                    backbone_name=backbone_name,
                    tuning_param_name=tuning_param_name,
                    value=value,
                    current_config=current_config,
                    metric_dict=test_res,
                    source="fine_tuning",
                )
                all_results.append(res_dict)
                write_plot_records(all_results, excel_filepath)
            except Exception as exc:
                print(f"[Error] Failed on {tuning_param_name}={value}: {exc}")
            finally:
                gc.collect()
                torch.cuda.empty_cache()

    print(f"\n[Done] Plot data saved to {excel_filepath}")
    return excel_filepath


def build_model_compare_workbook(model_names, dataset_name, result_dir):
    frames = []
    for model_name in model_names:
        file_path = os.path.join(result_dir, f"PlotData_{model_name}_{dataset_name}.xlsx")
        if not os.path.exists(file_path):
            continue
        frames.append(pd.read_excel(file_path))

    if len(frames) < 2:
        return None

    pair_df = pd.concat(frames, ignore_index=True)
    plot_wide = build_model_wide(pair_df, ["Dataset", "Tuning_Param", "Test_Value"])
    output_path = os.path.join(result_dir, f"PlotCompare_{dataset_name}_USTv2_Models.xlsx")
    write_comparison_workbook(
        output_path,
        {
            "plot_long": pair_df,
            "plot_wide": plot_wide,
        },
    )
    return output_path


def main():
    args = parse_args()
    sys.argv = [sys.argv[0]]

    project_root = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(project_root, "results_sensitivity_plot")
    os.makedirs(result_dir, exist_ok=True)

    target_models = [args.model]
    if args.compare_with and args.compare_with not in target_models:
        target_models.append(args.compare_with)

    for model_name in target_models:
        run_single_model_plotdata(model_name, args, project_root, result_dir)

    compare_path = None
    if len(target_models) >= 2:
        try:
            compare_path = build_model_compare_workbook(target_models, args.dataset, result_dir)
        except Exception as exc:
            print(f"[Warn] Failed to build comparison workbook: {exc}")

    if compare_path:
        print(f"[Done] Model comparison workbook saved to {compare_path}")


if __name__ == "__main__":
    main()
