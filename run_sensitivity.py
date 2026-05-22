import gc
import itertools
import os

import ust_bootstrap  # noqa: F401
import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
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


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


MODEL_DICT = {
    "USTv2_SASRec": USTv2_SASRec,
    "USTv2_GRU4Rec": USTv2_GRU4Rec,
}

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results_sensitivity")
os.makedirs(RESULT_DIR, exist_ok=True)

target_models = ["USTv2_SASRec", "USTv2_GRU4Rec"]
# Stage-1 coarse tuning datasets. After selecting the best row from the Excel
# output, backfill that config into run_all_plots.py -> PLOT_TASKS[*]["args"].
target_datasets = ["All_Beauty", "Cross_Sport_Beauty"]#, "Cross_Elec_Sport", "Tenrec_SBR"

param_grid = {
    "vocab_size": [128, 256],
    "beta1": [0.01, 0.05],
    "beta3": [0.05, 0.2],
}
cross_domain_beta2_list = [0.0, 0.1]
beta4_list = [0.05, 0.1]
DEFAULT_EXTRA_CONFIG = {
    "gumbel_tau": 1.0,
    "mm_loss_weight": 0.1,
}


def is_v2_model_name(model_name):
    return model_name.startswith("USTv2_")


def values_close(left_value, right_value, tol=1e-12):
    try:
        return abs(float(left_value) - float(right_value)) <= tol
    except (TypeError, ValueError):
        return False


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


def get_cache_key(
    vocab_size,
    beta1,
    beta2,
    beta3,
    beta4=0.0,
    gumbel_tau=1.0,
    tau_min=0.2,
):
    return (
        f"v{int(vocab_size)}_b1_{float(beta1):.4f}_b2_{float(beta2):.4f}"
        f"_b3_{float(beta3):.4f}_b4_{float(beta4):.4f}"
        f"_gt_{float(gumbel_tau):.4f}_tm_{float(tau_min):.4f}"
    )


print(f"[USTv2] Sensitivity sweep models={target_models}, datasets={target_datasets}")

for dataset_name in target_datasets:
    current_processed_dir = os.path.join(BASE_DATA_DIR, f"Processed_{dataset_name}")
    if not os.path.isdir(current_processed_dir):
        print(f"[Skip] Missing processed dataset directory: {current_processed_dir}")
        continue

    is_cross_domain = dataset_name.startswith("Cross_")
    expected_tau_min = 0.2 if is_cross_domain else 0.1
    expected_gumbel_tau = DEFAULT_EXTRA_CONFIG["gumbel_tau"]

    for model_name in target_models:
        print("\n" + "=" * 70)
        print(f"[Run] {model_name} x {dataset_name}")

        all_results = []
        excel_filepath = os.path.join(RESULT_DIR, f"Sensitivity_{model_name}_{dataset_name}.xlsx")
        run_history = set()

        if os.path.exists(excel_filepath):
            try:
                df_exist = pd.read_excel(excel_filepath)
                required_columns = ["Vocab_Size", "Beta1", "Beta2", "Beta3"]
                if "Beta4" not in df_exist.columns:
                    df_exist["Beta4"] = 0.0
                required_columns.append("Beta4")
                if "Gumbel_Tau" not in df_exist.columns:
                    df_exist["Gumbel_Tau"] = DEFAULT_EXTRA_CONFIG["gumbel_tau"]
                if "Tau_Min" not in df_exist.columns:
                    # Older coarse-search files were produced with tau_min=0.2.
                    # Keep that value explicit so single-domain tau_min=0.1
                    # runs are not incorrectly treated as cached.
                    df_exist["Tau_Min"] = 0.2
                required_columns += ["Gumbel_Tau", "Tau_Min"]

                protocol_mask = df_exist["Gumbel_Tau"].apply(
                    lambda value: values_close(value, expected_gumbel_tau)
                ) & df_exist["Tau_Min"].apply(
                    lambda value: values_close(value, expected_tau_min)
                )
                dropped_rows = int((~protocol_mask).sum())
                if dropped_rows:
                    print(
                        f"[Info] Drop {dropped_rows} stale coarse row(s) for "
                        f"{dataset_name} x {model_name} because they use an old "
                        f"(gumbel_tau, tau_min) protocol."
                    )
                df_exist = df_exist[protocol_mask].copy()

                df_exist = df_exist.drop_duplicates(subset=required_columns, keep="last")
                for _, row in df_exist.iterrows():
                    key = get_cache_key(
                        row["Vocab_Size"],
                        row["Beta1"],
                        row.get("Beta2", 0.0),
                        row["Beta3"],
                        row.get("Beta4", 0.0),
                        row.get("Gumbel_Tau", expected_gumbel_tau),
                        row.get("Tau_Min", expected_tau_min),
                    )
                    run_history.add(key)
                    all_results.append(row.to_dict())

                df_exist.to_excel(excel_filepath, index=False)
            except Exception as exc:
                print(f"[Warn] Failed to restore previous history: {exc}")
                gc.collect()
                torch.cuda.empty_cache()

        keys = ["vocab_size", "beta1", "beta3"]
        lists = [param_grid[key] for key in keys]
        keys.append("beta2")
        lists.append(cross_domain_beta2_list if is_cross_domain else [0.0])

        if is_v2_model_name(model_name):
            keys.append("beta4")
            lists.append(beta4_list)

        combinations = list(itertools.product(*lists))
        print(f"[USTv2] Total parameter combinations: {len(combinations)}")
        fixed_tau_min = expected_tau_min

        for combo in combinations:
            current_params = dict(zip(keys, combo))
            combo_key = get_cache_key(
                current_params["vocab_size"],
                current_params["beta1"],
                current_params["beta2"],
                current_params["beta3"],
                current_params.get("beta4", 0.0),
                DEFAULT_EXTRA_CONFIG["gumbel_tau"],
                fixed_tau_min,
            )
            if combo_key in run_history:
                print(f"[Skip] Existing combination: {current_params}")
                continue

            print(f"[Test] {current_params}")

            if "Elec" in dataset_name or "Tenrec" in dataset_name:
                cur_train_batch, cur_eval_batch, cur_worker = 512, 64, 0
            else:
                cur_train_batch, cur_eval_batch, cur_worker = 2048, 256, 4

            dynamic_config = {
                "dataset": dataset_name,
                "data_path": current_processed_dir,
                "show_progress": False,
                "use_ust": True,
                "train_batch_size": cur_train_batch,
                "eval_batch_size": cur_eval_batch,
                "MAX_ITEM_LIST_LENGTH": 50,
                "worker": cur_worker,
                "checkpoint_dir": resolve_checkpoint_dir(PROJECT_ROOT),
                "save_dataset": False,
                "save_dataloaders": False,
                "saved": True,
                **DEFAULT_EXTRA_CONFIG,
                "tau_min": fixed_tau_min,
                **current_params,
            }

            try:
                model_class = MODEL_DICT[model_name]
                result = run_custom_model(
                    model_class=model_class,
                    dataset_name=dataset_name,
                    config_dict=dynamic_config,
                    config_file_list=get_model_config_files(model_name),
                )

                test_res = result["test_result"]
                family_name, backbone_name = parse_model_metadata(model_name)
                res_dict = {
                    "Model": model_name,
                    "Dataset": dataset_name,
                    "Family": family_name,
                    "Backbone": backbone_name,
                    "Vocab_Size": current_params["vocab_size"],
                    "Beta1": current_params["beta1"],
                    "Beta2": current_params["beta2"],
                    "Beta3": current_params["beta3"],
                    "Beta4": current_params.get("beta4", 0.0),
                    "Gumbel_Tau": dynamic_config["gumbel_tau"],
                    "Tau_Min": dynamic_config["tau_min"],
                }
                res_dict.update(test_res)

                all_results.append(res_dict)
                df = pd.DataFrame(all_results)
                ordered_cols = [
                    "Model",
                    "Dataset",
                    "Family",
                    "Backbone",
                    "Vocab_Size",
                    "Beta1",
                    "Beta2",
                    "Beta3",
                    "Beta4",
                    "Gumbel_Tau",
                    "Tau_Min",
                ]
                ordered_cols += [column for column in df.columns if column not in ordered_cols]
                df[ordered_cols].to_excel(excel_filepath, index=False)
            except Exception as exc:
                print(f"[Error] Combination failed {current_params}: {exc}")
            finally:
                gc.collect()
                torch.cuda.empty_cache()

print(f"\n[Done] Sensitivity sweep finished. Results saved under {RESULT_DIR}")
