import json
import os
import shutil
import warnings

import numpy as np
import torch
import torch.nn as nn


warnings.filterwarnings("ignore", message="The given NumPy array is not writable")


DEFAULT_CHECKPOINT_SUBDIR = "saved_models"
ITEM_DOMAIN_FILENAMES = ("ust_item_domain.json", "item_domain.json")
UNIFORM_NEG_SAMPLE_ARGS = {
    "distribution": "uniform",
    "sample_num": 1,
    "alpha": 1.0,
    "dynamic": False,
    "candidate_num": 0,
}


def _has_atomic_inter_file(dataset_dir, dataset_name):
    inter_path = os.path.join(dataset_dir, f"{dataset_name}.inter")
    return os.path.isfile(inter_path)


def _find_atomic_inter_file(dataset_name, root_dir):
    for current_root, _, file_names in os.walk(root_dir):
        if f"{dataset_name}.inter" in file_names:
            candidate = os.path.join(current_root, f"{dataset_name}.inter")
            if os.path.isfile(candidate):
                return candidate
    return None


def ensure_recbole_dataset_layout(dataset_name, processed_dir):
    """Ensure RecBole can find ``<dataset>/<dataset>.inter`` under ``processed_dir``."""
    processed_dir = os.path.abspath(os.path.normpath(processed_dir))
    nested_dataset_dir = os.path.join(processed_dir, dataset_name)
    nested_inter_path = os.path.join(nested_dataset_dir, f"{dataset_name}.inter")
    if os.path.isfile(nested_inter_path):
        return nested_dataset_dir, nested_inter_path

    root_inter_path = os.path.join(processed_dir, f"{dataset_name}.inter")
    if os.path.isfile(root_inter_path):
        os.makedirs(nested_dataset_dir, exist_ok=True)
        shutil.copy2(root_inter_path, nested_inter_path)
        return nested_dataset_dir, nested_inter_path

    discovered_inter_path = _find_atomic_inter_file(dataset_name, processed_dir)
    if discovered_inter_path:
        return os.path.dirname(discovered_inter_path), discovered_inter_path

    return nested_dataset_dir, nested_inter_path


def resolve_dataset_layout(dataset_name, data_path):
    """Resolve the processed directory and the parent dir that RecBole expects.

    Project layout:
        Processed_<dataset>/
            text_features.npy
            vision_features.npy
            ust_meta.json
            <dataset>/
                <dataset>.inter

    RecBole will internally append ``dataset`` to ``data_path`` again, so the
    value passed to RecBole must be the parent directory of the atomic files,
    not the dataset directory itself.
    """
    normalized_path = os.path.abspath(os.path.normpath(data_path))

    if os.path.basename(normalized_path) == dataset_name and _has_atomic_inter_file(
        normalized_path, dataset_name
    ):
        processed_dir = os.path.dirname(normalized_path)
        recbole_data_path = processed_dir
        return processed_dir, recbole_data_path

    nested_dataset_dir, nested_inter_path = ensure_recbole_dataset_layout(
        dataset_name, normalized_path
    )
    if os.path.isfile(nested_inter_path):
        processed_dir = normalized_path
        recbole_data_path = normalized_path
        return processed_dir, recbole_data_path

    processed_dir = normalized_path
    recbole_data_path = normalized_path

    return processed_dir, recbole_data_path


def prepare_data_path_config(config_dict, dataset_name):
    runtime_config = dict(config_dict)
    processed_dir, recbole_data_path = resolve_dataset_layout(
        dataset_name, runtime_config["data_path"]
    )
    _, expected_inter_path = ensure_recbole_dataset_layout(dataset_name, processed_dir)
    if not os.path.isfile(expected_inter_path):
        raise FileNotFoundError(
            "RecBole atomic interaction file is missing. "
            f"Expected: {expected_inter_path}"
        )
    runtime_config["processed_dir"] = processed_dir
    runtime_config["data_path"] = recbole_data_path
    return runtime_config


def apply_model_runtime_compat(config_dict, model_identifier):
    """Inject per-model runtime settings required by the current RecBole version.

    In this project, SASRec-style models use CE loss and therefore must disable
    training negative sampling explicitly. LightGCN-style models remain pairwise
    and should keep uniform negative sampling enabled.
    """
    runtime_config = dict(config_dict)
    model_name = getattr(model_identifier, "__name__", str(model_identifier))

    if model_name.endswith("SASRec") or model_name.endswith("GRU4Rec"):
        runtime_config["loss_type"] = "CE"
        runtime_config["train_neg_sample_args"] = None
    elif model_name.endswith("BPR"):
        runtime_config.setdefault("train_neg_sample_args", dict(UNIFORM_NEG_SAMPLE_ARGS))
    elif model_name.endswith("LightGCN"):
        runtime_config["loss_type"] = "BPR"
        runtime_config.setdefault("train_neg_sample_args", dict(UNIFORM_NEG_SAMPLE_ARGS))

    return runtime_config


def get_processed_dir(config):
    processed_dir = config["processed_dir"] if "processed_dir" in config else None
    if processed_dir:
        return os.path.abspath(processed_dir)

    processed_dir, _ = resolve_dataset_layout(config["dataset"], config["data_path"])
    return processed_dir


def resolve_checkpoint_dir(project_root):
    env_checkpoint_dir = os.environ.get("UST_CHECKPOINT_DIR")
    if env_checkpoint_dir:
        return env_checkpoint_dir
    return os.path.join(project_root, DEFAULT_CHECKPOINT_SUBDIR)


def resolve_config_file_list(project_root, config_file_list):
    return [os.path.join(project_root, file_name) for file_name in config_file_list]


def get_model_config_files(model_name):
    if str(model_name).endswith("BPR"):
        return ["bpr_baseline_experiment.yaml"]
    return ["ustv2_experiment.yaml"]


def cleanup_checkpoint_file(trainer):
    checkpoint_path = getattr(trainer, "saved_model_file", None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except OSError:
            pass


def build_frozen_embedding(npy_path, device, chunk_size=10000):
    feature_np = np.load(npy_path, mmap_mode="r")
    embedding = nn.Embedding(feature_np.shape[0], feature_np.shape[1], device=device)

    with torch.no_grad():
        for start in range(0, feature_np.shape[0], chunk_size):
            stop = min(start + chunk_size, feature_np.shape[0])
            chunk = np.asarray(feature_np[start:stop], dtype=np.float32)
            embedding.weight[start:stop].copy_(torch.from_numpy(chunk).to(device))

    embedding.weight.requires_grad = False
    return embedding


def _load_item_domain_mapping(processed_dir):
    for file_name in ITEM_DOMAIN_FILENAMES:
        mapping_path = os.path.join(processed_dir, file_name)
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as file_obj:
                return json.load(file_obj)
    return None


def _normalize_domain_value(domain_value):
    if isinstance(domain_value, bool):
        return int(domain_value)
    if isinstance(domain_value, (int, np.integer)):
        return int(domain_value)
    if isinstance(domain_value, float) and float(domain_value).is_integer():
        return int(domain_value)
    if isinstance(domain_value, str) and domain_value.isdigit():
        return int(domain_value)
    return None


def build_item_domain_tensor(dataset, item_field, processed_dir, device):
    """
    Build domain ids that align with RecBole's internal item ids.

    Priority:
    1. explicit item-domain mapping file.
    2. boundary metadata stored in ust_meta.json.
    3. fallback to all-zero single-domain labels.
    """
    num_items = dataset.num(item_field)
    domain_ids = torch.zeros(num_items, dtype=torch.long, device=device)

    if item_field not in dataset.field2id_token:
        return domain_ids

    raw_tokens = dataset.field2id_token[item_field]
    explicit_mapping = _load_item_domain_mapping(processed_dir)
    if explicit_mapping:
        for internal_id in range(1, min(len(raw_tokens), num_items)):
            token = raw_tokens[internal_id]
            domain_value = explicit_mapping.get(str(token), explicit_mapping.get(token))
            normalized_value = _normalize_domain_value(domain_value)
            if normalized_value is not None:
                domain_ids[internal_id] = normalized_value
        return domain_ids

    meta_path = os.path.join(processed_dir, "ust_meta.json")
    if not os.path.exists(meta_path):
        return domain_ids

    with open(meta_path, "r", encoding="utf-8") as file_obj:
        meta_info = json.load(file_obj)

    if meta_info.get("source_domain") == meta_info.get("target_domain"):
        return domain_ids

    boundary = meta_info.get("domain_boundary")
    normalized_boundary = _normalize_domain_value(boundary)
    if normalized_boundary is None:
        return domain_ids

    for internal_id in range(1, min(len(raw_tokens), num_items)):
        try:
            raw_item_id = int(raw_tokens[internal_id])
        except (TypeError, ValueError):
            continue
        domain_ids[internal_id] = 0 if raw_item_id <= normalized_boundary else 1

    return domain_ids


def pairwise_domain_masks(item_ids, item_domain_ids):
    sample_domains = item_domain_ids[item_ids]
    unique_domains = torch.unique(sample_domains)
    unique_domains = unique_domains[unique_domains >= 0]
    if unique_domains.numel() < 2:
        return None

    reference_domain = unique_domains[0]
    reference_mask = sample_domains == reference_domain
    other_masks = [sample_domains == domain_id for domain_id in unique_domains[1:]]

    return reference_mask, other_masks
