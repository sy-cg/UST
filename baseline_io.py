import json
import math
import os
import shutil

import numpy as np
import pandas as pd


DEFAULT_TOPKS = (5, 10, 20)
TRANSFERABLE_MAX_SEQ_LENGTH = 50
VQREC_MIN_FAISS_TRAIN_ROWS = 9984
ITEM_DOMAIN_FILENAMES = ("ust_item_domain.json", "item_domain.json")


def is_cross_domain_dataset(dataset_name):
    return str(dataset_name).startswith("Cross_")


def load_ust_meta(processed_dir):
    meta_path = os.path.join(processed_dir, "ust_meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def resolve_cross_domain_pair(processed_dir):
    meta_info = load_ust_meta(processed_dir)
    source_domain = meta_info.get("source_domain")
    target_domain = meta_info.get("target_domain")
    if not source_domain or not target_domain:
        raise ValueError(f"Missing source/target domain metadata in {processed_dir}")
    if str(source_domain) == str(target_domain):
        raise ValueError(
            f"Processed dataset at {processed_dir} is not marked as a valid cross-domain pair."
        )
    return str(source_domain), str(target_domain)


def load_sequence_file(file_path):
    records = {}
    if not os.path.exists(file_path):
        return records

    with open(file_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) <= 1:
                continue
            user_id = int(parts[0])
            sequence = [int(item_id) for item_id in parts[1:]]
            records[user_id] = sequence
    return records


def write_sequence_file(file_path, sequence_map):
    with open(file_path, "w", encoding="utf-8") as file_obj:
        for user_id, sequence in sorted(sequence_map.items()):
            line = " ".join([str(int(user_id))] + [str(int(item_id)) for item_id in sequence])
            file_obj.write(line + "\n")


def _copy_if_exists(source_path, target_path):
    if os.path.exists(source_path):
        shutil.copy2(source_path, target_path)


def _atomic_header():
    return "user_id:token\titem_id_list:token_seq\titem_id:token\n"


def _write_atomic_file(file_path, records):
    with open(file_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(_atomic_header())
        for user_token, history_tokens, target_token in records:
            history_text = " ".join(history_tokens)
            file_obj.write(f"{user_token}\t{history_text}\t{target_token}\n")


def _truncate_history_tokens(item_sequence, max_history_length=TRANSFERABLE_MAX_SEQ_LENGTH):
    if max_history_length is None or max_history_length <= 0:
        return list(item_sequence)
    return list(item_sequence)[-int(max_history_length):]


def _truncate_c2dsr_sequence(item_sequence, len_max=TRANSFERABLE_MAX_SEQ_LENGTH):
    max_raw_length = int(len_max) + 1
    if max_raw_length <= 0:
        return list(item_sequence)
    return list(item_sequence)[-max_raw_length:]


def _build_downstream_atomic_records(split_sequences, max_history_length=TRANSFERABLE_MAX_SEQ_LENGTH):
    train_records = []
    valid_records = []
    test_records = []

    for user_id in sorted(split_sequences["train"]):
        train_sequence = list(split_sequences["train"].get(user_id, []))
        valid_sequence = list(split_sequences["valid"].get(user_id, []))
        test_sequence = list(split_sequences["test"].get(user_id, []))
        user_token = str(int(user_id))

        if len(train_sequence) >= 2:
            history_tokens = _truncate_history_tokens(
                [str(int(item_id)) for item_id in train_sequence[:-1]],
                max_history_length=max_history_length,
            )
            train_records.append(
                (
                    user_token,
                    history_tokens,
                    str(int(train_sequence[-1])),
                )
            )

        if len(valid_sequence) >= 2:
            history_tokens = _truncate_history_tokens(
                [str(int(item_id)) for item_id in valid_sequence[:-1]],
                max_history_length=max_history_length,
            )
            valid_records.append(
                (
                    user_token,
                    history_tokens,
                    str(int(valid_sequence[-1])),
                )
            )

        if len(test_sequence) >= 2:
            history_tokens = _truncate_history_tokens(
                [str(int(item_id)) for item_id in test_sequence[:-1]],
                max_history_length=max_history_length,
            )
            test_records.append(
                (
                    user_token,
                    history_tokens,
                    str(int(test_sequence[-1])),
                )
            )

    return train_records, valid_records, test_records


def _write_binary_feature_copy(feature_array, target_path):
    with open(target_path, "wb") as file_obj:
        for row_index in range(feature_array.shape[0]):
            np.asarray(feature_array[row_index], dtype=np.float32).tofile(file_obj)


def _ensure_feature_binary_in_dirs(source_feature_path, output_file_name, output_dirs):
    feature_array = np.load(source_feature_path, mmap_mode="r")
    written_paths = []
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)
        target_path = os.path.join(output_dir, output_file_name)
        if (not os.path.exists(target_path)) or (
            os.path.getmtime(target_path) < os.path.getmtime(source_feature_path)
        ):
            _write_binary_feature_copy(feature_array, target_path)
        written_paths.append(target_path)
    return written_paths


def _collect_items_from_atomic_records(records):
    item_ids = set()
    for _, history_tokens, target_token in records:
        for token in history_tokens:
            if token:
                item_ids.add(int(token))
        item_ids.add(int(target_token))
    return sorted(item_ids)


def load_split_sequences(processed_dir):
    return {
        "train": load_sequence_file(os.path.join(processed_dir, "train.txt")),
        "valid": load_sequence_file(os.path.join(processed_dir, "val.txt")),
        "test": load_sequence_file(os.path.join(processed_dir, "test.txt")),
    }


def _maybe_copy(source_path, target_path):
    if os.path.exists(source_path):
        shutil.copy2(source_path, target_path)


def export_recbole_fewshot_dataset(processed_dir, dataset_name, export_root, shot_ratio):
    if not (0 < float(shot_ratio) <= 1.0):
        raise ValueError(f"shot_ratio must be in (0, 1], got {shot_ratio}")

    split_sequences = load_split_sequences(processed_dir)
    train_sequences = split_sequences["train"]
    valid_sequences = split_sequences["valid"]
    test_sequences = split_sequences["test"]

    export_dir = os.path.join(
        export_root,
        f"{dataset_name}_fewshot_{int(round(float(shot_ratio) * 100)):02d}pct",
    )
    nested_dataset_dir = os.path.join(export_dir, dataset_name)
    os.makedirs(nested_dataset_dir, exist_ok=True)

    fewshot_train = {}
    fewshot_valid = {}
    fewshot_test = {}
    inter_rows = []

    for user_id, train_sequence in sorted(train_sequences.items()):
        train_sequence = list(train_sequence)
        valid_sequence = list(valid_sequences.get(user_id, []))
        test_sequence = list(test_sequences.get(user_id, []))
        if not train_sequence or len(valid_sequence) <= len(train_sequence):
            continue
        if len(test_sequence) <= len(valid_sequence):
            continue

        keep_length = max(1, int(math.ceil(len(train_sequence) * float(shot_ratio))))
        keep_length = min(keep_length, len(train_sequence))

        kept_train = train_sequence[:keep_length]
        valid_target = int(valid_sequence[-1])
        test_target = int(test_sequence[-1])

        fewshot_train[user_id] = kept_train
        fewshot_valid[user_id] = kept_train + [valid_target]
        fewshot_test[user_id] = kept_train + [valid_target, test_target]

        for timestamp, item_id in enumerate(fewshot_test[user_id], start=1):
            inter_rows.append(
                {
                    "user_id:token": int(user_id),
                    "item_id:token": int(item_id),
                    "timestamp:float": float(timestamp),
                }
            )

    write_sequence_file(os.path.join(export_dir, "train.txt"), fewshot_train)
    write_sequence_file(os.path.join(export_dir, "val.txt"), fewshot_valid)
    write_sequence_file(os.path.join(export_dir, "test.txt"), fewshot_test)

    inter_path = os.path.join(nested_dataset_dir, f"{dataset_name}.inter")
    pd.DataFrame(inter_rows).to_csv(inter_path, sep="\t", index=False)

    for file_name in (
        "item2id.json",
        "user2id.json",
        "item_text.json",
        "text_features.npy",
        "vision_features.npy",
    ):
        _maybe_copy(os.path.join(processed_dir, file_name), os.path.join(export_dir, file_name))

    target_meta = load_ust_meta(processed_dir)
    target_meta["source_domain"] = dataset_name
    target_meta["target_domain"] = dataset_name
    target_meta["fewshot_ratio"] = float(shot_ratio)
    with open(os.path.join(export_dir, "ust_meta.json"), "w", encoding="utf-8") as file_obj:
        json.dump(target_meta, file_obj, ensure_ascii=False, indent=2)

    explicit_mapping = _load_explicit_item_domain(processed_dir)
    if explicit_mapping is not None:
        with open(os.path.join(export_dir, "ust_item_domain.json"), "w", encoding="utf-8") as file_obj:
            json.dump(explicit_mapping, file_obj, ensure_ascii=False, indent=2)

    return export_dir


def export_transferable_downstream_atomic_dataset(processed_dir, dataset_name, export_root):
    split_sequences = load_split_sequences(processed_dir)
    train_records, valid_records, test_records = _build_downstream_atomic_records(split_sequences)

    export_root = os.path.abspath(export_root)
    dataset_dir = os.path.join(export_root, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    _write_atomic_file(os.path.join(dataset_dir, f"{dataset_name}.train.inter"), train_records)
    _write_atomic_file(os.path.join(dataset_dir, f"{dataset_name}.valid.inter"), valid_records)
    _write_atomic_file(os.path.join(dataset_dir, f"{dataset_name}.test.inter"), test_records)

    _, feature_shape = ensure_feat1cls_binary(processed_dir, dataset_name)
    item_mapping_path = os.path.join(processed_dir, "item2id.json")
    if os.path.exists(item_mapping_path):
        with open(item_mapping_path, "r", encoding="utf-8") as file_obj:
            item_mapping = json.load(file_obj)
        expected_feature_rows = len(item_mapping) + 1
        if int(feature_shape[0]) != int(expected_feature_rows):
            raise ValueError(
                "Text feature rows do not match item2id cardinality for "
                f"{dataset_name}: text_features rows={feature_shape[0]}, "
                f"expected={expected_feature_rows}. Rebuild the processed "
                "feature arrays before running transferable baselines."
            )
    source_binary_path = os.path.join(processed_dir, f"{dataset_name}.feat1CLS")
    if not os.path.exists(source_binary_path):
        source_binary_path = os.path.join(processed_dir, dataset_name, f"{dataset_name}.feat1CLS")
    for target_dir in (export_root, dataset_dir):
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(source_binary_path, os.path.join(target_dir, f"{dataset_name}.feat1CLS"))

    for file_name in ("item2id.json", "user2id.json", "item_text.json", "ust_meta.json"):
        _copy_if_exists(
            os.path.join(processed_dir, file_name),
            os.path.join(dataset_dir, file_name),
        )

    return {
        "export_root": export_root,
        "dataset_dir": dataset_dir,
        "dataset_name": dataset_name,
        "train_records": len(train_records),
        "valid_records": len(valid_records),
        "test_records": len(test_records),
    }


def export_unisrec_source_pretrain_dataset(processed_dir, dataset_name, export_root):
    split_sequences = load_split_sequences(processed_dir)
    train_records, _, _ = _build_downstream_atomic_records(split_sequences)

    export_root = os.path.abspath(export_root)
    dataset_dir = os.path.join(export_root, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    prefixed_train_records = []
    for user_token, history_tokens, target_token in train_records:
        prefixed_train_records.append(
            (
                f"0-{user_token}",
                [f"0-{token}" for token in history_tokens],
                f"0-{target_token}",
            )
        )

    _write_atomic_file(
        os.path.join(dataset_dir, f"{dataset_name}.train.inter"),
        prefixed_train_records,
    )

    pt_datasets_paths = []
    for target_dir in (export_root, dataset_dir):
        os.makedirs(target_dir, exist_ok=True)
        pt_datasets_path = os.path.join(target_dir, f"{dataset_name}.pt_datasets")
        with open(pt_datasets_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(f"{dataset_name}\n")
        pt_datasets_paths.append(pt_datasets_path)

    ensure_feat1cls_binary(processed_dir, dataset_name)
    source_binary_path = os.path.join(processed_dir, f"{dataset_name}.feat1CLS")
    if not os.path.exists(source_binary_path):
        source_binary_path = os.path.join(processed_dir, dataset_name, f"{dataset_name}.feat1CLS")
    for target_dir in (export_root, dataset_dir):
        feat1_path = os.path.join(target_dir, f"{dataset_name}.feat1CLS")
        shutil.copy2(source_binary_path, feat1_path)
        # The official UniSRec pre-train pipeline expects an augmented PLM file
        # (``feat2CLS``). Our processed datasets only store one text feature view,
        # so we provide a deterministic fallback by duplicating ``feat1CLS``.
        shutil.copy2(source_binary_path, os.path.join(target_dir, f"{dataset_name}.feat2CLS"))

    item_ids = sorted(
        {
            int(target_token.split("-", 1)[1])
            for _, _, target_token in prefixed_train_records
        }
        | {
            int(history_token.split("-", 1)[1])
            for _, history_tokens, _ in prefixed_train_records
            for history_token in history_tokens
        }
    )

    return {
        "export_root": export_root,
        "dataset_dir": dataset_dir,
        "dataset_name": dataset_name,
        "train_records": len(prefixed_train_records),
        "pt_datasets_paths": pt_datasets_paths,
        "train_item_ids": item_ids,
    }


def export_vqrec_source_pretrain_dataset(processed_dir, dataset_name, export_root):
    export_info = export_unisrec_source_pretrain_dataset(processed_dir, dataset_name, export_root)
    filtered_tokens = [f"0-{int(item_id)}" for item_id in export_info["train_item_ids"]]
    for target_dir in (export_info["export_root"], export_info["dataset_dir"]):
        filtered_id_path = os.path.join(target_dir, f"{dataset_name}.filtered_id")
        with open(filtered_id_path, "w", encoding="utf-8") as file_obj:
            for token in filtered_tokens:
                file_obj.write(token + "\n")
    export_info["filtered_tokens"] = filtered_tokens
    return export_info


def build_eval_records(processed_dir, stage="test"):
    split_sequences = load_split_sequences(processed_dir)
    train_sequences = split_sequences["train"]
    valid_sequences = split_sequences["valid"]
    test_sequences = split_sequences["test"]

    if stage == "valid":
        history_map = train_sequences
        target_map = valid_sequences
    elif stage == "test":
        history_map = valid_sequences
        target_map = test_sequences
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    eval_records = []
    for user_id, target_sequence in sorted(target_map.items()):
        history_sequence = history_map.get(user_id, [])
        if not target_sequence or len(target_sequence) <= len(history_sequence):
            continue
        eval_records.append(
            {
                "user_id": int(user_id),
                "history": list(history_sequence),
                "target": int(target_sequence[-1]),
            }
        )

    return eval_records


def compute_topk_metrics(ranks, topks=DEFAULT_TOPKS):
    metrics = {}
    if not ranks:
        for topk in topks:
            metrics[f"recall@{topk}"] = 0.0
            metrics[f"mrr@{topk}"] = 0.0
            metrics[f"ndcg@{topk}"] = 0.0
            metrics[f"hit@{topk}"] = 0.0
        return metrics

    total = float(len(ranks))
    for topk in topks:
        hit_count = 0.0
        mrr_total = 0.0
        ndcg_total = 0.0
        for rank in ranks:
            if rank <= topk:
                hit_count += 1.0
                mrr_total += 1.0 / rank
                ndcg_total += 1.0 / math.log2(rank + 1.0)
        metrics[f"recall@{topk}"] = round(hit_count / total, 4)
        metrics[f"mrr@{topk}"] = round(mrr_total / total, 4)
        metrics[f"ndcg@{topk}"] = round(ndcg_total / total, 4)
        metrics[f"hit@{topk}"] = round(hit_count / total, 4)
    return metrics


def ensure_feat1cls_binary(processed_dir, dataset_name):
    source_path = os.path.join(processed_dir, "text_features.npy")
    feature_array = np.load(source_path, mmap_mode="r")
    expected_binary_size = int(np.prod(feature_array.shape)) * np.dtype(np.float32).itemsize
    target_paths = [
        os.path.join(processed_dir, f"{dataset_name}.feat1CLS"),
        os.path.join(processed_dir, dataset_name, f"{dataset_name}.feat1CLS"),
    ]

    written_target = target_paths[0]
    for target_path in target_paths:
        target_dir = os.path.dirname(target_path)
        if target_dir and not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        target_exists = os.path.exists(target_path)
        target_size_matches = target_exists and os.path.getsize(target_path) == expected_binary_size
        source_is_newer = target_exists and (
            os.path.getmtime(target_path) < os.path.getmtime(source_path)
        )
        if (not target_exists) or (not target_size_matches) or source_is_newer:
            with open(target_path, "wb") as file_obj:
                for row_index in range(feature_array.shape[0]):
                    np.asarray(feature_array[row_index], dtype=np.float32).tofile(file_obj)
        written_target = target_path

    return written_target, feature_array.shape


def _ensure_vqrec_faiss_train_rows(feature_array, min_rows=VQREC_MIN_FAISS_TRAIN_ROWS):
    feature_array = np.ascontiguousarray(feature_array, dtype=np.float32)
    if feature_array.shape[0] >= int(min_rows):
        return feature_array
    if feature_array.shape[0] == 0:
        raise ValueError("Cannot build a VQ-Rec FAISS index from an empty feature matrix.")

    repeat_factor = int(np.ceil(float(min_rows) / float(feature_array.shape[0])))
    expanded = np.tile(feature_array, (repeat_factor, 1))
    return np.ascontiguousarray(expanded[: int(min_rows)], dtype=np.float32)


def parse_faiss_index(pq_index):
    import faiss

    vt = faiss.downcast_VectorTransform(pq_index.chain.at(0))
    opq_transform = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)

    ivf_index = faiss.downcast_index(pq_index.index)
    invlists = faiss.extract_index_ivf(ivf_index).invlists
    list_size = invlists.list_size(0)
    pq_codes = faiss.rev_swig_ptr(
        invlists.get_codes(0),
        list_size * invlists.code_size,
    ).reshape(-1, invlists.code_size)

    centroid_embeds = faiss.vector_to_array(ivf_index.pq.centroids).reshape(
        ivf_index.pq.M,
        ivf_index.pq.ksub,
        ivf_index.pq.dsub,
    )
    coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
    coarse_embeds = faiss.rev_swig_ptr(
        coarse_quantizer.get_xb(),
        ivf_index.pq.M * ivf_index.pq.dsub,
    ).reshape(-1)
    return pq_codes, centroid_embeds, coarse_embeds, opq_transform


def ensure_vqrec_downstream_index(
    dataset_dir,
    dataset_name,
    index_root,
    index_suffix="OPQ32,IVF1,PQ32x8.strict.index",
    plm_size=768,
):
    import faiss

    index_dir = os.path.join(index_root, dataset_name)
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, f"{dataset_name}.{index_suffix}")

    feat_path = os.path.join(dataset_dir, f"{dataset_name}.feat1CLS")
    loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, plm_size)
    train_inter_path = os.path.join(dataset_dir, f"{dataset_name}.train.inter")
    item_ids = set()
    with open(train_inter_path, "r", encoding="utf-8") as file_obj:
        file_obj.readline()
        for line in file_obj:
            user_token, item_seq_text, target_token = line.strip().split("\t")
            if item_seq_text:
                for token in item_seq_text.split(" "):
                    if token:
                        item_ids.add(int(token))
            item_ids.add(int(target_token))
    filter_id = np.array(sorted(item_ids), dtype=np.int64)
    filtered_feat = loaded_feat[filter_id]
    train_feat = _ensure_vqrec_faiss_train_rows(filtered_feat)

    if os.path.exists(index_path):
        try:
            existing_index = faiss.read_index(index_path)
            existing_codes, _, _, _ = parse_faiss_index(existing_index)
            if existing_codes.shape[0] == filtered_feat.shape[0]:
                return index_path
        except Exception:
            pass
        try:
            os.remove(index_path)
        except OSError:
            pass

    index = faiss.index_factory(plm_size, "OPQ32,IVF1,PQ32x8")
    if not index.is_trained:
        index.train(train_feat)
    index.add(filtered_feat)
    faiss.write_index(index, index_path)
    return index_path


def ensure_vqrec_pretrain_index(
    dataset_dir,
    dataset_name,
    index_root,
    index_suffix="OPQ32,IVF1,PQ32x8.strict.index",
    plm_size=768,
):
    import faiss

    index_dir = os.path.join(index_root, dataset_name)
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, f"{dataset_name}.{index_suffix}")

    feat_path = os.path.join(dataset_dir, f"{dataset_name}.feat1CLS")
    loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, plm_size)
    filtered_id_path = os.path.join(dataset_dir, f"{dataset_name}.filtered_id")
    filter_ids = []
    with open(filtered_id_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            token = line.strip()
            if not token:
                continue
            _, item_id = token.split("-", 1)
            filter_ids.append(int(item_id))
    filtered_feat = loaded_feat[np.array(filter_ids, dtype=np.int64)]
    train_feat = _ensure_vqrec_faiss_train_rows(filtered_feat)

    if os.path.exists(index_path):
        try:
            existing_index = faiss.read_index(index_path)
            existing_codes, _, _, _ = parse_faiss_index(existing_index)
            if existing_codes.shape[0] == filtered_feat.shape[0]:
                return index_path
        except Exception:
            pass
        try:
            os.remove(index_path)
        except OSError:
            pass

    index = faiss.index_factory(plm_size, "OPQ32,IVF1,PQ32x8")
    if not index.is_trained:
        index.train(train_feat)
    index.add(filtered_feat)
    faiss.write_index(index, index_path)
    return index_path


def ensure_vqrec_index(
    processed_dir,
    dataset_name,
    index_root,
    index_suffix="OPQ32,IVF1,PQ32x8.strict.index",
):
    import faiss

    index_dir = os.path.join(index_root, dataset_name)
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, f"{dataset_name}.{index_suffix}")
    if os.path.exists(index_path):
        return index_path

    feature_array = np.load(
        os.path.join(processed_dir, "text_features.npy"), mmap_mode="r"
    )
    train_features = feature_array[1:] if feature_array.shape[0] > 1 else feature_array
    train_features = np.ascontiguousarray(train_features, dtype=np.float32)
    train_features_for_fit = _ensure_vqrec_faiss_train_rows(train_features)
    index = faiss.index_factory(train_features.shape[1], "OPQ32,IVF1,PQ32x8")
    if not index.is_trained:
        index.train(train_features_for_fit)
    faiss.write_index(index, index_path)
    return index_path


def export_bm3_dataset(processed_dir, dataset_name, export_root):
    export_dir = os.path.join(export_root, dataset_name)
    os.makedirs(export_dir, exist_ok=True)

    split_sequences = load_split_sequences(processed_dir)
    inter_path = os.path.join(export_dir, f"{dataset_name}.inter")
    with open(inter_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("user_id\titem_id\tx_label\n")
        for user_id, train_sequence in split_sequences["train"].items():
            for item_id in train_sequence:
                file_obj.write(f"{int(user_id)}\t{int(item_id)}\t0\n")

            valid_sequence = split_sequences["valid"].get(user_id, [])
            if valid_sequence and len(valid_sequence) > len(train_sequence):
                file_obj.write(f"{int(user_id)}\t{int(valid_sequence[-1])}\t1\n")

            test_sequence = split_sequences["test"].get(user_id, [])
            history_length = len(valid_sequence) if valid_sequence else len(train_sequence)
            if test_sequence and len(test_sequence) > history_length:
                file_obj.write(f"{int(user_id)}\t{int(test_sequence[-1])}\t2\n")

    shutil.copy2(os.path.join(processed_dir, "vision_features.npy"), os.path.join(export_dir, "image_feat.npy"))
    shutil.copy2(os.path.join(processed_dir, "text_features.npy"), os.path.join(export_dir, "text_feat.npy"))
    return export_dir


def _load_explicit_item_domain(processed_dir):
    for file_name in ITEM_DOMAIN_FILENAMES:
        mapping_path = os.path.join(processed_dir, file_name)
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as file_obj:
                return json.load(file_obj)
    return None


def build_cross_domain_item_mapping(processed_dir):
    explicit_mapping = _load_explicit_item_domain(processed_dir)
    with open(os.path.join(processed_dir, "item2id.json"), "r", encoding="utf-8") as file_obj:
        item2id = json.load(file_obj)

    if explicit_mapping:
        domain_a_items = []
        domain_b_items = []
        for raw_item, mapped_item in item2id.items():
            domain_value = explicit_mapping.get(raw_item, explicit_mapping.get(str(mapped_item)))
            if int(domain_value) == 0:
                domain_a_items.append(int(mapped_item))
            else:
                domain_b_items.append(int(mapped_item))
    else:
        with open(os.path.join(processed_dir, "ust_meta.json"), "r", encoding="utf-8") as file_obj:
            meta_info = json.load(file_obj)
        if meta_info.get("source_domain") == meta_info.get("target_domain"):
            raise ValueError(
                f"Processed dataset at {processed_dir} is not marked as cross-domain."
            )
        boundary = int(meta_info["domain_boundary"])
        mapped_items = sorted(int(mapped_item) for mapped_item in item2id.values())
        domain_a_items = [item_id for item_id in mapped_items if item_id <= boundary]
        domain_b_items = [item_id for item_id in mapped_items if item_id > boundary]

    domain_a_items = sorted(set(domain_a_items))
    domain_b_items = sorted(set(domain_b_items))
    mapping = {}
    for new_id, raw_item_id in enumerate(domain_a_items):
        mapping[int(raw_item_id)] = int(new_id)
    offset = len(domain_a_items)
    for index, raw_item_id in enumerate(domain_b_items):
        mapping[int(raw_item_id)] = int(offset + index)

    return mapping, domain_a_items, domain_b_items


def prepare_c2dsr_domain_artifacts(processed_dir, dataset_name, export_root):
    export_root = os.path.abspath(export_root)
    path_raw = os.path.join(export_root, "raw", dataset_name)
    path_processed = os.path.join(export_root, dataset_name)
    os.makedirs(path_raw, exist_ok=True)
    os.makedirs(path_processed, exist_ok=True)

    item_mapping, domain_a_items, domain_b_items = build_cross_domain_item_mapping(processed_dir)
    split_sequences = load_split_sequences(processed_dir)

    for split_name, source_name in (("train", "train"), ("val", "valid"), ("test", "test")):
        input_sequences = split_sequences[source_name]
        output_path = os.path.join(path_raw, f"{split_name}_new.txt")
        with open(output_path, "w", encoding="utf-8") as file_obj:
            for user_id, sequence in sorted(input_sequences.items()):
                truncated_sequence = _truncate_c2dsr_sequence(sequence)
                mapped_events = [
                    f"{item_mapping[int(item_id)]}|{index + 1}"
                    for index, item_id in enumerate(truncated_sequence)
                    if int(item_id) in item_mapping
                ]
                line = "\t".join([str(user_id), "0"] + mapped_events)
                file_obj.write(line + "\n")

    with open(os.path.join(path_raw, "items_a.txt"), "w", encoding="utf-8") as file_obj:
        for raw_item_id in domain_a_items:
            file_obj.write(f"{raw_item_id}\n")

    with open(os.path.join(path_raw, "items_b.txt"), "w", encoding="utf-8") as file_obj:
        for raw_item_id in domain_b_items:
            file_obj.write(f"{raw_item_id}\n")

    return {
        "path_raw": path_raw,
        "path_data": path_processed,
        "n_item_a": len(domain_a_items),
        "n_item_b": len(domain_b_items),
    }
