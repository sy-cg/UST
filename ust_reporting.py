import os
import re

import pandas as pd


COMPARE_FAMILIES = ("Baseline", "USTv2")
DEFAULT_PREFERRED_METRICS = ["ndcg@10", "recall@10", "mrr@10", "hit@10"]


def parse_model_metadata(model_name):
    model_name = str(model_name)
    if model_name.startswith("USTv2_"):
        return "USTv2", model_name.split("_", 1)[1]
    return "Baseline", model_name


def normalize_variant_name(variant_name):
    if pd.isna(variant_name):
        return variant_name
    return re.sub(r"^\d+_", "", str(variant_name))


def enrich_metadata(df):
    if df is None or df.empty:
        return df

    enriched = df.copy()
    if "Model" in enriched.columns:
        metadata = enriched["Model"].apply(parse_model_metadata)
        enriched["Family"] = [item[0] for item in metadata]
        enriched["Backbone"] = [item[1] for item in metadata]

    if "Variant" in enriched.columns and "Variant_Name" not in enriched.columns:
        enriched["Variant_Name"] = enriched["Variant"].apply(normalize_variant_name)

    return enriched


def collect_excel_rows(directory, exclude_prefixes=None):
    exclude_prefixes = tuple(exclude_prefixes or [])
    if not os.path.isdir(directory):
        return pd.DataFrame()

    frames = []
    for file_name in os.listdir(directory):
        if not file_name.endswith(".xlsx"):
            continue
        if exclude_prefixes and file_name.startswith(exclude_prefixes):
            continue
        file_path = os.path.join(directory, file_name)
        try:
            frame = pd.read_excel(file_path)
            if "Dataset" not in frame.columns and file_name.startswith("Ablation_USTv2_"):
                stem = os.path.splitext(file_name)[0]
                remainder = stem.replace("Ablation_USTv2_", "", 1)
                parts = remainder.split("_", 1)
                if len(parts) == 2:
                    frame["Dataset"] = parts[1]
            frames.append(frame)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    return enrich_metadata(pd.concat(frames, ignore_index=True))


def metric_columns(df):
    if df is None or df.empty:
        return []

    meta_columns = {
        "Model",
        "Dataset",
        "Family",
        "Backbone",
        "Variant",
        "Variant_Name",
        "Tuning_Param",
        "Test_Value",
        "Vocab_Size",
        "Beta1",
        "Beta2",
        "Beta3",
        "Beta4",
        "Gumbel_Tau",
        "Tau_Min",
        "MM_Loss_Weight",
        "Source",
        "Reused_Variant",
        "Status",
        "Error_Message",
        "Source_Dataset",
        "Target_Dataset",
        "Shot_Ratio",
        "Transfer_Protocol",
        "Copied_Param_Count",
        "Skipped_Param_Count",
        "Domain_Setting",
        "Analysis_Tag",
        "use_ust",
        "ust_token_mode",
        "vocab_size",
        "beta1",
        "beta2",
        "beta3",
        "beta4",
        "gumbel_tau",
        "tau_min",
        "mm_loss_weight",
        "fusion_dropout",
    }
    return [
        column
        for column in df.columns
        if column not in meta_columns and pd.api.types.is_numeric_dtype(df[column])
    ]


def find_metric_column(df, preferred_metrics=None):
    preferred_metrics = preferred_metrics or DEFAULT_PREFERRED_METRICS
    lowercase_map = {str(column).lower(): column for column in df.columns}
    for metric_name in preferred_metrics:
        if metric_name.lower() in lowercase_map:
            return lowercase_map[metric_name.lower()]

    numeric_metrics = metric_columns(df)
    return numeric_metrics[0] if numeric_metrics else None


def flatten_pivot_columns(pivot_df):
    flat_columns = []
    for column in pivot_df.columns:
        if isinstance(column, tuple):
            flat_columns.append("_".join(str(part) for part in column if str(part) != ""))
        else:
            flat_columns.append(str(column))
    pivot_df = pivot_df.copy()
    pivot_df.columns = flat_columns
    return pivot_df


def build_family_compare(df, index_columns, aggfunc="first"):
    if df is None or df.empty:
        return pd.DataFrame()

    family_df = df[df["Family"].isin(COMPARE_FAMILIES)].copy()
    if family_df.empty:
        return pd.DataFrame()

    metrics = metric_columns(family_df)
    if not metrics:
        return pd.DataFrame()

    pivot_df = family_df.pivot_table(
        index=index_columns,
        columns="Family",
        values=metrics,
        aggfunc=aggfunc,
    )
    if pivot_df.empty:
        return pd.DataFrame()

    pivot_df = flatten_pivot_columns(pivot_df).reset_index()
    for metric_name in metrics:
        baseline_column = f"{metric_name}_Baseline"
        ustv2_column = f"{metric_name}_USTv2"
        if baseline_column in pivot_df.columns and ustv2_column in pivot_df.columns:
            pivot_df[f"{metric_name}_Delta_USTv2_minus_Baseline"] = (
                pivot_df[ustv2_column] - pivot_df[baseline_column]
            )
    return pivot_df


def build_model_wide(df, index_columns, aggfunc="first"):
    if df is None or df.empty:
        return pd.DataFrame()

    metrics = metric_columns(df)
    if not metrics:
        return pd.DataFrame()

    pivot_df = df.pivot_table(
        index=index_columns,
        columns="Model",
        values=metrics,
        aggfunc=aggfunc,
    )
    if pivot_df.empty:
        return pd.DataFrame()
    return flatten_pivot_columns(pivot_df).reset_index()


def build_best_sensitivity_compare(df):
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    metric_name = find_metric_column(df)
    if metric_name is None:
        return pd.DataFrame(), pd.DataFrame()

    best_rows = (
        df.sort_values(by=metric_name, ascending=False)
        .groupby(["Dataset", "Model"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_rows = enrich_metadata(best_rows)
    compare_df = build_family_compare(best_rows, ["Dataset", "Backbone"], aggfunc="first")
    return best_rows, compare_df


def write_comparison_workbook(file_path, sheet_frames):
    valid_frames = {name: frame for name, frame in sheet_frames.items() if frame is not None}
    if not valid_frames:
        return False

    with pd.ExcelWriter(file_path) as writer:
        wrote_sheet = False
        for sheet_name, frame in valid_frames.items():
            if frame.empty:
                continue
            safe_sheet_name = str(sheet_name)[:31]
            frame.to_excel(writer, sheet_name=safe_sheet_name, index=False)
            wrote_sheet = True

        if not wrote_sheet:
            pd.DataFrame({"Status": ["No data available"]}).to_excel(
                writer, sheet_name="README", index=False
            )
    return True
