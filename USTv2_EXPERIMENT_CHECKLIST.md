# USTv2 Experiment Execution Checklist

## 0. Current Project Convention

- The project now uses `USTv2` as the only UST family name.
- Main tuning / plotting / final-result scripts now focus on:
  - `USTv2_SASRec`
  - `USTv2_GRU4Rec`
- `USTv2_LightGCN` is kept only in `run_cli_ablation.py` and `run_ablation.sh` as supplemental ablation data.
- Legacy source files `UST_*`, `ust_core.py`, `ust_experiment.yaml`, and `ust_v2_experiment.yaml` have been removed from the source tree.

## 1. Parameter Flow Overview

| Stage | Script | What it does | Main output | Manual backfill target |
| --- | --- | --- | --- | --- |
| Stage 1 | `run_sensitivity.py` | Coarse search on selected datasets for `USTv2_SASRec` and `USTv2_GRU4Rec` | `results_sensitivity/Sensitivity_{model}_{dataset}.xlsx` | Copy the best coarse-search row into `run_all_plots.py -> PLOT_TASKS[*]["args"]` |
| Stage 2 | `run_plot_sensitivity.py` or `run_all_plots.py` | Fine-grained sweep around the coarse-search best config | `results_sensitivity_plot/PlotData_{model}_{dataset}.xlsx` | Copy the final best params into `run_all_baselines.py -> BEST_PARAMS_MAP[dataset][model]` |
| Stage 3 | `run_ablation.sh` or `run_cli_ablation.py` | Run ablations with the final frozen USTv2 params | `results_ablation/Ablation_USTv2_{backbone}_{dataset}.xlsx` | Copy the same final params into each command line in `run_ablation.sh` |
| Stage 4 | `run_all_baselines.py` | Run main tables with baselines plus `USTv2_*` models | `results/*.xlsx` and `results/Final_Summary_Results.xlsx` | No backfill here during runtime; this script consumes `BEST_PARAMS_MAP` |
| Stage 5 | `build_ust_comparison_reports.py` | Summarize baseline / ablation / sensitivity / plot outputs | `results_compare/*.xlsx` and figure PNGs | No parameter backfill needed |

## 2. Stage 1: Coarse Search

Script:
- `run_sensitivity.py`

You edit:
- `target_datasets`
- `param_grid`
- `cross_domain_beta2_list`
- `beta4_list`
- `DEFAULT_EXTRA_CONFIG`

You get:
- One Excel file per dataset-model pair:
  - `results_sensitivity/Sensitivity_USTv2_SASRec_{dataset}.xlsx`
  - `results_sensitivity/Sensitivity_USTv2_GRU4Rec_{dataset}.xlsx`

After Stage 1 finishes:
- For each dataset and each USTv2 model, pick the best row by your target metric, usually `ndcg@10`.
- Manually copy that row into the matching block in `run_all_plots.py -> PLOT_TASKS`.

Backfill fields:
- `vocab_size`
- `beta1`
- `beta2`
- `beta3`
- `beta4`
- `gumbel_tau`
- `tau_min`

Notes:
- `run_sensitivity.py` is the coarse stage, so it keeps `gumbel_tau` fixed at `1.0`.
- `tau_min` is fixed by dataset type: `0.1` for single-domain datasets and `0.2` for cross-domain datasets.
- If you add a new dataset to Stage 1, add the same dataset to `run_all_plots.py` if you want a Stage-2 fine search for it.

## 3. Stage 2: Fine Search and Plot Data

Scripts:
- Batch mode: `run_all_plots.py`
- Single task mode: `run_plot_sensitivity.py`

Recommended usage:
- Use `run_all_plots.py` when you want to run the whole Stage-2 queue.
- Use `run_plot_sensitivity.py` when you want to rerun just one dataset-model pair.

You edit:
- `run_all_plots.py -> PLOT_TASKS[*]["args"]`

Those args should come from:
- The best Stage-1 row in `results_sensitivity/*.xlsx`

You get:
- `results_sensitivity_plot/PlotData_{model}_{dataset}.xlsx`
- `results_compare/USTv2_PlotData_Comparison.xlsx`
- figure PNGs in `results_compare/figures/`

After Stage 2 finishes:
- Pick the final best parameter combination for each dataset-model pair.
- Manually copy it into:
  - `run_all_baselines.py -> BEST_PARAMS_MAP[dataset][model]`
  - `run_ablation.sh` matching command lines

Backfill fields:
- `vocab_size`
- `beta1`
- `beta2`
- `beta3`
- `beta4`
- `gumbel_tau`
- `tau_min`

Current fine-search scope in `run_plot_sensitivity.py`:
- `vocab_size`
- `beta1`
- `beta3`
- `beta4`
- `beta2` for cross-domain datasets only
- `gumbel_tau` for single-domain datasets only

Fixed in Stage 2:
- `tau_min=0.1` for single-domain datasets
- `tau_min=0.2` for cross-domain datasets

Notes:
- `mm_loss_weight` is only meaningful for the supplemental `USTv2_LightGCN` path, not for the main `USTv2_SASRec` / `USTv2_GRU4Rec` path.
- `run_all_plots.py` currently queues the focused fine-search scope: `All_Beauty` and `Cross_Sport_Beauty` for `USTv2_SASRec` and `USTv2_GRU4Rec`.
- If a dataset is missing from `run_all_plots.py`, that dataset will not get a Stage-2 fine search unless you run `run_plot_sensitivity.py` manually.

## 4. Stage 3: Ablation

Scripts:
- Batch mode: `run_ablation.sh`
- Single run mode: `run_cli_ablation.py`

You edit:
- The command lines inside `run_ablation.sh`

Those args should come from:
- The final best Stage-2 params from `results_sensitivity_plot/*.xlsx`

Important:
- `run_cli_ablation.py` is now USTv2-only.
- `LightGCN` is retained here only as supplemental ablation evidence.
- Main ablation backbones should be:
  - `SASRec`
  - `GRU4Rec`

Backfill fields for each command:
- `--dataset`
- `--model`
- `--beta1`
- `--beta2` for cross-domain only
- `--beta3`
- `--beta4`
- `--vocab_size`
- `--gumbel_tau`
- `--tau_min`
- `--mm_loss_weight` only for the supplemental `LightGCN` command if you are using it

## 5. Stage 4: Final Main Results

Script:
- `run_all_baselines.py`

You edit:
- `BEST_PARAMS_MAP`

Those params should come from:
- The final best Stage-2 rows in `results_sensitivity_plot/*.xlsx`

You get:
- One Excel file per dataset-model pair under `results/`
- Final merged file:
  - `results/Final_Summary_Results.xlsx`

Important behavior:
- If a dataset-model pair is missing in `BEST_PARAMS_MAP`, the script falls back to `DEFAULT_V2_PARAMS`.
- The script now prints a warning with the exact key you need to backfill:
  - `BEST_PARAMS_MAP['dataset']['model']`
- If a matching ablation result already exists, `run_all_baselines.py` reuses it instead of retraining:
  - baseline models reuse `1_Vanilla`
  - USTv2 models reuse `Full_UST_Single` or `Full_UST_Cross`
- For USTv2 reuse, the script checks the stored parameter snapshot. Old ablation files without config columns will be skipped and retrained.

Current datasets that you should verify in `BEST_PARAMS_MAP` before the final paper run:
- `All_Beauty`
- `Electronics`
- `Sports_and_Outdoors`
- `Tenrec_SBR`
- `Cross_Sport_Beauty`
- `Cross_Elec_Sport`

## 6. Recommended Execution Order

1. Run `run_sensitivity.py`.
2. Open `results_sensitivity/*.xlsx` and select the best coarse-search row for each dataset-model pair.
3. Backfill those coarse-search best params into `run_all_plots.py -> PLOT_TASKS`.
4. Run `run_all_plots.py` or run `run_plot_sensitivity.py` per task.
5. Open `results_sensitivity_plot/*.xlsx` and select the final best params.
6. Backfill those final best params into:
   - `run_all_baselines.py -> BEST_PARAMS_MAP`
   - `run_ablation.sh` command lines
7. Run `run_ablation.sh`.
8. Run `run_all_baselines.py`.
9. Run `build_ust_comparison_reports.py` if you want refreshed comparison workbooks and figures.

## 7. Practical Rule of Thumb

- `run_sensitivity.py` decides the rough neighborhood.
- `run_all_plots.py` / `run_plot_sensitivity.py` decides the final best config.
- `run_ablation.sh` consumes the final best config.
- `run_all_baselines.py` consumes the same final best config for the paper tables.

If these four places disagree, trust the Stage-2 fine-search result and backfill everything else to match it.
