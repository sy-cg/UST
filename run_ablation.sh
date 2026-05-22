#!/bin/sh

set -eu

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "USTv2 Ablation Runner"
echo "Project root: $PROJECT_ROOT"
echo "Tips:"
echo "1. Edit each python run_cli_ablation.py command below as needed."
echo "2. Add # before a block if you want to skip that experiment."
echo "3. LightGCN is kept only as one supplemental USTv2 ablation."
echo "4. Copy the final tuned params here after finishing run_plot_sensitivity.py."
echo "============================================================"

if [ ! -d "$PROJECT_ROOT/data/Processed_All_Beauty" ]; then
    echo "[Error] Missing dataset directory: $PROJECT_ROOT/data/Processed_All_Beauty"
    exit 1
fi

echo ""
echo "[1/3] Running: All_Beauty x SASRec x USTv2"
python run_cli_ablation.py --dataset All_Beauty --model SASRec --beta1 0.05 --beta3 0.05 --beta4 0.05 --vocab_size 128 --gumbel_tau 1.0 --tau_min 0.1

echo ""
echo "[2/3] Running: All_Beauty x GRU4Rec x USTv2"
python run_cli_ablation.py --dataset All_Beauty --model GRU4Rec --beta1 0.01 --beta3 0.05 --beta4 0.1 --vocab_size 256 --gumbel_tau 1.0 --tau_min 0.1

echo ""
echo "[3/3] Running: All_Beauty x LightGCN x USTv2 (supplemental)"
python run_cli_ablation.py --dataset All_Beauty --model LightGCN --beta1 0.05 --beta3 0.1 --beta4 0.05 --vocab_size 128 --gumbel_tau 0.1 --tau_min 0.1 --mm_loss_weight 0.05

# =========================
# Optional templates below. Uncomment to run them.
# =========================

# echo ""
# echo "[5/6] Running: Cross_Sport_Beauty x SASRec x USTv2"
# python run_cli_ablation.py --dataset Cross_Sport_Beauty --model SASRec --beta1 0.01 --beta2 0.05 --beta3 0.05 --beta4 0.1 --vocab_size 512 --gumbel_tau 1.0 --tau_min 0.2

# echo ""
# echo "[6/6] Running: Cross_Sport_Beauty x GRU4Rec x USTv2"
# python run_cli_ablation.py --dataset Cross_Sport_Beauty --model GRU4Rec --beta1 0.01 --beta2 0.05 --beta3 0.05 --beta4 0.1 --vocab_size 512 --gumbel_tau 1.0 --tau_min 0.2

echo ""
echo "All ablation commands finished."
echo "Results directory: $PROJECT_ROOT/results_ablation"
