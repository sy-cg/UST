#!/bin/sh

set -eu

python run_plot_sensitivity.py --dataset All_Beauty --model USTv2_SASRec --vocab_size 128 --beta1 0.05 --beta3 0.05 --beta4 0.05 --gumbel_tau 1.0
python run_plot_sensitivity.py --dataset All_Beauty --model USTv2_GRU4Rec --vocab_size 256 --beta1 0.01 --beta3 0.05 --beta4 0.1 --gumbel_tau 1.0


python run_plot_sensitivity.py --dataset Cross_Sport_Beauty --model USTv2_SASRec --vocab_size 512 --beta1 0.01 --beta2 0.05 --beta3 0.05 --beta4 0.1 --gumbel_tau 1.0
python run_plot_sensitivity.py --dataset Cross_Sport_Beauty --model USTv2_GRU4Rec --vocab_size 512 --beta1 0.01 --beta2 0.05 --beta3 0.05 --beta4 0.1 --gumbel_tau 1.0


echo ""
echo "All sensitivity commands finished."
