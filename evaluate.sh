#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/evaluate_FreqMixFormer/hrnet_2D/9classes/%A_%a.out
#SBATCH --error=logs/evaluate_FreqMixFormer/hrnet_2D/9classes/%A_%a.err
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -euo pipefail

module load miniforge

# IMPORTANT: make "conda activate" work in batch shells
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV=freqmix

# Create env only if it doesn't exist
if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  conda create -y -n "$ENV" python=3.10
fi

conda activate "$ENV"

# Always install using the env's python
python -m pip install --upgrade pip

# Install your repo deps
python -m pip install -r requirements.txt

python -m pip install -e ./torchlight
# Install PyTorch (CUDA 11.8 build)
python -m pip install torch torchvision  torchaudio --index-url https://download.pytorch.org/whl/cu118
# Example: Evaluate the model on NTU60 dataset.
# Change the commands depending on your evaluation

#weights for training full : epoch nb 56
# weights for training heavy : epoch nb 57
## FOR FULL
# python main.py \
# --config work_dir/sails/mediapipe/full/config.yaml \
# --work-dir work_dir/sails/mediapipe/full --phase test --save-score True \
# --weights /orcd/data/satra/001/users/lucie271/models/FreqMixFormer/work_dir/sails/mediapipe/full/runs-56-2240.pt
# --device 0

## FOR HEAVY
# python main.py \
# --config work_dir/sails/mediapipe/full_9classes/config.yaml \
# --work-dir work_dir/sails/mediapipe/full_9classes --phase test --save-score True \
# --weights /orcd/data/satra/001/users/lucie271/models/FreqMixFormer/work_dir/sails/mediapipe/full_9classes/runs-87-174.pt
# --device 0

## FOR FINETUNE
# python main.py \
# --config work_dir/sails/mediapipe/finetune_sails/config.yaml \
# --work-dir work_dir/sails/mediapipe/finetune_sails --phase test --save-score True \
# --weights /orcd/data/satra/001/users/lucie271/models/FreqMixFormer/work_dir/sails/mediapipe/finetune_sails/runs-14-28.pt \
# --device 0

python main.py \
--config work_dir/coco17_2d/hrnet/config.yaml \
--work-dir work_dir/coco17_2d/hrnet --phase test --save-score True \
--weights /orcd/data/satra/001/users/lucie271/models/FreqMixFormer/work_dir/coco17_2d/hrnet/runs-17-238.pt \
--device 0