#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/pretrain/full/%A_%a.out
#SBATCH --error=logs/pretrain/full/%A_%a.err
#SBATCH --mem=32G
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

python -m pip install --upgrade pip

# Install your repo deps
python -m pip install -r requirements.txt

python -m pip install -e ./torchlight
# Install PyTorch (CUDA 11.8 build)
python -m pip install torch torchvision  torchaudio --index-url https://download.pytorch.org/whl/cu118
python main.py \
 --config config/pretrain_ntu.yaml \
 --work-dir work_dir/pretrain_ntu_subset \
 --device 0