#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/train_FreqMixFormer/HRNet2D/%A_%a.out
#SBATCH --error=logs/train_FreqMixFormer/HRNet2D/%A_%a.err
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

# Always install using the env's python
python -m pip install --upgrade pip

# Install your repo deps
python -m pip install -r requirements.txt

python -m pip install -e ./torchlight
# Install PyTorch (CUDA 11.8 build)
python -m pip install torch torchvision  torchaudio --index-url https://download.pytorch.org/whl/cu118

# Debug: prove which python you're using + that torch imports
which python
python -c "import sys; print(sys.executable)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# python main.py \
#   --config config/mediapipe_heavy.yaml \
#   --work-dir work_dir/sails/mediapipe/heavy_9classes \
#   --device 0


# ############################## COCO 17 ##############################
python main.py \
  --config config/coco_17_2d.yaml \
  --work-dir work_dir/coco17_2d/hrnet \
  --save-epoch 1 \
  --device 0 

# ##################################  60CV  ################################# 

# python main.py \
# --config config/nturgbd-cross-view/joint.yaml \
# --work-dir work_dir/ntu/cview/skfreq_joint \
# --device 2 3


# ##################################  120CSUB  ################################# 
# python main.py \
# --config config/nturgbd120-cross-subject/joint.yaml \
# --work-dir work_dir/ntu120/csub/skfreq_joint  \
# --device 4 5


# ##################################  120CSET  ################################# 
# python main.py \
# --config config/nturgbd120-cross-set/joint.yaml \
# --work-dir work_dir/ntu120/cset/skfreq_joint  \
# --device 6 7

# ##################################  UCLA  #################################     
# python main.py \
# --config config/ucla/default.yaml \
# --work-dir work_dir/ucla/skfreq_joint  \
# --device 0 1 
