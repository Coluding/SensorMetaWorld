#!/bin/bash
#SBATCH --job-name=collect_metaworld
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=72
#SBATCH --time=32:00:00
#SBATCH --output=/home/dpasero/scratch-shared/logs/collect_metaworld_%j.log
#SBATCH --error=/home/dpasero/scratch-shared/logs/collect_metaworld_%j.err

USERNAME=$(whoami)
export HF_TOKEN
export MUJOCO_GL=egl

cd latent_diffusion/
poetry install
poetry run python src/ldwma/datasets/metaworld_helpers/collect_metaworld_avid.py \
    --cpus 50 \
    --num_episodes_expert 50 \
    --num_episodes_random 50 \
    --dataset_path /home/$USERNAME/scratch-shared/metaworld-dataset/metaworld_all_cams_256.hdf5 \
    --temp_dir /home/$USERNAME/scratch-shared/metaworld-dataset/temp/