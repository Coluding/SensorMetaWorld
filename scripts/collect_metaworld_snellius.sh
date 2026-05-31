#!/bin/bash
#SBATCH --job-name=collect_metaworld
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=72
#SBATCH --time=5:00:00
#SBATCH --output=/home/dpasero/scratch-shared/logs/collect_metaworld_%j.log
#SBATCH --error=/home/dpasero/scratch-shared/logs/collect_metaworld_%j.err

USERNAME=$(whoami)
export MUJOCO_GL=egl

cd SensorMetaWorld/

source .venv/bin/activate

MUJOCO_GL=egl srun python scripts/collect_metaworld.py \
  --cpus 70 \
  --tasks reach-v3 \
  --num_episodes 4096 \
  --policy_mode mixed \
  --expert_noise_min 0.0 \
  --expert_noise_max 0.0 \
  --dataset_path /projects/prjs2070/metaworld_reach.hdf5 \
  --temp_dir /projects/prjs2070/metaworld_reach/temp
