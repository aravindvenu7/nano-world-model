#!/bin/bash
#SBATCH --job-name=occupancy # create a short name for your job
#SBATCH --qos=normal
#SBATCH --partition=general
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=750G                 # memory (4G per cpu-core is default)
#SBATCH --gres=gpu:A6000:1             # number of gpus per node
#SBATCH --time=48:00:00           # total run time limit (HH:MM:SS)
#SBATCH --output=output/occupancy-%j.out
#SBATCH --error=error/occupancy-%j.err

source ~/.bashrc 
conda activate nanowm

export HF_HOME=/data/user_data/ndarosa/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export MT50_DATA_ROOT=/data/user_data/ndarosa/datasets/metaworld_mt50
export RESULTS_DIR=/home/ndarosa/nano-world-model/results

python src/main_occupancy.py \
  dataset=mt50/mt50 \
  model=nanowm_b2 \
  logger.name=tensorboard \
  experiment.evaluation.metrics.evaluate=false \
  experiment.infra.num_workers=4 \
  +experiment.occupancy.gamma=0.99 \
  +experiment.occupancy.flow_steps=10 \
  +experiment.occupancy.target_tau=0.005 \
  +experiment.occupancy.num_bootstrap_samples=1