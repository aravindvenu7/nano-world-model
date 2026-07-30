#!/bin/bash
#SBATCH --job-name=occupancy # create a short name for your job
#SBATCH --qos=normal
#SBATCH --partition=general
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=300G                 # memory (4G per cpu-core is default)
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

: <<'END_COMMENT'

python src/main_occupancy.py \
  dataset=mt50/mt50 \
  model=nanowm_b2 \
  logger.name=tensorboard \
  experiment.evaluation.metrics.evaluate=false \
  experiment.infra.num_workers=8 \
  experiment.training.optimizer.lr=1e-4 \
  experiment.training.max_steps=100000 \
  experiment.training.checkpointing.across_timesteps.save_top_k=1 \
  experiment.training.gradient_clip_start_step=0 \
  +experiment.occupancy.gamma=0.99 \
  +experiment.occupancy.flow_steps=10 \
  +experiment.occupancy.target_tau=0.005 \
  +experiment.occupancy.num_bootstrap_samples=1

python src/sample/extract_rewards_occupancy.py \
  --run-dir results/20260725_161640-NanoWM-B-2-F4S1-mt50 \
  --ckpt-path results/20260725_161640-NanoWM-B-2-F4S1-mt50/checkpoints/latest/latest-epoch=238-step=67000.ckpt \
  --output-dir results/20260725_161640-NanoWM-B-2-F4S1-mt50/ors_rewards_[0.3,0.7] \
  --mc-samples 100 \
  --num-trajectories 3

python src/sample/compute_voc_occupancy.py \
  results/20260725_161640-NanoWM-B-2-F4S1-mt50/ors_rewards_[0.3,0.7]/rewards.csv \
  --alpha 1.0

python src/sample/compute_voc_alpha_sweep_occupancy.py \
  results/20260725_161640-NanoWM-B-2-F4S1-mt50/ors_rewards_[0.3,0.7]/rewards.csv

python src/sample/render_reward_videos_occupancy.py \
  results/20260725_161640-NanoWM-B-2-F4S1-mt50/ors_rewards_[0.3,0.7]_full/rewards.csv

END_COMMENT

python src/sample/extract_rewards_occupancy.py \
  --run-dir results/20260725_161640-NanoWM-B-2-F4S1-mt50 \
  --ckpt-path results/20260725_161640-NanoWM-B-2-F4S1-mt50/checkpoints/latest/latest-epoch=238-step=67000.ckpt \
  --output-dir results/20260725_161640-NanoWM-B-2-F4S1-mt50/ors_rewards_[0.3,0.7]_full \
  --mc-samples 100 \
  --num-trajectories 100

python src/sample/compute_voc_alpha_sweep_occupancy.py \
  results/20260725_161640-NanoWM-B-2-F4S1-mt50/ors_rewards_[0.3,0.7]_full/rewards.csv

python src/sample/extract_rewards_occupancy.py \
  --run-dir results/20260725_161640-NanoWM-B-2-F4S1-mt50 \
  --ckpt-path results/20260725_161640-NanoWM-B-2-F4S1-mt50/checkpoints/latest/latest-epoch=238-step=67000.ckpt \
  --output-dir results/20260725_161640-NanoWM-B-2-F4S1-mt50/ors_rewards_[0.3,0.7]_n=25_full \
  --mc-samples 25 \
  --num-trajectories 100

python src/sample/compute_voc_alpha_sweep_occupancy.py \
  results/20260725_161640-NanoWM-B-2-F4S1-mt50/ors_rewards_[0.3,0.7]_n=25_full/rewards.csv