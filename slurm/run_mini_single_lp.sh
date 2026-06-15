#!/bin/bash
# LavaMiniGrid — single-modality LP experiments
# 4 modalities x 10 seeds = 40 array tasks
# task_id = seed_idx * 4 + modality_idx
#
# Usage:
#   sbatch --account=<acct> --partition=<part> [-M <cluster>] [-t HH:MM:SS] run_mini_single_lp.sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH -t 24:00:00
# Pass --account, --partition, and -M (cluster) via sbatch CLI or submit_all.sh

#SBATCH --array=0-39
#SBATCH -J mini_single_lp
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

module load python/3.10.3
source ~/machine-teaching-irl/.venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

MODALITIES=("demo" "pairwise" "estop" "improvement")
N_MODALITIES=4

SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_MODALITIES ))
MOD_IDX=$(( SLURM_ARRAY_TASK_ID % N_MODALITIES ))

SEED=$(( 1337 + SEED_IDX ))
MODALITY="${MODALITIES[$MOD_IDX]}"

REPO_DIR=$HOME/machine-teaching-irl/mdp
OUT_DIR=/scratch/general/nfs1/$USER/paper_results/mini_single/${MODALITY}/seed_${SEED}

mkdir -p "$OUT_DIR" logs

cd "$REPO_DIR" || exit 1

echo "Task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} modality=${MODALITY}"
echo "Output: ${OUT_DIR}"

python two_stage_scot_vs_random_minigrid_lp.py \
  --n_envs 50 \
  --grid_size 8 \
  --gamma 0.99 \
  --state_fraction 0.4 \
  --feedback "$MODALITY" \
  --total_budget 2000 \
  --random_trials 10 \
  --seed "$SEED" \
  --heldout_frac 0.2 \
  --epsilon 1e-6 \
  --alloc_method uniform \
  --n_jobs "$SLURM_CPUS_PER_TASK" \
  --result_dir "$OUT_DIR"
