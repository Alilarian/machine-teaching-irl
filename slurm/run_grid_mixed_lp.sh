#!/bin/bash
# GridWorld — mixed-modality LP experiments (demo + one other)
# 3 configs x 10 seeds = 30 array tasks
# task_id = seed_idx * 3 + config_idx
#
# Usage:
#   sbatch --account=<acct> --partition=<part> [-M <cluster>] [-t HH:MM:SS] run_grid_mixed_lp.sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH -t 16:00:00
#SBATCH --mem=96G
# Pass --account, --partition, and -M (cluster) via sbatch CLI or submit_all.sh

#SBATCH --array=0-29
#SBATCH -J grid_mixed_lp
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

module load python/3.10.3
source ~/machine-teaching-irl/.venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# config_idx -> feedback arguments and a short label for the output path
CONFIG_FEEDBACK=("demo pairwise" "demo improvement" "demo estop")
CONFIG_LABEL=("demo_pairwise" "demo_improvement" "demo_estop")
N_CONFIGS=3

SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_CONFIGS ))
CFG_IDX=$(( SLURM_ARRAY_TASK_ID % N_CONFIGS ))

SEED=$(( 1337 + SEED_IDX ))
FEEDBACK="${CONFIG_FEEDBACK[$CFG_IDX]}"
LABEL="${CONFIG_LABEL[$CFG_IDX]}"

REPO_DIR=$HOME/machine-teaching-irl/mdp
OUT_DIR=/scratch/general/nfs1/$USER/paper_results/grid_mixed/${LABEL}/seed_${SEED}

mkdir -p "$OUT_DIR" logs

cd "$REPO_DIR" || exit 1

echo "Task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} feedback=${FEEDBACK}"
echo "Output: ${OUT_DIR}"

# shellcheck disable=SC2086
python two_stage_vs_random_lp.py \
  --n_envs 50 \
  --mdp_size 6 \
  --feature_dim 4 \
  --feedback $FEEDBACK \
  --demo_env_fraction 1.0 \
  --total_budget 10000 \
  --random_trials 10 \
  --seed "$SEED" \
  --heldout-frac 0.2 \
  --lp-epsilon 1e-6 \
  --alloc_method uniform \
  --result_dir "$OUT_DIR"
