#!/bin/bash
#SBATCH -M kingspeak
#SBATCH --account=soc-kp
#SBATCH --partition=soc-kp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH -t 24:00:00

#SBATCH --array=0-5
#SBATCH -J A7_scot
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# -------------------------------
# Setup
# -------------------------------
module load python/3.10.3
source ~/machine-teaching-irl/.venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# -------------------------------
# Paths
# -------------------------------
REPO_DIR=$HOME/machine-teaching-irl/mdp
OUT_BASE=/scratch/general/nfs1/$USER/two_stage_runs_mini_lp/${SLURM_ARRAY_JOB_ID}


mkdir -p "$OUT_BASE"
mkdir -p logs

# -------------------------------
# Seed handling
# -------------------------------
# Fixed seed (reproducible across array)
SEED=1377

# If you want per-array randomness later, use:
SEED=$((1377 + SLURM_ARRAY_TASK_ID))

# -------------------------------
# Run
# -------------------------------
cd "$REPO_DIR" || exit 1

echo "Starting task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Seed: ${SEED}"
echo "Output dir: ${OUT_BASE}/run_${SLURM_ARRAY_TASK_ID}"


python two_stage_scot_vs_random_minigrid_lp.py \
  --n_envs 10 \
  --grid_size 8 \
  --gamma 0.99 \
  --state_fraction 0.4 \
  --seed "$SEED" \
  --feedback pairwise estop improvement \
  --total_budget 50000 \
  --alloc_method uniform \
  --epsilon 1e-6 \
  --random_trials 10 \
  --result_dir "${OUT_BASE}/run_${SLURM_ARRAY_TASK_ID}"
