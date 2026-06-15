#!/bin/bash
# GridWorld — single-modality LP experiments
# 4 modalities x 10 seeds = 40 array tasks
# task_id = seed_idx * 4 + modality_idx
#SBATCH -M kingspeak
#SBATCH --account=soc-kp
#SBATCH --partition=soc-kp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH -t 16:00:00

#SBATCH --array=0-39
#SBATCH -J grid_single_lp
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
OUT_DIR=/scratch/general/nfs1/$USER/paper_results/grid_single/${MODALITY}/seed_${SEED}

mkdir -p "$OUT_DIR" logs

cd "$REPO_DIR" || exit 1

echo "Task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} modality=${MODALITY}"
echo "Output: ${OUT_DIR}"

python two_stage_vs_random_lp.py \
  --n_envs 50 \
  --mdp_size 6 \
  --feature_dim 2 \
  --feedback "$MODALITY" \
  --demo_env_fraction 1.0 \
  --total_budget 2000 \
  --random_trials 10 \
  --seed "$SEED" \
  --heldout-frac 0.2 \
  --lp-epsilon 1e-6 \
  --alloc_method uniform \
  --result_dir "$OUT_DIR"
