#!/bin/bash
#SBATCH -M kingspeak
#SBATCH --account=dbrown
#SBATCH --partition=kingspeak-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH -t 08:00:00

#SBATCH --array=0-9
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
REPO_DIR=$HOME/machine-teaching-irl
CONFIG=$REPO_DIR/configs/A7.yaml
OUT_BASE=/scratch/general/nfs1/$USER/two_stage_runs/A7

mkdir -p $OUT_BASE
mkdir -p logs

# -------------------------------
# Fixed seed or array seed
# -------------------------------
SEED=1377
# If you want array-based seeds later, replace with:
# SEED=$SLURM_ARRAY_TASK_ID

# -------------------------------
# Run
# -------------------------------
cd $REPO_DIR

python two_stage_vs_random.py \
  --n_envs $(yq '.n_envs' $CONFIG) \
  --mdp_size $(yq '.mdp_size' $CONFIG) \
  --feature_dim $(yq '.feature_dim' $CONFIG) \
  --feedback estop \
  --demo_env_fraction $(yq '.demo_env_fraction' $CONFIG) \
  --total_budget $(yq '.total_budget' $CONFIG) \
  --random_trials $(yq '.random_trials' $CONFIG) \
  --samples $(yq '.samples' $CONFIG) \
  --stepsize $(yq '.stepsize' $CONFIG) \
  --beta $(yq '.beta' $CONFIG) \
  --seed $SEED \
  --result_dir $OUT_BASE/run_${SLURM_ARRAY_TASK_ID}