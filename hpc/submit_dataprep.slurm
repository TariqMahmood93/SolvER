#!/bin/bash
#SBATCH --job-name=solvier-prep
#SBATCH --output=logs/prep_%j.out
#SBATCH --error=logs/prep_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
### ADJUST FOR UNICAL: see submit_solver.slurm.
###SBATCH --partition=gpu

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR/..}"
HF_CACHE="${HF_CACHE:-$HOME/hf_cache}"

mkdir -p "$SLURM_SUBMIT_DIR/logs"

### ADJUST FOR UNICAL:
# module purge
# module load python/3.12 cuda/12.1
# source "$HOME/venvs/solvier/bin/activate"

export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# merge_splits.ipynb resolves its paths from the working directory
# (Path.cwd().parent / "Data"), so it must be executed from code/.
cd "$REPO/Data_preparation/code"

jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 \
    merge_splits.ipynb

echo "data prep done - outputs in $REPO/Data_preparation/Data/"
