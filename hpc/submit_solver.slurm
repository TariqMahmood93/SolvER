#!/bin/bash
#SBATCH --job-name=solvier
#SBATCH --output=logs/solvier_%j.out
#SBATCH --error=logs/solvier_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
### ADJUST FOR UNICAL: partition/account names are site-specific.
### Run `sinfo -s` to list partitions and uncomment the right one.
###SBATCH --partition=gpu
###SBATCH --account=your_account

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR/..}"          # override: REPO=/scratch/$USER/SolvIER-main sbatch ...
HF_CACHE="${HF_CACHE:-$HOME/hf_cache}"

mkdir -p "$SLURM_SUBMIT_DIR/logs"

### ADJUST FOR UNICAL: replace with the cluster's own module names (`module avail`).
# module purge
# module load python/3.12 cuda/12.1
# source "$HOME/venvs/solvier/bin/activate"

# Models are pre-downloaded by prefetch_models.py on a login node. OFFLINE=1
# makes a cache miss fail immediately instead of hanging on a dead network.
export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "node     : $(hostname)"
echo "repo     : $REPO"
echo "hf cache : $HF_HOME"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

python "$SLURM_SUBMIT_DIR/run_solver.py" \
    --repo "$REPO" \
    --dataset DBLPACM \
    --seeds 584 1576 \
    --mechanisms MCAR MAR MNAR \
    --pcts 5 10 20 30 40 \
    --batch-size 8 \
    --n-epochs 15 \
    --retrain
