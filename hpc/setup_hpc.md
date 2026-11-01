# Running SolvIER on an HPC cluster

Written for: whoever deploys this repo on the Unical cluster (or any Slurm site).

The code itself is portable — every path is relative or derived at runtime. Four
things differ from a laptop and need handling: a stale `configs.json`, no network
on compute nodes, the FAISS/CUDA pairing, and an unusable `requirements.txt`.

Anything marked **ADJUST** is site-specific and must be checked against the
cluster; the values here are placeholders.

---

## 1. Trim the upload

The checkout is 2.4 GB, of which 1.3 GB is dead weight:

| file | size | why drop it |
|---|---|---|
| `SolvIER.zip` | 544 MB | an archive of the repo, inside the repo |
| `SolvIER/SolvER/checkpoints/` | 761 MB | trained on an older split; the pipeline retrains anyway |

```bash
rm SolvIER.zip
rm -rf SolvIER/SolvER/checkpoints
```

Both are already in `.gitignore`. Upload with `rsync`, which resumes if the link
drops:

```bash
rsync -avz --progress --exclude-from=.gitignore \
      ~/Desktop/SolvIER2/ HPCunical:~/SolvIER2/
```

`HPCunical` is the alias in `~/.ssh/config` on the laptop (user `u59352`). The
rsync runs **on the laptop**, not inside an ssh session on the cluster — from
the cluster there is no `~/Desktop` and the login host does not resolve.

**ADJUST**: this site has no `/scratch`. Check with `df -h $HOME` and
`ls -d /scratch /work /lustre /data 2>/dev/null` before settling on `$HOME` —
home quotas are usually small and home is often slow or not mounted on compute
nodes. Move the checkout if a larger filesystem exists.

## 2. Build the environment

Do this on a login node.

```bash
### ADJUST: module names come from `module avail`
module load python/3.12 cuda/12.1

python -m venv $HOME/venvs/solvier
source $HOME/venvs/solvier/bin/activate
pip install --upgrade pip

# torch FIRST, from the index matching the cluster's CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r hpc/requirements-hpc.txt
```

**Do not** `pip install -r SolvIER/SolvER/requirements.txt`. It pins
`torch==1.9.0+cu111`, `numpy==1.19.2` and `sklearn==0.0` (a dead stub package) and
will not resolve. `hpc/requirements-hpc.txt` lists the versions this pipeline
actually runs on.

### FAISS

Install it from the cluster's own conda/module system so it matches the node's
CUDA. A blind `pip install faiss-gpu` is the most common way to break this step.

```bash
### ADJUST: if the site provides conda
conda install -c pytorch faiss-gpu    # GPU nodes
conda install -c pytorch faiss-cpu    # CPU-only fallback
```

`SENTI/Code/faiss_utils.py` calls `index_cpu_to_all_gpus` when
`torch.cuda.is_available()`. With `faiss-cpu` on a GPU node it silently falls back
to CPU — correct, just slower. With a mismatched `faiss-gpu` it crashes.

### Not needed

- **apex** — `ditto.py` and `matcher.py` both guard the import and only use it for
  `--fp16`, which this pipeline never passes. The README overstates it.
- **spacy / `en_core_web_lg`** — only imported inside `knowledge.py`, reached when
  `--dk` is set. The pipeline does not set it.

## 3. Pre-download the models

Compute nodes usually have no outbound network. Two models are fetched at
runtime: `distilbert-base-uncased` (Ditto) and `sentence-transformers/LaBSE`
(SENTI, ~1.8 GB).

On a **login node**:

```bash
python hpc/prefetch_models.py --cache-dir $HOME/hf_cache
```

If the login node has no network either, copy your local cache up instead:

```bash
rsync -avz ~/.cache/huggingface/ HPCunical:~/hf_cache/
```

The job scripts then set `HF_HOME=$HF_CACHE` and `HF_HUB_OFFLINE=1`, so a cache
miss fails immediately rather than hanging on a dead connection.

## 4. Submit

```bash
cd ~/SolvIER2/hpc
mkdir -p logs

sbatch submit_dataprep.slurm    # optional: regenerate the data (needs LaBSE)
sbatch submit_solver.slurm      # train once on clean, score all 25 variants
```

Override the checkout location per submission:

```bash
REPO=$HOME/SolvIER2 sbatch submit_solver.slurm
```

**ADJUST** in both scripts: `--partition`, `--account`, the `module load` lines,
and the venv path. Run `sinfo -s` to list partitions.

### Or run it directly

`run_solver.py` is the notebook's logic without Jupyter, so it works under
`srun`, in a shell, or anywhere:

```bash
python hpc/run_solver.py --dry-run                      # list experiments, run nothing
python hpc/run_solver.py --repo $HOME/SolvIER2 --retrain
python hpc/run_solver.py --data-dir /any/other/folder   # read the .txt files elsewhere
python hpc/run_solver.py --mechanisms MCAR --pcts 40    # a single cell of the grid
```

`--dry-run` touches nothing and needs no GPU — use it as the first check after
uploading, to confirm every file resolves.

## 5. What the run does

Train and valid are **always** the clean split (`train_DBLPACM.txt` /
`valid_DBLPACM.txt`). Only the test file varies, so **one** model is trained and
scored against every mechanism, null percentage, variant and seed.

With the defaults that is 25 runs: 1 clean baseline + 3 mechanisms × 4
percentages × 2 variants.

Outputs land in `SolvIER/SolvER/output/`:
- `Results_<DATASET>.csv` — one row per run, with `mechanism` and `null_pct`
  columns; reruns replace the matching row rather than appending duplicates
- `output_<run>.jsonl` — per-pair predictions

## Gotchas

**`configs.json` holds absolute paths.** The copy in the repo points at a laptop
(`/home/tariq/...`). Both `run_solver.py` and `SolvER.ipynb` rewrite it before
every single run, so it self-heals — but a direct `python train_ditto.py` reads it
as-is and will fail. Go through the runner.

**Batch size must not exceed the train set.** `num_steps = (len(train) //
batch_size) * n_epochs`. The 90/5/5 split leaves 10 training rows, so the default
`batch_size=32` gives 0 steps and the LR scheduler never advances.
`run_solver.py` refuses to start and names the value to use; the default here is
8.

**Notebooks depend on their working directory.** `merge_splits.ipynb` resolves
`Path.cwd().parent / "Data"`, so it must run from `Data_preparation/code/`;
`SolvER.ipynb` must run from `SolvIER/SolvER/`. The Slurm script `cd`s first.

**The checkpoint is reused unless you force it.** `run_solver.py` skips training
when `checkpoints/<DATASET>/model.pt` exists. Pass `--retrain` after changing the
data — the submission script already does.
