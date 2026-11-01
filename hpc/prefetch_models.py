#!/usr/bin/env python3
"""Download every model the pipeline needs into a local cache directory.

Run this on a login node (which usually has network access), then point the
compute nodes at the same directory with HF_HOME. See setup_hpc.md.

    python prefetch_models.py --cache-dir $HOME/hf_cache
"""
import argparse
import os
import sys
from pathlib import Path

MODELS = {
    "distilbert-base-uncased": "Ditto's language model (train_ditto.py --lm distilbert)",
    "sentence-transformers/LaBSE": "SENTI's embedding model (~1.8 GB)",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME", str(Path.home() / "hf_cache")))
    args = ap.parse_args()

    cache = Path(args.cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ.pop("HF_HUB_OFFLINE", None)  # this script is the one step that needs network
    print(f"HF_HOME = {cache}\n")

    from transformers import AutoModel, AutoTokenizer

    for name, why in MODELS.items():
        print(f"--- {name}\n    {why}", flush=True)
        try:
            if name.startswith("sentence-transformers/"):
                from sentence_transformers import SentenceTransformer
                SentenceTransformer(name)
            else:
                AutoTokenizer.from_pretrained(name)
                AutoModel.from_pretrained(name)
            print("    ok\n", flush=True)
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}\n", file=sys.stderr)
            return 1

    size = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file()) / 1e9
    print(f"cache ready: {cache} ({size:.1f} GB)")
    print("\nAdd to your job script:")
    print(f"    export HF_HOME={cache}")
    print( "    export HF_HUB_OFFLINE=1   # fail loudly on a cache miss instead of hanging")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
