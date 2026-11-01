# faiss_utils.py
# ------------------------------------------------------------
# This module wraps:
#   (1) loading a SentenceTransformer model
#   (2) turning a list of strings into L2-normalised embeddings
#   (3) building / updating a FAISS index using GPU if supported,
#       via `index_cpu_to_all_gpus`, else CPU-only.
# ------------------------------------------------------------

import os

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(embed_model: str = "sentence-transformers/LaBSE") -> SentenceTransformer:
    return SentenceTransformer(embed_model, device=DEVICE)

def embed_list(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 64
) -> np.ndarray:
    # total = len(texts)
    # print(f"[embed_list] Encoding {total} texts in batches of {batch_size}...", flush=True)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
    # print(f"[embed_list] Encoding complete", flush=True)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norms, 1e-12, None)

def faiss_has_gpu() -> bool:
    """True only if this faiss build can actually create a GPU index.

    `hasattr(faiss, "index_cpu_to_all_gpus")` is NOT enough: the pure-python
    wrapper ships in faiss-cpu too, and calling it raises
    `NameError: name 'GpuResourcesVector' is not defined` because the compiled
    GPU symbols are absent. Check for those symbols instead, and let
    FAISS_FORCE_CPU=1 override the whole thing.
    """
    if os.environ.get("FAISS_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return False
    if DEVICE != "cuda":
        return False
    needed = ("StandardGpuResources", "GpuResourcesVector", "index_cpu_to_all_gpus")
    if not all(hasattr(faiss, n) for n in needed):
        return False
    try:
        return faiss.get_num_gpus() > 0
    except Exception:
        return False


def build_vector_DB(emb: np.ndarray) -> faiss.IndexIDMap:
    dim = emb.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)

    idx = None
    if faiss_has_gpu():
        try:
            idx = faiss.IndexIDMap(faiss.index_cpu_to_all_gpus(cpu_index))
            print("Built FAISS GPU index via index_cpu_to_all_gpus", flush=True)
        except Exception as e:
            # a GPU-capable build can still fail at runtime (driver, memory, MPS)
            print(f"FAISS GPU index failed ({type(e).__name__}: {e}); using CPU", flush=True)
            idx = None
    if idx is None:
        idx = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        print("Built FAISS CPU index", flush=True)

    idx.add_with_ids(emb, np.arange(len(emb), dtype=np.int64))
    return idx

def update_vector_DB(idx: faiss.IndexIDMap, emb_row: np.ndarray, ridx: int) -> None:
    # print(f"[update_vector_DB] Adding embedding for row {ridx}", flush=True)
    ids = np.array([ridx], dtype=np.int64)
    idx.add_with_ids(emb_row[np.newaxis], ids)
