"""
Feature cache for storing precomputed molecular/protein features.

Uses HDF5 with gzip compression for space-efficient storage. Lookup uses a
sorted uint64 hash index with binary search — no large Python dicts needed.

Storage details:
- Morgan fingerprints: packed bits (uint8, N × n_bits//8) — ~8× smaller than float32
- All other features:  float16 — ~2× smaller than float32
- Index:               sorted uint64 hashes + int32 row mapping (~48 MB per 4M entries)
"""

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uint64_hash(s: str) -> int:
    """Hash a string to uint64 (first 8 bytes of SHA-256)."""
    return struct.unpack("<Q", hashlib.sha256(s.encode()).digest()[:8])[0]


def config_to_cache_filename(config: Dict[str, Any]) -> str:
    """Return a deterministic filename for a given featurizer config."""
    ftype = config.get("type", "unknown")
    if ftype == "morgan_fingerprint":
        r = config.get("radius", 2)
        b = config.get("n_bits", 2048)
        uf = "_feat" if config.get("use_features", False) else ""
        return f"morgan_r{r}_b{b}{uf}.h5"
    elif ftype == "molecular_descriptors":
        n = len(config.get("descriptor_names") or [])
        return f"descriptors_{n}.h5"
    elif ftype == "protein_sequence":
        parts = []
        if config.get("include_composition", True):
            parts.append("comp")
        if config.get("include_properties", True):
            parts.append("prop")
        if config.get("include_dipeptides", False):
            parts.append("dipep")
        return f"protein_seq_{'_'.join(parts)}.h5"
    else:
        h = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        return f"{ftype}_{h}.h5"


# ---------------------------------------------------------------------------
# FeatureCache
# ---------------------------------------------------------------------------

class FeatureCache:
    """
    Disk-backed cache for precomputed molecular/protein features.

    Usage
    -----
    cache = FeatureCache(cache_dir="training_data_full/feature_cache", config=fp_config)
    features, invalid = cache.featurize_with_cache(smiles_list, compute_fn)
    """

    def __init__(self, cache_dir: str, config: Dict[str, Any]):
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.cache_path = self.cache_dir / config_to_cache_filename(config)

        self._is_packed = config.get("type") == "morgan_fingerprint"
        self._n_bits: Optional[int] = config.get("n_bits", 2048) if self._is_packed else None

        # Lazy-loaded index
        self._sorted_hashes: Optional[np.ndarray] = None   # uint64, sorted
        self._sorted_to_row: Optional[np.ndarray] = None   # int32

    # ------------------------------------------------------------------
    # Public metadata
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        return self.cache_path.exists()

    def count(self) -> int:
        """Number of cached entries."""
        if not self.exists():
            return 0
        with h5py.File(self.cache_path, "r") as f:
            return len(f["key_hashes"])

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        with h5py.File(self.cache_path, "r") as f:
            key_hashes = f["key_hashes"][:]  # uint64
        order = np.argsort(key_hashes, kind="stable")
        self._sorted_hashes = key_hashes[order]
        self._sorted_to_row = order.astype(np.int32)

    def _ensure_index(self) -> None:
        if self._sorted_hashes is None and self.exists():
            self._load_index()

    def _invalidate_index(self) -> None:
        self._sorted_hashes = None
        self._sorted_to_row = None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup_rows(self, keys: List[str]) -> np.ndarray:
        """
        Return cache row indices for each key. Returns -1 for misses.

        Parameters
        ----------
        keys : list[str]

        Returns
        -------
        rows : np.ndarray int32, shape (len(keys),)
        """
        self._ensure_index()
        if self._sorted_hashes is None:
            return np.full(len(keys), -1, dtype=np.int32)

        query = np.array([_uint64_hash(k) for k in keys], dtype=np.uint64)
        pos = np.searchsorted(self._sorted_hashes, query)

        # Clamp positions for safe indexing, then verify hash match
        clamped = np.minimum(pos, len(self._sorted_hashes) - 1)
        matched = self._sorted_hashes[clamped] == query
        rows = np.where(matched, self._sorted_to_row[clamped], np.int32(-1))
        return rows.astype(np.int32)

    def retrieve(self, rows: np.ndarray) -> np.ndarray:
        """
        Retrieve float32 features for given row indices.
        Automatically unpacks bits for Morgan fingerprints.
        """
        # HDF5 fancy indexing requires strictly increasing unique indices.
        # np.unique returns sorted unique rows + inverse mapping so that
        # unique_rows[inverse] == rows (handles duplicates too).
        unique_rows, inverse = np.unique(rows, return_inverse=True)
        with h5py.File(self.cache_path, "r") as f:
            raw = f["features"][unique_rows]
        raw = raw[inverse]  # expand back to original order (with duplicates)

        if self._is_packed:
            # (N, n_bits//8) uint8 → (N, n_bits) float32
            unpacked = np.unpackbits(raw, axis=1)
            return unpacked[:, : self._n_bits].astype(np.float32)
        else:
            return raw.astype(np.float32)

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(self, keys: List[str], features: np.ndarray) -> None:
        """
        Append (keys, features) to the HDF5 cache.

        Parameters
        ----------
        keys : list[str]
        features : np.ndarray float32, shape (N, D)
        """
        if len(keys) == 0:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key_hashes = np.array([_uint64_hash(k) for k in keys], dtype=np.uint64)

        if self._is_packed:
            bits = features.astype(np.uint8)
            pad = (8 - self._n_bits % 8) % 8
            if pad:
                bits = np.pad(bits, ((0, 0), (0, pad)))
            stored = np.packbits(bits, axis=1)
        else:
            stored = features.astype(np.float16)

        chunk_rows = min(len(stored), 65536)
        mode = "a" if self.cache_path.exists() else "w"
        with h5py.File(self.cache_path, mode) as f:
            if "key_hashes" not in f:
                f.create_dataset(
                    "key_hashes", data=key_hashes,
                    maxshape=(None,),
                    chunks=(chunk_rows,),
                    compression="gzip", compression_opts=4,
                )
                f.create_dataset(
                    "features", data=stored,
                    maxshape=(None, stored.shape[1]),
                    chunks=(chunk_rows, stored.shape[1]),
                    compression="gzip", compression_opts=4,
                )
                f.attrs["config"] = json.dumps(self.config)
                f.attrs["packed_bits"] = self._is_packed
                if self._n_bits is not None:
                    f.attrs["n_bits"] = self._n_bits
            else:
                n = len(f["key_hashes"])
                m = len(key_hashes)
                f["key_hashes"].resize(n + m, axis=0)
                f["key_hashes"][n:] = key_hashes
                f["features"].resize(n + m, axis=0)
                f["features"][n:] = stored

        self._invalidate_index()

    # ------------------------------------------------------------------
    # High-level interface
    # ------------------------------------------------------------------

    def featurize_with_cache(
        self,
        keys: List[str],
        compute_fn: Callable[[List[str]], Tuple[np.ndarray, List[int]]],
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Return features for all keys, using cache where available.

        Parameters
        ----------
        keys : list[str]
            SMILES strings or protein identifiers.
        compute_fn : callable(keys) -> (np.ndarray, invalid_indices)
            Called only for cache misses.
        show_progress : bool

        Returns
        -------
        features : np.ndarray float32, shape (N, D)
        invalid_indices : list[int]   (indices into *keys* that failed)
        """
        rows = self.lookup_rows(keys)
        miss_mask = rows == -1
        n_miss = int(miss_mask.sum())
        n_hit = len(keys) - n_miss

        # ---- All cached -----------------------------------------------
        if n_miss == 0:
            print(f"   Cache: all {n_hit:,} features loaded from cache ({self.cache_path.name})")
            return self.retrieve(rows), []

        if n_hit > 0:
            print(f"   Cache: {n_hit:,} hits, {n_miss:,} misses — computing misses")

        # ---- Compute misses -------------------------------------------
        miss_indices = np.where(miss_mask)[0]
        miss_keys = [keys[i] for i in miss_indices]
        computed, computed_invalid_local = compute_fn(miss_keys)

        # Store valid entries
        invalid_local_set = set(computed_invalid_local)
        valid_local = [i for i in range(len(miss_keys)) if i not in invalid_local_set]
        if valid_local:
            self.store(
                [miss_keys[i] for i in valid_local],
                computed[valid_local],
            )

        # Map local invalid indices back to original positions
        invalid_indices = [int(miss_indices[i]) for i in computed_invalid_local]

        # ---- Assemble result ------------------------------------------
        n_features = computed.shape[1]
        result = np.zeros((len(keys), n_features), dtype=np.float32)

        # Hits from cache
        if n_hit > 0:
            hit_mask = ~miss_mask
            result[hit_mask] = self.retrieve(rows[hit_mask])

        # Newly computed (fill directly, skip re-reading from disk)
        for local_i, orig_i in enumerate(miss_indices):
            if local_i not in invalid_local_set:
                result[orig_i] = computed[local_i]

        return result, invalid_indices
