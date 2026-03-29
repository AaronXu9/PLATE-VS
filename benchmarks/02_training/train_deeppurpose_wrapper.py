"""DeepPurpose wrapper for affinity-regression benchmarks."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import yaml


sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.metrics import summarize_regression


MODEL_ARCHITECTURE_ENCODINGS = {
    "deepdta": ("CNN", "CNN"),
    "graphdta": ("DGL_GCN", "CNN"),
}


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


def setup_logging(output_dir: str) -> logging.Logger:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("deeppurpose_training")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path / f"deeppurpose_training_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    logger.info(f"Logging to {log_file}")
    return logger


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(subvalue) for subvalue in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(subvalue) for subvalue in value]
    if callable(value):
        return getattr(value, "__name__", str(value))
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def infer_encodings(config: dict) -> tuple[str, str]:
    encodings = config.get("encodings", {})
    drug_encoding = encodings.get("drug")
    target_encoding = encodings.get("target")

    if drug_encoding and target_encoding:
        return drug_encoding, target_encoding

    model_architecture = str(config.get("model_architecture", "GraphDTA")).lower()
    if model_architecture not in MODEL_ARCHITECTURE_ENCODINGS:
        raise ValueError(
            f"Unsupported model_architecture '{config.get('model_architecture')}'. "
            f"Set encodings.drug and encodings.target explicitly if you want a custom combination."
        )
    return MODEL_ARCHITECTURE_ENCODINGS[model_architecture]


def _resolve_path(base_dir: Path, path_str: str | None) -> Path | None:
    if not path_str:
        return None
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def load_protein_references(protein_refs_path: Path | None) -> dict:
    if protein_refs_path is None or not protein_refs_path.exists():
        return {}
    with open(protein_refs_path, "r") as handle:
        return json.load(handle)


def extract_sequence_from_cif(cif_path: Path) -> str:
    try:
        import gemmi
    except ImportError as exc:
        raise ImportError(
            "gemmi is required to extract protein sequences from CIF files. "
            "Install the deep-learning environment from benchmarks/envs/env_deep_learning.yml."
        ) from exc

    structure = gemmi.read_structure(str(cif_path))
    if len(structure) == 0:
        return ""

    model = structure[0]
    for chain in model:
        residues = []
        for residue in chain:
            if residue.entity_type == gemmi.EntityType.Polymer:
                aa = gemmi.find_tabulated_residue(residue.name)
                if aa.is_amino_acid():
                    residues.append(aa.one_letter_code)
        if residues:
            return "".join(residues)

    return ""


def get_protein_sequence(row: pd.Series, protein_refs: dict, refs_base_dir: Path, sequence_cache: dict) -> str:
    uniprot_id = row["uniprot_id"]
    if uniprot_id in sequence_cache:
        return sequence_cache[uniprot_id]

    sequence = protein_refs.get(uniprot_id, {}).get("sequence", "")
    if sequence:
        sequence_cache[uniprot_id] = sequence
        return sequence

    cif_path_str = protein_refs.get(uniprot_id, {}).get("cif_path") or row.get("cif_path")
    cif_path = _resolve_path(refs_base_dir, cif_path_str)
    if cif_path is None or not cif_path.exists():
        remote_sequence = fetch_uniprot_sequence(uniprot_id, sequence_cache)
        return remote_sequence

    sequence = extract_sequence_from_cif(cif_path)
    if not sequence:
        remote_sequence = fetch_uniprot_sequence(uniprot_id, sequence_cache)
        return remote_sequence

    sequence_cache[uniprot_id] = sequence
    return sequence


def fetch_uniprot_sequence(uniprot_id: str, sequence_cache: dict) -> str:
    cache_key = f"uniprot::{uniprot_id}"
    if cache_key in sequence_cache:
        return sequence_cache[cache_key]

    sequence = ""
    try:
        import requests

        response = requests.get(
            f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta",
            timeout=10,
        )
        if response.ok:
            lines = [line.strip() for line in response.text.splitlines() if line and not line.startswith(">")]
            sequence = "".join(lines)
            sequence = re.sub(r"[^A-Z]", "", sequence)
    except Exception:
        sequence = ""

    sequence_cache[cache_key] = sequence
    if sequence:
        sequence_cache[uniprot_id] = sequence
    return sequence


def transform_targets(values: pd.Series, transform: str, unit_assumption: str) -> pd.Series:
    transformed = values.astype(float).copy()
    transform_name = transform.lower()
    unit_name = unit_assumption.lower()

    if transform_name in {"none", "raw"}:
        return transformed

    if transform_name != "pic50":
        raise ValueError(f"Unsupported target transform '{transform}'")

    if unit_name == "nm":
        return pd.Series(-np.log10(transformed.to_numpy() * 1e-9), index=transformed.index)
    if unit_name == "um":
        return pd.Series(-np.log10(transformed.to_numpy() * 1e-6), index=transformed.index)
    if unit_name == "m":
        return pd.Series(-np.log10(transformed.to_numpy()), index=transformed.index)

    raise ValueError(f"Unsupported unit assumption '{unit_assumption}'")


def prepare_regression_frame(registry: pd.DataFrame, similarity_threshold: str, split_name: str, config: dict) -> pd.DataFrame:
    data_config = config.get("data", {})
    frame = registry.copy()
    frame = frame[frame["split"] == split_name]
    frame = frame[frame["similarity_threshold"] == similarity_threshold]

    if data_config.get("only_actives", True):
        frame = frame[frame["is_active"] == True]

    if data_config.get("require_affinity", True):
        frame = frame[frame["affinity_value"].notna()]

    affinity_types = data_config.get("affinity_types", [])
    if affinity_types:
        frame = frame[frame["affinity_type"].isin(affinity_types)]

    frame = frame.dropna(subset=["smiles", "uniprot_id", "affinity_value"]).copy()
    return frame.reset_index(drop=True)


def split_train_val(train_frame: pd.DataFrame, val_fraction: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(train_frame) < 2 or val_fraction <= 0:
        empty_frame = cast(pd.DataFrame, train_frame.iloc[:0].copy().reset_index(drop=True))
        return cast(pd.DataFrame, train_frame.reset_index(drop=True)), empty_frame

    val_size = max(1, int(round(len(train_frame) * val_fraction)))
    val_size = min(val_size, len(train_frame) - 1)
    shuffled = train_frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val_frame = shuffled.iloc[:val_size].copy().reset_index(drop=True)
    train_subframe = shuffled.iloc[val_size:].copy().reset_index(drop=True)
    return train_subframe, val_frame


def limit_frame_size(frame: pd.DataFrame, cap: int | None, random_state: int) -> pd.DataFrame:
    if cap is None or len(frame) <= cap:
        return frame.reset_index(drop=True)
    return frame.sample(n=cap, random_state=random_state).reset_index(drop=True)


def encode_deeppurpose_frame(frame: pd.DataFrame, dp_utils, drug_encoding: str, target_encoding: str):
    return dp_utils.data_process(
        X_drug=frame["smiles"].tolist(),
        X_target=frame["protein_sequence"].tolist(),
        y=frame["target"].astype(float).tolist(),
        drug_encoding=drug_encoding,
        target_encoding=target_encoding,
        split_method="no_split",
    )


def build_deeppurpose_config(config: dict, drug_encoding: str, target_encoding: str, result_folder: Path, dp_utils) -> dict:
    training = config.get("training", {})
    model = config.get("model", {})

    cuda_id = training.get("cuda_id")
    try:
        import torch
        if not torch.cuda.is_available():
            cuda_id = -1
    except Exception:
        cuda_id = -1

    kwargs = {
        "drug_encoding": drug_encoding,
        "target_encoding": target_encoding,
        "result_folder": str(result_folder),
        "batch_size": training.get("batch_size", 128),
        "train_epoch": training.get("epochs", 10),
        "test_every_X_epoch": training.get("test_every_n_epochs", 20),
        "LR": training.get("learning_rate", 1e-4),
        "decay": training.get("weight_decay", 0.0),
        "num_workers": training.get("num_workers", 0),
        "cuda_id": cuda_id,
    }

    optional_keys = [
        "cls_hidden_dims",
        "hidden_dim_drug",
        "hidden_dim_protein",
        "mlp_hidden_dims_drug",
        "mlp_hidden_dims_target",
        "cnn_drug_filters",
        "cnn_drug_kernels",
        "cnn_target_filters",
        "cnn_target_kernels",
        "mpnn_hidden_size",
        "mpnn_depth",
        "gnn_hid_dim_drug",
        "gnn_num_layers",
        "neuralfp_max_degree",
        "neuralfp_predictor_hid_dim",
        "attentivefp_num_timesteps",
    ]
    for key in optional_keys:
        if key in model:
            kwargs[key] = model[key]

    return dp_utils.generate_config(**kwargs)


def evaluate_split(model, encoded_frame, original_frame, split_name: str, output_dir: Path):
    predictions = np.asarray(model.predict(encoded_frame), dtype=float)
    targets = original_frame["target"].to_numpy(dtype=float)
    metrics = summarize_regression(targets, predictions)

    prediction_frame = pd.DataFrame(
        {
            "sample_id": original_frame["sample_id"].tolist(),
            "uniprot_id": original_frame["uniprot_id"].tolist(),
            "smiles": original_frame["smiles"].tolist(),
            "y_true": targets,
            "y_pred": predictions,
        }
    )
    prediction_frame.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)
    return sanitize_for_json(metrics)


def import_deeppurpose():
    try:
        dp_models = importlib.import_module("DeepPurpose.DTI")
        dp_utils = importlib.import_module("DeepPurpose.utils")
    except ImportError as exc:
        raise ImportError(
            "DeepPurpose is not installed in the active environment. "
            "Use benchmarks/envs/env_deep_learning.yml to create the deep-learning environment before training GraphDTA."
        ) from exc
    return dp_models, dp_utils


def train_deeppurpose(
    config_path: str,
    registry_path: str = "../../training_data_full/registry.csv",
    output_dir: str = "./trained_models",
    protein_refs_path: str | None = None,
    quick_test: bool = False,
    test_samples: int = 1000,
):
    logger = setup_logging(output_dir)
    logger.info("=" * 70)
    logger.info("DeepPurpose Regression Training Pipeline")
    logger.info("=" * 70)

    config = load_config(config_path)
    logger.info(f"Loading configuration from {config_path}")

    dp_models, dp_utils = import_deeppurpose()

    registry_file = Path(registry_path).resolve()
    registry = pd.read_csv(registry_file)
    refs_file = Path(protein_refs_path).resolve() if protein_refs_path else registry_file.with_name("protein_references.json")
    protein_refs = load_protein_references(refs_file)
    refs_base_dir = refs_file.parent

    data_config = config.get("data", {})
    target_config = config.get("target", {})
    similarity_threshold = data_config.get("similarity_threshold", "0p7")
    random_state = data_config.get("random_state", 42)
    val_split = data_config.get("val_split", 0.2)
    target_transform = target_config.get("transform", "pIC50")
    unit_assumption = target_config.get("unit_assumption", "nM")

    train_frame = prepare_regression_frame(registry, similarity_threshold, "train", config)
    test_frame = prepare_regression_frame(registry, similarity_threshold, "test", config)
    train_frame, val_frame = split_train_val(train_frame, val_split, random_state)

    if quick_test:
        logger.warning(f"QUICK TEST MODE: limiting train/test sizes with cap={test_samples}")
        train_frame = limit_frame_size(train_frame, test_samples, random_state)
        val_frame = limit_frame_size(val_frame, max(1, test_samples // 5), random_state)
        test_frame = limit_frame_size(test_frame, max(1, test_samples // 2), random_state)

    if len(train_frame) == 0:
        raise ValueError("No regression training rows found after filtering affinity-bearing active compounds.")

    sequence_cache = {}
    split_frames = [("train", train_frame), ("val", val_frame), ("test", test_frame)]
    for split_name, frame in split_frames:
        if len(frame) == 0:
            frame["protein_sequence"] = []
            continue
        frame["protein_sequence"] = frame.apply(
            lambda row: get_protein_sequence(row, protein_refs, refs_base_dir, sequence_cache),
            axis=1,
        )
        frame["target"] = transform_targets(frame["affinity_value"], target_transform, unit_assumption)
        before_rows = len(frame)
        frame["protein_sequence"] = frame["protein_sequence"].replace("", np.nan)
        frame.dropna(subset=["target", "protein_sequence"], inplace=True)
        dropped_rows = before_rows - len(frame)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} {split_name} rows with unresolved protein sequences or targets")
        frame.reset_index(drop=True, inplace=True)

    if len(train_frame) == 0:
        raise ValueError(
            "No training rows remain after protein sequence and target filtering. "
            "Provide protein sequences via protein_references.json or ensure UniProt lookups are reachable."
        )

    drug_encoding, target_encoding = infer_encodings(config)
    run_name = config.get("run_name") or str(config.get("model_architecture", "deeppurpose")).lower()
    feature_type = f"deeppurpose_{drug_encoding.lower()}_{target_encoding.lower()}"
    artifacts_dir = Path(output_dir) / f"{run_name}_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Train rows: {len(train_frame)}")
    logger.info(f"Val rows: {len(val_frame)}")
    logger.info(f"Test rows: {len(test_frame)}")
    logger.info(f"Drug encoding: {drug_encoding}")
    logger.info(f"Target encoding: {target_encoding}")

    encoded_train = encode_deeppurpose_frame(train_frame, dp_utils, drug_encoding, target_encoding)
    encoded_val = encode_deeppurpose_frame(val_frame, dp_utils, drug_encoding, target_encoding) if len(val_frame) > 1 else None
    encoded_test = encode_deeppurpose_frame(test_frame, dp_utils, drug_encoding, target_encoding) if len(test_frame) > 1 else None

    model_config = build_deeppurpose_config(config, drug_encoding, target_encoding, artifacts_dir, dp_utils)
    model = dp_models.model_initialize(**model_config)

    start_time = time.time()
    model.train(encoded_train, encoded_val, encoded_test)
    training_time = time.time() - start_time

    model.save_model(str(artifacts_dir))

    with open(Path(output_dir) / f"{run_name}_config.json", "w") as handle:
        json.dump(sanitize_for_json(config), handle, indent=2)

    with open(Path(output_dir) / f"{run_name}_deeppurpose_config.json", "w") as handle:
        json.dump(sanitize_for_json(model_config), handle, indent=2)

    training_history = {
        "n_train_samples": len(train_frame),
        "n_val_samples": len(val_frame),
        "n_test_samples": len(test_frame),
        "training_time": training_time,
    }

    training_history["train_metrics"] = evaluate_split(model, encoded_train, train_frame, f"{run_name}_train", Path(output_dir))
    if encoded_val is not None:
        training_history["val_metrics"] = evaluate_split(model, encoded_val, val_frame, f"{run_name}_val", Path(output_dir))
    if encoded_test is not None:
        training_history["test_metrics"] = evaluate_split(model, encoded_test, test_frame, f"{run_name}_test", Path(output_dir))

    summary = {
        "task_type": "regression",
        "source": "DeepPurpose",
        "model_type": config.get("model_type", run_name),
        "model_architecture": config.get("model_architecture", run_name),
        "feature_type": feature_type,
        "drug_encoding": drug_encoding,
        "target_encoding": target_encoding,
        "target_transform": target_transform,
        "similarity_threshold": similarity_threshold,
        "training_history": sanitize_for_json(training_history),
        "data_config": sanitize_for_json(data_config),
        "training_config": sanitize_for_json(config.get("training", {})),
        "use_precomputed_split": data_config.get("use_precomputed_split", True),
    }

    summary_path = Path(output_dir) / f"{run_name}_training_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Training complete")
    logger.info(f"Artifacts saved to {artifacts_dir}")
    logger.info(f"Summary saved to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train DeepPurpose models on the PLATE-VS regression track")
    parser.add_argument("--config", type=str, default="../configs/deep_purpose_config.yaml", help="Path to the YAML config")
    parser.add_argument(
        "--registry",
        type=str,
        default="../../training_data_full/registry.csv",
        help="Path to the full registry CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./trained_models",
        help="Directory to save trained models and summaries",
    )
    parser.add_argument(
        "--protein-refs",
        type=str,
        default=None,
        help="Optional path to protein_references.json",
    )
    parser.add_argument("--quick-test", action="store_true", help="Run on a capped subset for fast validation")
    parser.add_argument(
        "--test-samples",
        type=int,
        default=1000,
        help="Sample cap for quick-test mode",
    )
    args = parser.parse_args()
    train_deeppurpose(
        config_path=args.config,
        registry_path=args.registry,
        output_dir=args.output,
        protein_refs_path=args.protein_refs,
        quick_test=args.quick_test,
        test_samples=args.test_samples,
    )


if __name__ == "__main__":
    main()
