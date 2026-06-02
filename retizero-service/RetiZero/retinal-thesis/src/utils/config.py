"""
Configuration management.
Merges base.yaml with experiment-specific overrides, then syncs to W&B.
"""

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base. Override wins on conflicts."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_base_config(config_dir: str = "configs") -> Dict[str, Any]:
    """Load base.yaml configuration."""
    base_path = Path(config_dir) / "base.yaml"
    with open(base_path, "r") as f:
        return yaml.safe_load(f)


def load_experiment_config(
    experiment_id: str,
    config_dir: str = "configs",
) -> Dict[str, Any]:
    """
    Load and merge base config with experiment-specific overrides.

    Scans all YAML files in configs/experiments/ for the experiment_id key.
    """
    base = load_base_config(config_dir)
    experiments_dir = Path(config_dir) / "experiments"

    # Search all experiment files for the given ID
    for yaml_file in experiments_dir.glob("*.yaml"):
        with open(yaml_file, "r") as f:
            # Handle multi-document YAML (separated by ---)
            docs = list(yaml.safe_load_all(f))
            for doc in docs:
                if doc is None:
                    continue
                if experiment_id in doc:
                    overrides = doc[experiment_id]
                    merged = _deep_merge(base, overrides)
                    merged["experiment_id"] = experiment_id
                    return merged

    raise ValueError(
        f"Experiment '{experiment_id}' not found in {experiments_dir}. "
        f"Available files: {[f.name for f in experiments_dir.glob('*.yaml')]}"
    )


def flatten_config(config: Dict, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict for W&B config (dot notation)."""
    flat = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, full_key))
        elif isinstance(value, list):
            flat[full_key] = json.dumps(value)
        else:
            flat[full_key] = value
    return flat


def get_model_config(config: Dict) -> Dict[str, Any]:
    """Extract model-specific config for factory function."""
    model_cfg = config.get("model", {})
    model_cfg["num_classes"] = config["num_classes"]
    # Inject ViT-specific overrides if present
    if "vit" in config:
        model_cfg["vit"] = config["vit"]
    return model_cfg


def get_training_config(config: Dict) -> Dict[str, Any]:
    """Extract training-specific config."""
    train_cfg = copy.deepcopy(config.get("training", {}))
    train_cfg["optimizer"] = config.get("optimizer", {})
    train_cfg["scheduler"] = config.get("scheduler", {})
    train_cfg["loss"] = config.get("loss", {})
    return train_cfg
