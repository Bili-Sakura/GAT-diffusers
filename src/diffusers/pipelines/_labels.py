"""Shared ImageNet label helpers for class-conditional pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch


def normalize_id2label(id2label: Optional[Dict[Union[int, str], str]]) -> Dict[int, str]:
    if not id2label:
        return {}
    return {int(key): value for key, value in id2label.items()}


def read_id2label_from_model_index(variant_path: Optional[str]) -> Dict[int, str]:
    if not variant_path:
        return {}
    model_index_path = Path(variant_path).resolve() / "model_index.json"
    if not model_index_path.exists():
        return {}
    raw = json.loads(model_index_path.read_text(encoding="utf-8"))
    id2label = raw.get("id2label")
    if not isinstance(id2label, dict):
        return {}
    return {int(key): value for key, value in id2label.items()}


def build_label2id(id2label: Dict[int, str]) -> Dict[str, int]:
    label2id: Dict[str, int] = {}
    for class_id, value in id2label.items():
        for synonym in value.split(","):
            synonym = synonym.strip()
            if synonym:
                label2id[synonym] = int(class_id)
    return dict(sorted(label2id.items()))


def normalize_class_labels(
    class_labels: Union[int, str, List[Union[int, str]], torch.LongTensor],
    *,
    device: torch.device,
    label2id: Dict[str, int],
) -> torch.LongTensor:
    if torch.is_tensor(class_labels):
        return class_labels.to(device=device, dtype=torch.long).reshape(-1)

    if isinstance(class_labels, int):
        class_label_ids = [class_labels]
    elif isinstance(class_labels, str):
        if not label2id:
            raise ValueError("No English labels loaded. Provide `id2label` in the pipeline config.")
        if class_labels not in label2id:
            raise ValueError(f"Unknown English label: {class_labels}")
        class_label_ids = [label2id[class_labels]]
    elif class_labels and isinstance(class_labels[0], str):
        if not label2id:
            raise ValueError("No English labels loaded. Provide `id2label` in the pipeline config.")
        missing = [item for item in class_labels if item not in label2id]
        if missing:
            raise ValueError(f"Unknown English label(s): {missing}")
        class_label_ids = [label2id[item] for item in class_labels]
    else:
        class_label_ids = list(class_labels)

    return torch.tensor(class_label_ids, device=device, dtype=torch.long).reshape(-1)


def get_label_ids(label: Union[str, List[str]], label2id: Dict[str, int]) -> List[int]:
    labels = [label] if isinstance(label, str) else label
    if not label2id:
        raise ValueError("No English labels loaded. Provide `id2label` in the pipeline config.")
    missing = [item for item in labels if item not in label2id]
    if missing:
        preview = ", ".join(list(label2id.keys())[:8])
        raise ValueError(f"Unknown English label(s): {missing}. Example valid labels: {preview}, ...")
    return [label2id[item] for item in labels]
