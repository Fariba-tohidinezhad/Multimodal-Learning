import os
import json
import random
import numpy as np
import torch
from typing import Tuple
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault
from GIST import GISTDataset

DEFAULT_SPLITS_DIR = "splits"


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _pooled_split_cache_path(fold: int, n_splits: int, seed: int, val_frac: float) -> str:
    os.makedirs(DEFAULT_SPLITS_DIR, exist_ok=True)
    fname = f"pooledCV_balanced_fold-{fold}_of-{n_splits}_seed-{seed}_valfrac-{int(val_frac*100)}.json"
    return os.path.join(DEFAULT_SPLITS_DIR, fname)


def _load_cached_split(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _save_cached_split(path: str, train_ids, val_ids, test_ids):
    payload = {
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
        "test_ids": list(test_ids),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def gist_dataloaders_pooled(
    data_dir_img: str,
    data_dir_seg: str,
    clinical_csv_path: str,
    fold: int,
    n_splits: int = 5,
    largest_tumor: Tuple[int, int, int] = (80, 347, 498),
    num_workers: int = 10,
    batch_size: int = 8,
    val_frac: float = 0.20,
    seed: int = 17,
    save_split: bool = True,
    angle_range: Tuple[float, float] = (-10, 10),
    dropout_p: float = 0.1,
):
    """
    Create train/val/test dataloaders for pooled CV.
    Stratification is based on BOTH outcome (TKIResponse) and center.
    """

    # Build manifest
    full_dataset = GISTDataset.dataset(
        data_dir_img=data_dir_img,
        data_dir_seg=data_dir_seg,
        clinical_csv_path=clinical_csv_path,
        train=False,
        sample_ids=None,
        largest_tumor=largest_tumor,
        angle_range=angle_range,
        dropout_p=dropout_p,
    )

    manifest = [
        (
            s["data.sample_id"],
            s["data.input.clinical.raw.TKIResponse"],
            s["data.input.clinical.raw.Center"],
        )
        for s in full_dataset
    ]

    sids = np.array([m[0] for m in manifest])
    y = np.array([m[1] for m in manifest])
    centers = np.array([m[2] for m in manifest])

    # Combined stratification label: (class, center)
    strat_labels = np.array([f"{yy}_{cc}" for yy, cc in zip(y, centers)])

    # Cache path
    cache_path = _pooled_split_cache_path(fold, n_splits, seed, val_frac) if save_split else None
    split_payload = _load_cached_split(cache_path) if save_split else None

    if split_payload is not None:
        train_ids = split_payload["train_ids"]
        val_ids = split_payload["val_ids"]
        test_ids = split_payload["test_ids"]
    else:
        # Outer folds: stratify by combined outcome+center label
        skf_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        all_folds = list(skf_outer.split(sids, strat_labels))
        trainval_idx, test_idx = all_folds[fold]

        trainval_ids = sids[trainval_idx]
        test_ids = sids[test_idx]
        strat_labels_trainval = strat_labels[trainval_idx]

        # Inner split: stratify by combined label as well
        skf_inner = StratifiedKFold(
            n_splits=int(1 / val_frac),
            shuffle=True,
            random_state=seed,
        )
        inner_train_idx, inner_val_idx = next(skf_inner.split(trainval_ids, strat_labels_trainval))

        train_ids = trainval_ids[inner_train_idx].tolist()
        val_ids = trainval_ids[inner_val_idx].tolist()
        test_ids = test_ids.tolist()

        if save_split and cache_path is not None:
            _save_cached_split(cache_path, train_ids, val_ids, test_ids)

    # --- Build datasets & loaders ---
    train_dataset = GISTDataset.dataset(
        data_dir_img=data_dir_img,
        data_dir_seg=data_dir_seg,
        clinical_csv_path=clinical_csv_path,
        train=True,
        sample_ids=train_ids,
        largest_tumor=largest_tumor,
        angle_range=angle_range,
        dropout_p=dropout_p,
    )

    sampler = BatchSamplerDefault(
        dataset=train_dataset,
        balanced_class_name="data.input.clinical.raw.TKIResponse",
        num_balanced_classes=2,
        batch_size=batch_size,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=sampler,
        collate_fn=CollateDefault(),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn,
    )

    val_dataset = GISTDataset.dataset(
        data_dir_img=data_dir_img,
        data_dir_seg=data_dir_seg,
        clinical_csv_path=clinical_csv_path,
        train=False,
        sample_ids=val_ids,
        largest_tumor=largest_tumor,
        angle_range=angle_range,
        dropout_p=dropout_p,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        collate_fn=CollateDefault(),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn,
    )

    test_dataset = GISTDataset.dataset(
        data_dir_img=data_dir_img,
        data_dir_seg=data_dir_seg,
        clinical_csv_path=clinical_csv_path,
        train=False,
        sample_ids=test_ids,
        largest_tumor=largest_tumor,
        angle_range=angle_range,
        dropout_p=dropout_p,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=CollateDefault(),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn,
    )

    return train_dataloader, val_dataloader, test_dataloader
