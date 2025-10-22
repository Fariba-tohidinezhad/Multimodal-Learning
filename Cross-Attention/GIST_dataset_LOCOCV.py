import os
import json
import random
import numpy as np
import torch
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault
from GIST import GISTDataset

# default directory for cached split files
DEFAULT_SPLITS_DIR = "splits"


# Ensure each DataLoader worker has an independent RNG state
def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# Build a deterministic cache-file path for storing/retrieving split IDs
def _split_cache_path(test_center: str, seed: int, val_frac: float) -> str:
    os.makedirs(DEFAULT_SPLITS_DIR, exist_ok=True)
    fname = f"split_test-{test_center}_seed-{seed}_valfrac-{int(val_frac*100)}.json"
    return os.path.join(DEFAULT_SPLITS_DIR, fname)


# Load a cached split JSON {train_ids, val_ids, test_ids} if it exists
def _load_cached_split(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# Save the current split (lists of patient IDs) to JSON for exact reuse later
def _save_cached_split(path: str, train_ids, val_ids, test_ids):
    payload = {
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
        "test_ids": list(test_ids),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# Create train/val/test DataLoaders for LOCO-CV with a per-center 80/20 dev split
def gist_dataloaders(
    data_dir_img: str,
    data_dir_seg: str,
    clinical_csv_path: str,
    test_center: Optional[str] = None,
    patch_size: Tuple[int, int, int] = (8, 16, 16),
    largest_tumor: Tuple[int, int, int] = (83, 274, 301),
    num_workers: int = 10,
    batch_size: int = 8,
    val_frac: float = 0.20,
    seed: int = 17,
    save_split: bool = True,  # if True: load-if-exists else create&save; if False: always recompute
    angle_range: Tuple[float, float] = (-10, 10),
    mask_pad_threshold: float = 0.8,
    dropout_p: float = 0.1,
):
    # Build manifest by iterating the dataset to read metadata (center, label, sample_id)
    full_dataset = GISTDataset.dataset(
        data_dir_img=data_dir_img,
        data_dir_seg=data_dir_seg,
        clinical_csv_path=clinical_csv_path,
        train=False,
        sample_ids=None,
        patch_size=patch_size,
        largest_tumor=largest_tumor,
        angle_range=angle_range,
        mask_pad_threshold=mask_pad_threshold,
        dropout_p=dropout_p,
    )

    # Collect per-sample info
    dev_manifest = []  # list of tuples: (sid, center, label)
    test_ids = []

    for sample in full_dataset:
        center = sample['data.input.clinical.raw.Center']
        sid = sample['data.sample_id']
        label = sample['data.input.clinical.raw.TKIResponse']

        if test_center is not None and center == test_center:
            test_ids.append(sid)
        else:
            dev_manifest.append((sid, center, label))

    # With save_split=True we try to reuse an existing split; otherwise compute fresh every time
    cache_path = _split_cache_path(test_center or "NONE", seed, val_frac) if save_split else None
    split_payload = _load_cached_split(cache_path) if save_split else None

    if split_payload is not None:
        train_ids = split_payload["train_ids"]
        val_ids = split_payload["val_ids"]
        # Sanity: cached test IDs must match current dataset selection
        if set(split_payload["test_ids"]) != set(test_ids):
            raise ValueError(
                "Cached split's test_ids do not match current dataset selection. "
                "Delete the cached file in ./splits or regenerate splits."
            )
    else:
        if len(dev_manifest) == 0:
            raise ValueError("Dev set is empty after removing test_center.")

        # ---- NEW: per-center stratified split (fallback to non-stratified if needed) ----
        # Group sample IDs and labels by center
        centers = sorted({ctr for _, ctr, _ in dev_manifest})
        train_ids, val_ids = [], []

        for ctr in centers:
            # Extract this center's sids and labels
            sids_c = [sid for sid, c, _ in dev_manifest if c == ctr]
            y_c = [lab for _, c, lab in dev_manifest if c == ctr]

            sids_c = np.array(sids_c)
            y_c = np.array(y_c)

            # Try stratified split within this center
            try:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
                tr_idx, va_idx = next(sss.split(np.zeros_like(y_c), y_c))
            except ValueError:
                # Fallback when a class has <2 samples etc.
                ss = ShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
                tr_idx, va_idx = next(ss.split(np.zeros_like(y_c)))

            train_ids.extend(sids_c[tr_idx].tolist())
            val_ids.extend(sids_c[va_idx].tolist())
        # ---- END NEW ----

        if save_split and cache_path is not None:
            _save_cached_split(cache_path, train_ids, val_ids, test_ids)

    # Train dataset (augmentations on via train=True)
    train_dataset = GISTDataset.dataset(
        data_dir_img=data_dir_img,
        data_dir_seg=data_dir_seg,
        clinical_csv_path=clinical_csv_path,
        train=True,
        sample_ids=train_ids,
        patch_size=patch_size,
        largest_tumor=largest_tumor,
        angle_range=angle_range,
        mask_pad_threshold=mask_pad_threshold,
        dropout_p=dropout_p,
    )

    # Balanced-class batch sampler for training
    sampler = BatchSamplerDefault(
        dataset=train_dataset,
        balanced_class_name="data.input.clinical.raw.TKIResponse",
        num_balanced_classes=2,
        batch_size=batch_size
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=sampler,
        collate_fn=CollateDefault(),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn
    )

    # Validation dataset (no augmentation)
    val_dataset = GISTDataset.dataset(
        data_dir_img=data_dir_img,
        data_dir_seg=data_dir_seg,
        clinical_csv_path=clinical_csv_path,
        train=False,
        sample_ids=val_ids,
        patch_size=patch_size,
        largest_tumor=largest_tumor,
        angle_range=angle_range,
        mask_pad_threshold=mask_pad_threshold,
        dropout_p=dropout_p,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        collate_fn=CollateDefault(),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn
    )

    # Test dataset = held-out center (no augmentation)
    test_dataset = GISTDataset.dataset(
        data_dir_img=data_dir_img,
        data_dir_seg=data_dir_seg,
        clinical_csv_path=clinical_csv_path,
        train=False,
        sample_ids=test_ids,
        patch_size=patch_size,
        largest_tumor=largest_tumor,
        angle_range=angle_range,
        mask_pad_threshold=mask_pad_threshold,
        dropout_p=dropout_p,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=CollateDefault(),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn
    )

    return train_dataloader, val_dataloader, test_dataloader
