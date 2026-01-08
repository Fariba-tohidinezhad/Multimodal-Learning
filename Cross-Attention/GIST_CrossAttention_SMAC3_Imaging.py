#!/usr/bin/env python
import sys
import os
sys.path.insert(0, '/gpfs/home4/ftohidinezhad/fuse-med-ml')

import json
import copy
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from collections import OrderedDict, Counter

from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Categorical,
    Float
)
from smac import HyperparameterOptimizationFacade, Scenario

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from fuse.dl.models.model_multihead import ModelMultiHead
from fuse.dl.losses.loss_default import LossDefault
from fuse.eval.metrics.classification.metrics_classification_common import (
    MetricAUCROC, MetricConfusion, MetricBSS, MetricAccuracy
)
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds

import torch.optim as optim
import pytorch_lightning as pl
from fuse.dl.lightning.pl_module import LightningModuleDefault
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault
from GIST import GISTDataset

from HeadMLPClassifier import HeadMLPClassifier

from x_transformers import Encoder, TransformerWrapper

import pandas as pd

# =============================
# Global constants
# =============================

RUNS_DIR_NAME = "runs_smac_imaging_only"
TEST_CENTER = "EMC"

SMAC_OUTPUT_DIR = "smac3_output_imaging_only"
SMAC_RUN_NAME = "gist_imaging_only_smac"

KEY_PROB = "model.prob.TKI_Classification"
KEY_LOGITS = "model.logits.TKI_Classification"
KEY_TARGET = "data.input.clinical.raw.TKIResponse"
KEY_TARGET_F = "data.input.clinical.raw.TKIResponse.f"
KEY_SAMPLE_ID = "data.sample_id"
KEY_CENTER = "data.input.clinical.raw.Center"

REPORT_KEYS = [
    "test_center",
    "patch_size", "emb_dim",
    "lr", "wd",
    "depth_a", "heads_a",
    "depth_b", "heads_b",
    "depth_cross_attn", "heads_cross",
    "mlp_layers",
    "mask_pad_thresh", "clinical_aug", "imaging_aug_deg",
    "batch_size", "num_workers",
]

# =============================
# Utils
# =============================

def seed_everything(seed=17):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ceil_div(a, b):
    return (a + b - 1) // b


def compute_max_num_tokens(largest_tumor: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> int:
    zc = ceil_div(largest_tumor[0], patch_size[0])
    yc = ceil_div(largest_tumor[1], patch_size[1])
    xc = ceil_div(largest_tumor[2], patch_size[2])
    return int(zc * yc * xc)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def config_slug(cfg: Dict[str, Any], max_len: int = 120) -> str:
    import hashlib, re
    ABBR = {
        "patch_size": "ps", "emb_dim": "ed", "lr": "lr", "wd": "wd",
        "depth_a": "da", "heads_a": "ha", "depth_b": "db", "heads_b": "hb",
        "depth_cross_attn": "dca", "heads_cross": "hc",
        "mlp_layers": "mlp", "mask_pad_thresh": "mpt",
        "clinical_aug": "ca", "imaging_aug_deg": "ia",
    }
    ORDER = [
        "patch_size","emb_dim","lr","wd",
        "depth_a","heads_a","depth_b","heads_b","depth_cross_attn","heads_cross",
        "mlp_layers","mask_pad_thresh","clinical_aug","imaging_aug_deg"
    ]
    def fmt_lr(x: float) -> str:
        if float(x) == 0.0:
            return "0"
        s = np.format_float_scientific(float(x), precision=0, unique=True, exp_digits=1)
        return s.replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")
    def fmt_ps(ps) -> str:
        return "x".join(str(int(v)) for v in ps)
    def fmt_mlp(v: str) -> str:
        return "S" if str(v).lower().startswith("s") else "D"
    def fmt_pct10(v: float) -> str:
        return f"{int(round(float(v)*10)):02d}"
    def clean(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9\-x]", "", s)

    parts = []
    for k in ORDER:
        if k not in cfg:
            continue
        ab = ABBR[k]
        v = cfg[k]
        if k == "patch_size":
            val = fmt_ps(v)
        elif k in ("emb_dim","depth_a","heads_a","depth_b","heads_b","depth_cross_attn","heads_cross","imaging_aug_deg"):
            val = str(int(v))
        elif k in ("lr","wd"):
            val = fmt_lr(v)
        elif k == "mlp_layers":
            val = fmt_mlp(v)
        elif k in ("mask_pad_thresh","clinical_aug"):
            val = fmt_pct10(v)
        else:
            val = str(v)
        parts.append(f"{ab}{clean(val)}")
    slug = "-".join(parts)
    if len(slug) > max_len:
        cfg_subset = {k: cfg.get(k) for k in ORDER if k in cfg}
        h = hashlib.sha1(json.dumps(cfg_subset, sort_keys=True, default=str).encode()).hexdigest()[:8]
        slug = slug[: max_len - (len(h) + 2)] + "-h" + h
    return slug


# =============================
# Imaging-only backbone
# =============================

class ImagingTransformerEncoder(nn.Module):
    """
    Imaging-only backbone using TransformerWrapper + Encoder on tumor patch tokens.

    Inputs via ModelMultiHead.backbone_args:
        xb           <- batch_dict['data.input.img.tumor3d.patches']   [B, N_p, patch_dim]
        embed_mask_b <- batch_dict['model.embed_mask_b']               [B, N_p] (bool)

    Output:
        [B, N_p, D] token embeddings
    """
    def __init__(
        self,
        emb_dim: int,
        depth_b: int,
        heads_b: int,
        patch_dim: int,
        max_seq_len_b: int,
    ):
        super().__init__()

        token_emb = nn.Linear(patch_dim, emb_dim)

        kwargs_wrapper_b = {
            'token_emb': token_emb,
            'use_abs_pos_emb': False,
            'emb_dropout': 0.1,
            'post_emb_norm': True,
            'return_only_embed': True,
        }
        kwargs_encoder_b = {
            'attn_dropout': 0.1,
            'ff_dropout': 0.1,
            'ff_glu': False,
            'use_rmsnorm': True,
            'rotary_pos_emb': True,
        }

        self.enc_b = TransformerWrapper(
            num_tokens=None,
            max_seq_len=max_seq_len_b,
            **kwargs_wrapper_b,
            attn_layers=Encoder(
                dim=emb_dim,
                depth=depth_b,
                heads=heads_b,
                **kwargs_encoder_b,
            ),
        )

    def forward(
        self,
        xb: torch.Tensor,
        embed_mask_b: torch.Tensor,
    ) -> torch.Tensor:
        enc_xb = self.enc_b(
            xb,
            return_embeddings=True,
            mask=embed_mask_b
        )
        return enc_xb


def build_model(
    cfg: Dict[str, Any],
    patch_dim: int,
    max_num_tokens_b: int,
) -> ModelMultiHead:
    """
    Imaging-only model:
    - ImagingTransformerEncoder backbone (sequence of CT patch tokens)
    - Attention pooling + MLP classifier head
    """
    if cfg.get("mlp_layers", "single") == "double":
        layers_desc = (cfg["emb_dim"], max(1, cfg["emb_dim"] // 2))
    else:
        layers_desc = (cfg["emb_dim"],)

    backbone = ImagingTransformerEncoder(
        emb_dim=cfg["emb_dim"],
        depth_b=cfg.get("depth_b", 2),
        heads_b=cfg.get("heads_b", 4),
        patch_dim=patch_dim,
        max_seq_len_b=max_num_tokens_b,
    )

    model = ModelMultiHead(
        backbone=backbone,
        key_out_features='model.backbone_features',
        backbone_args=[
            'data.input.img.tumor3d.patches',  # xb
            'model.embed_mask_b',             # mask for xb
        ],
        heads=[
            HeadMLPClassifier(
                input_key='model.backbone_features',
                prob_key=KEY_PROB,
                logits_key=KEY_LOGITS,
                in_ch=cfg["emb_dim"],
                num_classes=1,
                layers_description=layers_desc,
                dropout_rate=0.1,
                pooling="attention",
            )
        ],
    )
    return model


def make_training_elements():
    losses = {
        "TKI_cls_loss": LossDefault(
            pred=KEY_LOGITS,
            target=KEY_TARGET_F,
            callable=lambda pred, target: F.binary_cross_entropy_with_logits(pred, target.unsqueeze(1)),
            weight=1.0
        )
    }
    confusion_metrics = ['sensitivity', 'specificity', 'ppv', 'f1']
    common_metrics = OrderedDict([
        ("auc", MetricAUCROC(pred=KEY_PROB, target=KEY_TARGET)),
        ("apply_thresh", MetricApplyThresholds(pred=KEY_PROB, operation_point=0.5)),
        ("confusion", MetricConfusion(
            pred="results:metrics.apply_thresh.cls_pred",
            target=KEY_TARGET,
            metrics=confusion_metrics
        )),
        ("accuracy", MetricAccuracy(
            pred="results:metrics.apply_thresh.cls_pred",
            target=KEY_TARGET
        )),
        ("bss", MetricBSS(pred=KEY_PROB, target=KEY_TARGET)),
    ])
    train_metrics = common_metrics.copy()
    validation_metrics = copy.deepcopy(common_metrics)
    best_epoch_source = dict(monitor="validation.metrics.auc", mode="max")
    return losses, train_metrics, validation_metrics, best_epoch_source


# =============================
# Data helpers for SMAC CV
# =============================

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_dev_test_manifest(
    data_paths: Dict[str, str],
    patch_size: Tuple[int, int, int],
    largest_tumor: Tuple[int, int, int],
) -> Tuple[List[Tuple[str, str, int]], List[str]]:
    """
    Build dev_manifest = list of (sid, center, label) for centers != TEST_CENTER,
    and test_ids = list of sids for TEST_CENTER.
    """
    full_dataset = GISTDataset.dataset(
        data_dir_img=data_paths["img"],
        data_dir_seg=data_paths["seg"],
        clinical_csv_path=data_paths["csv"],
        train=False,
        sample_ids=None,
        patch_size=patch_size,
        largest_tumor=largest_tumor,
        angle_range=(0.0, 0.0),
        mask_pad_threshold=0.8,
        dropout_p=0.0,
    )

    dev_manifest: List[Tuple[str, str, int]] = []
    test_ids: List[str] = []

    for sample in full_dataset:
        center = sample[KEY_CENTER]
        sid = sample[KEY_SAMPLE_ID]
        label = sample[KEY_TARGET]
        if center == TEST_CENTER:
            test_ids.append(sid)
        else:
            dev_manifest.append((sid, center, label))

    return dev_manifest, test_ids


def build_dataloaders_from_ids(
    data_paths: Dict[str, str],
    cfg: Dict[str, Any],
    largest_tumor: Tuple[int, int, int],
    train_ids: List[str],
    val_ids: List[str],
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Build train & val dataloaders from explicit sample ID lists.
    Uses the same augmentation and sampler logic as the original gist_dataloaders.
    """
    patch_size = cfg["patch_size"]
    angle_range = (-cfg["imaging_aug_deg"], cfg["imaging_aug_deg"])
    mask_pad_threshold = cfg["mask_pad_thresh"]
    dropout_p = cfg["clinical_aug"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]

    train_dataset = GISTDataset.dataset(
        data_dir_img=data_paths["img"],
        data_dir_seg=data_paths["seg"],
        clinical_csv_path=data_paths["csv"],
        train=True,
        sample_ids=train_ids,
        patch_size=patch_size,
        largest_tumor=largest_tumor,
        angle_range=angle_range,
        mask_pad_threshold=mask_pad_threshold,
        dropout_p=dropout_p,
    )

    sampler = BatchSamplerDefault(
        dataset=train_dataset,
        balanced_class_name=KEY_TARGET,
        num_balanced_classes=2,
        batch_size=batch_size,
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=sampler,
        collate_fn=CollateDefault(),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn,
    )

    val_dataset = GISTDataset.dataset(
        data_dir_img=data_paths["img"],
        data_dir_seg=data_paths["seg"],
        clinical_csv_path=data_paths["csv"],
        train=False,
        sample_ids=val_ids,
        patch_size=patch_size,
        largest_tumor=largest_tumor,
        angle_range=angle_range,
        mask_pad_threshold=mask_pad_threshold,
        dropout_p=dropout_p,
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        collate_fn=CollateDefault(),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn,
    )

    return train_dataloader, val_dataloader


# =============================
# SMAC: ConfigSpace & mapping (IMAGING-ONLY)
# =============================

def create_configspace(base_cfg: Dict[str, Any]) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=0)

    patch_size_options = [
        (12, 48, 48),
        (8, 64, 64),
        (16, 32, 32),
        (16, 64, 64)
    ]
    patch_size_strs = [f"{pz}x{py}x{px}" for (pz, py, px) in patch_size_options]
    default_patch_str = f"{base_cfg['patch_size'][0]}x{base_cfg['patch_size'][1]}x{base_cfg['patch_size'][2]}"
    patch_size_hp = Categorical("patch_size", patch_size_strs, default=default_patch_str)

    emb_dim_hp = Categorical("emb_dim", [64, 128, 256], default=base_cfg["emb_dim"])

    lr_hp = Float("lr", (3e-5, 5e-4), default=base_cfg.get("lr", 1e-4), log=True)
    wd_hp = Float("wd", (1e-6, 5e-3), default=base_cfg.get("wd", 1e-4), log=True)

    depth_b_hp = Categorical("depth_b", [2, 3], default=base_cfg["depth_b"])
    heads_b_hp = Categorical("heads_b", [4, 8], default=base_cfg["heads_b"])

    mlp_hp = Categorical("mlp_layers", ["single", "double"], default=base_cfg["mlp_layers"])

    mpt_hp = Categorical("mask_pad_thresh", [0.7, 0.8, 0.9], default=base_cfg["mask_pad_thresh"])
    ia_hp = Categorical("imaging_aug_deg", [0, 10, 20], default=base_cfg["imaging_aug_deg"])

    cs.add([
        patch_size_hp,
        emb_dim_hp,
        lr_hp,
        wd_hp,
        depth_b_hp,
        heads_b_hp,
        mlp_hp,
        mpt_hp,
        ia_hp,
    ])

    return cs


def cfg_from_config(config: Configuration, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    conf = dict(config)

    if "patch_size" in conf:
        patch_str = conf["patch_size"]
        pz, py, px = patch_str.split("x")
        cfg["patch_size"] = (int(pz), int(py), int(px))

    if "emb_dim" in conf:
        cfg["emb_dim"] = int(conf["emb_dim"])
    if "lr" in conf:
        cfg["lr"] = float(conf["lr"])
    if "wd" in conf:
        cfg["wd"] = float(conf["wd"])
    if "depth_b" in conf:
        cfg["depth_b"] = int(conf["depth_b"])
    if "heads_b" in conf:
        cfg["heads_b"] = int(conf["heads_b"])
    if "mlp_layers" in conf:
        cfg["mlp_layers"] = str(conf["mlp_layers"])
    if "mask_pad_thresh" in conf:
        cfg["mask_pad_thresh"] = float(conf["mask_pad_thresh"])
    if "imaging_aug_deg" in conf:
        cfg["imaging_aug_deg"] = float(conf["imaging_aug_deg"])

    cfg["test_center"] = TEST_CENTER
    return cfg


# =============================
# Training a single (config, fold, seed) trial
# =============================

def train_one_trial(
    run_dir: str,
    cfg: Dict[str, Any],
    data_paths: Dict[str, str],
    largest_tumor: Tuple[int, int, int],
    train_ids: List[str],
    val_ids: List[str],
    seed: int,
) -> float:
    """
    Train an imaging-only model for a single config on a single CV fold and single seed.
    Returns the validation AUC (float) based on best epoch.
    """
    ensure_dir(run_dir)
    seed_everything(seed)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    patch_size = cfg["patch_size"]
    patch_dim = int(np.prod(patch_size))
    max_num_tokens_b = compute_max_num_tokens(largest_tumor, patch_size)

    train_dl, val_dl = build_dataloaders_from_ids(
        data_paths=data_paths,
        cfg=cfg,
        largest_tumor=largest_tumor,
        train_ids=train_ids,
        val_ids=val_ids,
    )

    model = build_model(cfg, patch_dim, max_num_tokens_b)
    losses, train_metrics, validation_metrics, best_epoch_source = make_training_elements()

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=(0.5 ** 4) * cfg["lr"],
    )
    optimizers_and_lr_sch = dict(
        optimizer=optimizer,
        lr_scheduler=dict(scheduler=lr_scheduler, monitor="validation.metrics.auc")
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=run_dir,
        monitor="validation.metrics.auc",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="best_epoch",
        auto_insert_metric_name=False,
        verbose=True
    )

    early_stop_cb = EarlyStopping(
        monitor="validation.metrics.auc",
        patience=10,
        mode="max",
        verbose=True
    )
    csv_logger = CSVLogger(save_dir=run_dir, name=".")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    pl_module = LightningModuleDefault(
        model_dir=run_dir,
        model=model,
        losses=losses,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=optimizers_and_lr_sch,
    )

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        logger=csv_logger,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        callbacks=[early_stop_cb, checkpoint_cb, lr_monitor],
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    trainer.fit(pl_module, train_dl, val_dl)

    # Flatten CSV logs into run_dir (metrics.csv)
    try:
        log_dir = csv_logger.log_dir
        src_metrics = os.path.join(log_dir, "metrics.csv")
        dst_metrics = os.path.join(run_dir, "metrics.csv")
        if os.path.exists(src_metrics):
            if os.path.exists(dst_metrics):
                with open(dst_metrics, "a") as out_f, open(src_metrics, "r") as in_f:
                    lines = in_f.readlines()
                    start = 1 if lines and lines[0].lower().startswith("step") else 0
                    out_f.writelines(lines[start:])
            else:
                shutil.move(src_metrics, dst_metrics)
        src_hparams = os.path.join(log_dir, "hparams.yaml")
        if os.path.exists(src_hparams):
            shutil.move(src_hparams, os.path.join(run_dir, "hparams.yaml"))
        try:
            shutil.rmtree(log_dir)
        except Exception:
            pass
    except Exception as e:
        print(f"[WARN] Failed to flatten logs for {run_dir}: {e}")

    # Determine best validation AUC from metrics.csv
    val_auc = 0.0
    metrics_path = os.path.join(run_dir, "metrics.csv")

    if os.path.exists(metrics_path):
        try:
            df = pd.read_csv(metrics_path)
            if "validation.metrics.auc" in df.columns:
                values = df["validation.metrics.auc"].dropna()
                if len(values) > 0:
                    val_auc = float(values.max())
                    print(f"[INFO] Using best val_auc={val_auc:.4f} from metrics.csv")
                else:
                    print(f"[WARN] validation.metrics.auc column empty in {metrics_path}")
            else:
                print(f"[WARN] validation.metrics.auc not found in {metrics_path}")
        except Exception as e:
            print(f"[WARN] Could not read metrics.csv for {run_dir}: {e}")
    else:
        print(f"[WARN] metrics.csv not found in {run_dir}, leaving val_auc=0.0")

    best_ckpt_path = os.path.join(run_dir, "best_epoch.ckpt")
    if not os.path.exists(best_ckpt_path):
        print(
            f"[WARN] Expected best checkpoint at {best_ckpt_path} not found. "
            f"val_auc={val_auc:.4f} will be returned, best_ckpt set to null."
        )
        best_ckpt_path = None

    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(
            {
                "val_auc": val_auc,
                "best_ckpt": best_ckpt_path,
                "seed": seed,
            },
            f,
            indent=2,
        )

    return val_auc


# =============================
# Checkpoint loading helper (robust mapping)
# =============================

def load_model_from_checkpoint(
    cfg: Dict[str, Any],
    ckpt_path: str,
    largest_tumor: Tuple[int, int, int],
    device: torch.device,
) -> ModelMultiHead:
    """
    Build imaging-only model from cfg and load weights from a Lightning checkpoint.

    We strip everything before 'backbone.' or 'heads.' and load with strict=False.
    """
    patch_size = tuple(cfg["patch_size"])
    patch_dim = int(np.prod(patch_size))
    max_num_tokens_b = compute_max_num_tokens(largest_tumor, patch_size)

    model = build_model(cfg, patch_dim, max_num_tokens_b)
    model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = None
        if "backbone." in k:
            idx = k.index("backbone.")
            new_key = k[idx:]
        elif "heads." in k:
            idx = k.index("heads.")
            new_key = k[idx:]
        if new_key is not None:
            new_state_dict[new_key] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading {ckpt_path}: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading {ckpt_path}: {unexpected}")

    model.eval()
    return model


# =============================
# Evaluation helper: best-epoch predictions on validation
# =============================

def evaluate_fold_best_epoch(
    run_dir: str,
    cfg: Dict[str, Any],
    data_paths: Dict[str, str],
    largest_tumor: Tuple[int, int, int],
    val_ids: List[str],
    fold_idx: int,
    slug: str,
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load best_epoch.ckpt, run inference on the fold's validation set on CPU,
    and return:
      - a fold_summary dict
      - a list of per-sample prediction rows
    """
    summary: Dict[str, Any] = {
        "fold_idx": fold_idx,
        "run_dir": run_dir,
        "status": "ok",
    }
    rows: List[Dict[str, Any]] = []

    metrics_path = os.path.join(run_dir, "metrics.csv")
    summary["metrics_csv"] = metrics_path

    if os.path.exists(metrics_path):
        try:
            dfm = pd.read_csv(metrics_path)
            if "validation.metrics.auc" in dfm.columns:
                auc_col = dfm["validation.metrics.auc"].dropna()
                if len(auc_col) > 0:
                    summary["val_auc_best"] = float(auc_col.max())
                    last_row = dfm[dfm["validation.metrics.auc"].notna()].iloc[-1]
                    summary["val_auc_last"] = float(last_row["validation.metrics.auc"])
                    if "epoch" in dfm.columns:
                        best_idx = auc_col.idxmax()
                        best_epoch_val = dfm.loc[best_idx, "epoch"]
                        last_epoch_val = last_row["epoch"]
                        summary["best_epoch"] = int(best_epoch_val)
                        summary["last_epoch"] = int(last_epoch_val)
                        summary["num_epochs_trained"] = int(last_epoch_val) + 1
                else:
                    summary["status"] = "no_auc_values"
            else:
                summary["status"] = "no_auc_column"
        except Exception as e:
            print(f"[WARN] Could not parse metrics.csv for {run_dir}: {e}")
            summary["status"] = "metrics_parse_error"
    else:
        summary["status"] = "no_metrics"

    best_ckpt_path = os.path.join(run_dir, "best_epoch.ckpt")
    if not os.path.exists(best_ckpt_path):
        summary["best_ckpt"] = None
        print(f"[WARN] Cannot evaluate fold {fold_idx} in {run_dir}: best_epoch.ckpt missing.")
        return summary, rows

    summary["best_ckpt"] = best_ckpt_path

    patch_size = cfg["patch_size"]
    angle_range = (-cfg["imaging_aug_deg"], cfg["imaging_aug_deg"])
    mask_pad_threshold = cfg["mask_pad_thresh"]
    dropout_p = cfg["clinical_aug"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]

    val_dataset = GISTDataset.dataset(
        data_dir_img=data_paths["img"],
        data_dir_seg=data_paths["seg"],
        clinical_csv_path=data_paths["csv"],
        train=False,
        sample_ids=val_ids,
        patch_size=patch_size,
        largest_tumor=largest_tumor,
        angle_range=angle_range,
        mask_pad_threshold=mask_pad_threshold,
        dropout_p=dropout_p,
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        collate_fn=CollateDefault(),
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        worker_init_fn=worker_init_fn,
    )

    device = torch.device("cpu")
    model = load_model_from_checkpoint(
        cfg=cfg,
        ckpt_path=best_ckpt_path,
        largest_tumor=largest_tumor,
        device=device,
    )

    with torch.no_grad():
        for batch in val_dataloader:
            batch_out = model(batch)
            probs = batch_out[KEY_PROB].detach().cpu().view(-1)
            labels = batch_out[KEY_TARGET].detach().cpu().view(-1)
            centers = batch_out[KEY_CENTER]
            sids = batch_out[KEY_SAMPLE_ID]

            if isinstance(centers, torch.Tensor):
                centers_list = [str(c) for c in centers.cpu().tolist()]
            else:
                centers_list = list(centers)

            if isinstance(sids, torch.Tensor):
                sids_list = [str(s) for s in sids.cpu().tolist()]
            else:
                sids_list = list(sids)

            for i in range(len(sids_list)):
                rows.append(
                    {
                        "sample_id": str(sids_list[i]),
                        "center": str(centers_list[i]),
                        "y_true": int(labels[i].item()),
                        "y_prob": float(probs[i].item()),
                        "fold_idx": fold_idx,
                        "config_slug": slug,
                        "seed": seed,
                    }
                )

    summary["n_val_samples"] = len(rows)
    center_counts = Counter([r["center"] for r in rows])
    summary["val_center_counts"] = dict(center_counts)

    auc_per_center: Dict[str, Optional[float]] = {}
    for c, cnt in center_counts.items():
        ys = [r["y_true"] for r in rows if r["center"] == c]
        ps = [r["y_prob"] for r in rows if r["center"] == c]
        if len(set(ys)) >= 2:
            auc_per_center[c] = float(roc_auc_score(ys, ps))
        else:
            auc_per_center[c] = None
    summary["auc_per_center_best"] = auc_per_center

    return summary, rows


# =============================
# MAIN: SMAC-based HPO with 3-fold CV over dev (AVL+LUMC+RUMC), EMC as test
# =============================

def main():
    GLOBAL_SEED = 17
    seed_everything(GLOBAL_SEED)

    # Data paths
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    data_dir_img = os.path.join(data_dir, "img")
    data_dir_seg = os.path.join(data_dir, "seg")
    clinical_csv_path = os.path.join(data_dir, "GIST_clinical_data.csv")
    data_paths = {"img": data_dir_img, "seg": data_dir_seg, "csv": clinical_csv_path}

    largest_tumor = (80, 347, 498)

    base_cfg = {
        "patch_size": (8, 64, 64),
        "emb_dim": 64,
        "lr": 1e-4,
        "wd": 1e-3,
        "batch_size": 8,
        "num_workers": 10,
        # encoder defaults (kept for slug consistency; unused ones not tuned)
        "depth_a": 2, "heads_a": 2,
        "depth_b": 2, "heads_b": 4,
        "depth_cross_attn": 2, "heads_cross": 2,
        # head
        "mlp_layers": "single",
        # dataset params
        "mask_pad_thresh": 0.8,
        "clinical_aug": 0.0,   # imaging-only: keep fixed
        "imaging_aug_deg": 10,
        # fixed test center
        "test_center": TEST_CENTER,
    }

    runs_root = os.path.join(os.getcwd(), RUNS_DIR_NAME)
    ensure_dir(runs_root)

    dev_manifest, test_ids = build_dev_test_manifest(
        data_paths=data_paths,
        patch_size=base_cfg["patch_size"],
        largest_tumor=largest_tumor,
    )
    if len(dev_manifest) == 0:
        raise RuntimeError("Dev manifest is empty (no non-EMC centers found).")

    dev_sids = [sid for (sid, ctr, lab) in dev_manifest]
    dev_centers = [ctr for (sid, ctr, lab) in dev_manifest]
    dev_labels = [lab for (sid, ctr, lab) in dev_manifest]
    strat_labels = [f"{c}_{y}" for c, y in zip(dev_centers, dev_labels)]

    skf = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=GLOBAL_SEED,
    )
    cv_splits: List[Tuple[List[str], List[str]]] = []
    for tr_idx, va_idx in skf.split(np.zeros(len(strat_labels)), strat_labels):
        train_ids = [dev_sids[i] for i in tr_idx]
        val_ids = [dev_sids[i] for i in va_idx]
        cv_splits.append((train_ids, val_ids))

    print(f"Built {len(cv_splits)} CV folds for dev set. Test center is fixed to {TEST_CENTER}.")

    cs = create_configspace(base_cfg)

    walltime_limit_seconds = (5 * 24) * 3600
    scenario = Scenario(
        cs,
        deterministic=True,
        n_trials=10 ** 9,
        walltime_limit=walltime_limit_seconds,
        seed=GLOBAL_SEED,
        output_directory=SMAC_OUTPUT_DIR,
        name=SMAC_RUN_NAME,
    )

    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario,
        n_configs=10,
    )

    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario,
        max_config_calls=1,
    )

    # NEW objective (same as cross-attention "fixed" objective):
    # cost = 1 - (0.5 * AUC_dev + 0.5 * center_mean) computed from best-epoch val predictions.
    def smac_objective(config: Configuration, seed: int = 0) -> float:
        cfg = cfg_from_config(config, base_cfg)
        slug = config_slug(cfg)

        val_aucs: List[float] = []
        fold_summaries: List[Dict[str, Any]] = []
        all_pred_rows: List[Dict[str, Any]] = []

        for fold_idx, (train_ids, val_ids) in enumerate(cv_splits):
            fold_run_dir = os.path.join(
                runs_root,
                f"{slug}__fold{fold_idx}_seed{seed}"
            )
            print(f"\n[SMAC] (imaging-only) Config slug={slug} | seed={seed} | CV fold={fold_idx}")
            report = {k: cfg.get(k) for k in REPORT_KEYS if k in cfg}
            print(f"[SMAC] (imaging-only) CONFIG -> {report}")

            val_auc = train_one_trial(
                run_dir=fold_run_dir,
                cfg=cfg,
                data_paths=data_paths,
                largest_tumor=largest_tumor,
                train_ids=train_ids,
                val_ids=val_ids,
                seed=seed,
            )
            print(f"[SMAC] (imaging-only) slug={slug} | seed={seed} | fold={fold_idx} | val_auc={val_auc:.4f}")
            val_aucs.append(val_auc)

            fold_summary, fold_rows = evaluate_fold_best_epoch(
                run_dir=fold_run_dir,
                cfg=cfg,
                data_paths=data_paths,
                largest_tumor=largest_tumor,
                val_ids=val_ids,
                fold_idx=fold_idx,
                slug=slug,
                seed=seed,
            )
            fold_summaries.append(fold_summary)
            all_pred_rows.extend(fold_rows)

        AUC_dev: Optional[float] = None
        AUC_per_center: Dict[str, Optional[float]] = {}
        center_mean: Optional[float] = None
        n_dev_samples: int = 0
        n_dev_per_center: Dict[str, int] = {}

        if len(all_pred_rows) > 0:
            df_pred = pd.DataFrame(all_pred_rows)
            n_dev_samples = len(df_pred)

            if df_pred["y_true"].nunique() >= 2:
                AUC_dev = float(roc_auc_score(df_pred["y_true"], df_pred["y_prob"]))
            else:
                AUC_dev = None

            for center, group in df_pred.groupby("center"):
                n_dev_per_center[center] = len(group)
                if group["y_true"].nunique() >= 2:
                    AUC_per_center[center] = float(roc_auc_score(group["y_true"], group["y_prob"]))
                else:
                    AUC_per_center[center] = None

            valid_center_aucs = [v for v in AUC_per_center.values() if v is not None]
            if len(valid_center_aucs) > 0:
                center_mean = float(np.mean(valid_center_aucs))
            else:
                center_mean = None

            if AUC_dev is not None and center_mean is not None:
                score = 0.5 * AUC_dev + 0.5 * center_mean
            elif AUC_dev is not None:
                score = AUC_dev
            elif center_mean is not None:
                score = center_mean
            else:
                score = 0.0
        else:
            score = 0.0

        cost = 1.0 - score

        if len(all_pred_rows) > 0:
            df_pred = pd.DataFrame(all_pred_rows)
            pred_csv_path = os.path.join(runs_root, f"{slug}__seed{seed}_dev_val_predictions.csv")
            df_pred.to_csv(pred_csv_path, index=False)
        else:
            pred_csv_path = None

        mean_val_auc = float(np.mean(val_aucs)) if len(val_aucs) > 0 else 0.0

        aggregate_path = os.path.join(runs_root, f"{slug}__seed{seed}_aggregate.json")
        aggregate_payload: Dict[str, Any] = {
            "config_slug": slug,
            "cfg": cfg,
            "seed": seed,
            "fold_val_aucs": val_aucs,
            "mean_val_auc": mean_val_auc,
            "folds": fold_summaries,
            "dev_metrics": {
                "AUC_dev": AUC_dev,
                "AUC_per_center": AUC_per_center,
                "center_mean": center_mean,
                "n_dev_samples": n_dev_samples,
                "n_dev_per_center": n_dev_per_center,
            },
            "objective": {
                "formula": "1 - (0.5 * AUC_dev + 0.5 * center_mean)",
                "AUC_dev": AUC_dev,
                "center_mean": center_mean,
                "score": score,
                "cost": cost,
            },
            "predictions_csv": pred_csv_path,
            "test_center": TEST_CENTER,
        }

        with open(aggregate_path, "w") as f:
            json.dump(aggregate_payload, f, indent=2)

        print(
            f"[SMAC] (imaging-only) slug={slug} | seed={seed} | "
            f"mean_val_auc={mean_val_auc:.4f} | "
            f"AUC_dev={AUC_dev} | center_mean={center_mean} | cost={cost:.4f}"
        )
        return cost

    smac = HyperparameterOptimizationFacade(
        scenario,
        smac_objective,
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=False,
    )

    incumbent = smac.optimize()

    incumbent_cfg = cfg_from_config(incumbent, base_cfg)
    incumbent_slug = config_slug(incumbent_cfg)
    with open(os.path.join(runs_root, f"incumbent_imaging_only_{incumbent_slug}.json"), "w") as f:
        json.dump(incumbent_cfg, f, indent=2)

    print("\n[SMAC] (imaging-only) Optimization finished.")
    print(f"[SMAC] (imaging-only) Incumbent slug: {incumbent_slug}")
    print(f"[SMAC] (imaging-only) Incumbent config: {incumbent_cfg}")


if __name__ == "__main__":
    main()
