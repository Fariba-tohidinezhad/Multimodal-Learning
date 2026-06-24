# ============================================================
# SMAC_Scratch_CrossAttn.py
#
# SMAC3 HPO for downstream GIST model trained FROM SCRATCH
# (no pretrained enc_b, no LoRA).
#
# Uses:
#   - 5-fold StratifiedKFold CV over dev (centers != TEST_CENTER)
#   - fixed external test center EMC
#   - best-epoch evaluation on validation per fold
#   - objective / cost:
#       cost = 1 - mean(fold_val_aucs)
#
# Search space adapted from older scratch-training code:
#   - patch_size_zyx, emb_dim
#   - depth/heads for enc_a and enc_b
#   - depth/heads for cross-attention
#   - mlp_layers
#   - clinical_aug, imaging_aug_deg
#   - lr, wd
# ============================================================

import sys
import os
sys.path.insert(0, '/gpfs/home4/ftohidinezhad/fuse-med-ml')

import json
import copy
import random
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import OrderedDict, Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from ConfigSpace import Configuration, ConfigurationSpace, Categorical, Float
from smac import HyperparameterOptimizationFacade, Scenario

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from fuse.utils.ndict import NDict
from fuse.dl.models.model_multihead import ModelMultiHead
from fuse.dl.models.backbones.backbone_transformer import CrossAttentionTransformerEncoder
from fuse.dl.losses.loss_default import LossDefault
from fuse.eval.metrics.classification.metrics_classification_common import (
    MetricAUCROC, MetricConfusion, MetricBSS, MetricAccuracy
)
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
from fuse.dl.lightning.pl_module import LightningModuleDefault
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault

from TFS_Dataset import GISTDataset
from HeadMLPClassifier import HeadMLPClassifier


# =============================
# Global constants
# =============================

RUNS_DIR_NAME = "runs_smac_ca_scratch"
TEST_CENTER = "EMC"

SMAC_OUTPUT_DIR = "output_smac_ca_scratch"
SMAC_RUN_NAME = "gist_crossattn_smac_scratch"

# Fixed dataset / preprocessing defaults aligned to current dataset code
TARGET_SPACING_ZYX = (5.0, 0.7636719, 0.7636719)
CLIP_RANGE_HU = (-150.0, 250.0)
TARGET_SHAPE_ZYX = (80, 347, 498)
PAD_VAL_FIXED = 0.0
PAD_TO_MULTIPLE_FIXED = True
ROTATION_PROB_FIXED = 0.5

# Training defaults
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 10
GRAD_CLIP_VAL = 1.0

KEY_PROB = "model.prob.TKI_Classification"
KEY_LOGITS = "model.logits.TKI_Classification"
KEY_TARGET = "data.input.clinical.raw.TKIResponse"
KEY_TARGET_F = "data.input.clinical.raw.TKIResponse.f"
KEY_SAMPLE_ID = "data.sample_id"
KEY_CENTER = "data.input.clinical.raw.Center"

REPORT_KEYS = [
    "test_center",
    "patch_size_zyx", "emb_dim",
    "lr", "wd",
    "depth_a", "heads_a",
    "depth_b", "heads_b",
    "depth_cross_attn", "heads_cross",
    "mlp_layers",
    "min_tumor_frac", "clinical_aug", "imaging_aug_deg",
    "batch_size", "num_workers",
]


# =============================
# Utils
# =============================

def seed_everything(seed: int = 17) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def compute_max_num_tokens(
    largest_shape_zyx: Tuple[int, int, int],
    patch_size_zyx: Tuple[int, int, int],
) -> int:
    zc = ceil_div(largest_shape_zyx[0], patch_size_zyx[0])
    yc = ceil_div(largest_shape_zyx[1], patch_size_zyx[1])
    xc = ceil_div(largest_shape_zyx[2], patch_size_zyx[2])
    return int(zc * yc * xc)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def config_slug(cfg: Dict[str, Any], max_len: int = 140) -> str:
    import hashlib
    import re

    abbr = {
        "patch_size_zyx": "ps",
        "emb_dim": "ed",
        "lr": "lr",
        "wd": "wd",
        "depth_a": "da",
        "heads_a": "ha",
        "depth_b": "db",
        "heads_b": "hb",
        "depth_cross_attn": "dca",
        "heads_cross": "hc",
        "mlp_layers": "mlp",
        "min_tumor_frac": "mtf",
        "clinical_aug": "ca",
        "imaging_aug_deg": "ia",
    }
    order = [
        "patch_size_zyx", "emb_dim",
        "lr", "wd",
        "depth_a", "heads_a",
        "depth_b", "heads_b",
        "depth_cross_attn", "heads_cross",
        "mlp_layers", "min_tumor_frac", "clinical_aug", "imaging_aug_deg",
    ]

    def fmt_lr(x: float) -> str:
        if float(x) == 0.0:
            return "0"
        s = np.format_float_scientific(float(x), precision=0, unique=True, exp_digits=1)
        return s.replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")

    def fmt_ps(ps: Tuple[int, int, int]) -> str:
        return "x".join(str(int(v)) for v in ps)

    def fmt_mlp(v: str) -> str:
        return "S" if str(v).lower().startswith("s") else "D"

    def fmt_pct10(v: float) -> str:
        return f"{int(round(float(v) * 10)):02d}"

    def clean(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9\-x\.]", "", s)

    parts: List[str] = []
    for k in order:
        if k not in cfg:
            continue
        ab = abbr.get(k, k)
        v = cfg[k]

        if k == "patch_size_zyx":
            val = fmt_ps(v)
        elif k in ("emb_dim", "depth_a", "heads_a", "depth_b", "heads_b",
                   "depth_cross_attn", "heads_cross", "imaging_aug_deg"):
            val = str(int(v))
        elif k in ("lr", "wd"):
            val = fmt_lr(v)
        elif k == "mlp_layers":
            val = fmt_mlp(v)
        elif k in ("min_tumor_frac", "clinical_aug"):
            val = fmt_pct10(v)
        else:
            val = str(v)

        parts.append(f"{ab}{clean(val)}")

    slug = "-".join(parts)
    if len(slug) > max_len:
        cfg_subset = {k: cfg.get(k) for k in order if k in cfg}
        h = hashlib.sha1(json.dumps(cfg_subset, sort_keys=True, default=str).encode()).hexdigest()[:8]
        slug = slug[: max_len - (len(h) + 2)] + "-h" + h
    return slug


def make_slug_with_smac_id(cfg: Dict[str, Any], smac_config_id: int) -> str:
    return f"cfg{int(smac_config_id):03d}_{config_slug(cfg)}"


def ndict_to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device)

    if isinstance(x, NDict):
        for k in x.keys():
            x[k] = ndict_to_device(x[k], device)
        return x

    if isinstance(x, dict):
        for k in list(x.keys()):
            x[k] = ndict_to_device(x[k], device)
        return x

    if isinstance(x, list):
        return [ndict_to_device(v, device) for v in x]

    if isinstance(x, tuple):
        return tuple(ndict_to_device(v, device) for v in x)

    return x


# =============================
# Model building
# =============================

def build_model(
    cfg: Dict[str, Any],
    embed_num_tokens: Dict[str, int],
    patch_dim: int,
    max_num_tokens_b: int,
) -> ModelMultiHead:
    if cfg.get("mlp_layers", "single") == "double":
        layers_desc = (cfg["emb_dim"], max(1, cfg["emb_dim"] // 2))
    else:
        layers_desc = (cfg["emb_dim"],)

    dropout_rate = 0.1

    backbone = CrossAttentionTransformerEncoder(
        emb_dim=cfg["emb_dim"],

        num_tokens_a=16,
        max_seq_len_a=16,
        depth_a=cfg.get("depth_a", 2),
        heads_a=cfg.get("heads_a", 2),

        num_tokens_b=None,
        max_seq_len_b=max_num_tokens_b,
        depth_b=cfg.get("depth_b", 2),
        heads_b=cfg.get("heads_b", 4),

        depth_cross_attn=cfg.get("depth_cross_attn", 2),
        output_dim=cfg["emb_dim"],
        context="both",

        kwargs_wrapper_a={
            "embed_num_tokens": embed_num_tokens,
            "use_abs_pos_emb": False,
            "emb_dropout": dropout_rate,
            "post_emb_norm": True,
        },
        kwargs_wrapper_b={
            "token_emb": nn.Linear(patch_dim, cfg["emb_dim"]),
            "use_abs_pos_emb": False,
            "emb_dropout": dropout_rate,
            "post_emb_norm": True,
            "return_only_embed": True,
        },
        kwargs_encoder_a={
            "attn_dropout": dropout_rate,
            "ff_dropout": dropout_rate,
            "ff_glu": False,
            "use_rmsnorm": True,
        },
        kwargs_encoder_b={
            "attn_dropout": dropout_rate,
            "ff_dropout": dropout_rate,
            "ff_glu": False,
            "use_rmsnorm": True,
            "rotary_pos_emb": True,
        },
        kwargs_cross_attn={
            "heads": cfg.get("heads_cross", 2),
            "attn_dropout": dropout_rate,
            "ff_dropout": dropout_rate,
            "use_rmsnorm": True,
            "residual_attn": False,
            "cross_residual_attn": False,
        },
    )

    model = ModelMultiHead(
        backbone=backbone,
        key_out_features="model.backbone_features",
        backbone_args=[
            "data.input.clinical.vector",
            "data.input.img.tumor3d.patches",
            "model.embed_ids_a",
            "model.embed_mask_a",
            "model.embed_mask_b",
        ],
        heads=[
            HeadMLPClassifier(
                input_key="model.backbone_features",
                prob_key=KEY_PROB,
                logits_key=KEY_LOGITS,
                in_ch=cfg["emb_dim"],
                num_classes=1,
                layers_description=layers_desc,
                dropout_rate=dropout_rate,
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
            weight=1.0,
        )
    }

    confusion_metrics = ["sensitivity", "specificity", "ppv", "f1"]
    common_metrics = OrderedDict([
        ("auc", MetricAUCROC(pred=KEY_PROB, target=KEY_TARGET)),
        ("apply_thresh", MetricApplyThresholds(pred=KEY_PROB, operation_point=0.5)),
        ("confusion", MetricConfusion(
            pred="results:metrics.apply_thresh.cls_pred",
            target=KEY_TARGET,
            metrics=confusion_metrics,
        )),
        ("accuracy", MetricAccuracy(
            pred="results:metrics.apply_thresh.cls_pred",
            target=KEY_TARGET,
        )),
        ("bss", MetricBSS(pred=KEY_PROB, target=KEY_TARGET)),
    ])

    train_metrics = common_metrics.copy()
    validation_metrics = copy.deepcopy(common_metrics)
    best_epoch_source = dict(monitor="validation.metrics.auc", mode="max")

    return losses, train_metrics, validation_metrics, best_epoch_source


# =============================
# Data helpers
# =============================

def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_dev_test_manifest_from_csv(
    data_dir_seg: str,
    clinical_csv_path: str,
) -> Tuple[List[Tuple[str, str, int]], List[str]]:
    """
    Returns:
      dev_manifest: [(sample_id, center, label)] for centers != TEST_CENTER
      test_ids: list of sample_id for center == TEST_CENTER
    """
    sample_ids = GISTDataset.sample_ids(data_dir_seg)

    df = pd.read_csv(clinical_csv_path)
    df = df[df["sample_id"].isin(sample_ids)]

    need_cols = {"sample_id", "Center", "TKIResponse"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"Clinical CSV missing columns: {missing}")

    dev_manifest: List[Tuple[str, str, int]] = []
    test_ids: List[str] = []

    for _, row in df.iterrows():
        sid = str(row["sample_id"])
        center = str(row["Center"])
        y = int(row["TKIResponse"])

        if center == TEST_CENTER:
            test_ids.append(sid)
        else:
            dev_manifest.append((sid, center, y))

    return dev_manifest, test_ids


def build_dataloaders_from_ids(
    data_paths: Dict[str, str],
    cfg: Dict[str, Any],
    train_ids: List[str],
    val_ids: List[str],
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    patch_size_zyx = tuple(cfg["patch_size_zyx"])
    angle_range_deg = (-float(cfg["imaging_aug_deg"]), float(cfg["imaging_aug_deg"]))
    min_tumor_frac = float(cfg["min_tumor_frac"])
    dropout_p = float(cfg["clinical_aug"])
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg["num_workers"])

    train_dataset = GISTDataset.dataset(
        data_dir_img=data_paths["img"],
        data_dir_seg=data_paths["seg"],
        clinical_csv_path=data_paths["csv"],

        target_spacing_zyx=TARGET_SPACING_ZYX,
        clip_range_hu=CLIP_RANGE_HU,
        target_shape_zyx=TARGET_SHAPE_ZYX,
        pad_val=PAD_VAL_FIXED,

        patch_size_zyx=patch_size_zyx,
        min_tumor_frac=min_tumor_frac,
        pad_to_multiple=PAD_TO_MULTIPLE_FIXED,
        angle_range_deg=angle_range_deg,
        rotation_prob=ROTATION_PROB_FIXED,

        dropout_p=dropout_p,
        train=True,
        sample_ids=train_ids,
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

        target_spacing_zyx=TARGET_SPACING_ZYX,
        clip_range_hu=CLIP_RANGE_HU,
        target_shape_zyx=TARGET_SHAPE_ZYX,
        pad_val=PAD_VAL_FIXED,

        patch_size_zyx=patch_size_zyx,
        min_tumor_frac=min_tumor_frac,
        pad_to_multiple=PAD_TO_MULTIPLE_FIXED,
        angle_range_deg=(0.0, 0.0),
        rotation_prob=0.0,

        dropout_p=0.0,
        train=False,
        sample_ids=val_ids,
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
# FineTuning: ConfigSpace & config mapping
# =============================

def create_configspace(base_cfg: Dict[str, Any]) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=0)

    patch_size_options = [
        (12, 48, 48),
        (8, 64, 64),
        (16, 32, 32),
        (16, 64, 64),
    ]
    patch_size_strs = [f"{pz}x{py}x{px}" for (pz, py, px) in patch_size_options]
    default_patch_str = f"{base_cfg['patch_size_zyx'][0]}x{base_cfg['patch_size_zyx'][1]}x{base_cfg['patch_size_zyx'][2]}"
    patch_size_hp = Categorical("patch_size_zyx", patch_size_strs, default=default_patch_str)

    emb_dim_hp = Categorical("emb_dim", [64, 128, 256], default=base_cfg["emb_dim"])

    lr_hp = Float("lr", (3e-5, 5e-4), default=base_cfg.get("lr", 1e-4), log=True)
    wd_hp = Float("wd", (1e-6, 5e-3), default=base_cfg.get("wd", 1e-4), log=True)

    depth_a_hp = Categorical("depth_a", [1, 2], default=base_cfg["depth_a"])
    heads_a_hp = Categorical("heads_a", [2, 4], default=base_cfg["heads_a"])

    depth_b_hp = Categorical("depth_b", [2, 3], default=base_cfg["depth_b"])
    heads_b_hp = Categorical("heads_b", [4, 8], default=base_cfg["heads_b"])

    depth_cross_hp = Categorical("depth_cross_attn", [1, 2], default=base_cfg["depth_cross_attn"])
    heads_cross_hp = Categorical("heads_cross", [2, 4], default=base_cfg["heads_cross"])

    mlp_hp = Categorical("mlp_layers", ["single", "double"], default=base_cfg["mlp_layers"])

    # aligned to CURRENT dataset API
    mtf_hp = Categorical("min_tumor_frac", [0.2, 0.3, 0.4], default=base_cfg["min_tumor_frac"])

    ca_hp = Categorical("clinical_aug", [0.0, 0.1], default=base_cfg["clinical_aug"])
    ia_hp = Categorical("imaging_aug_deg", [0, 5, 10], default=base_cfg["imaging_aug_deg"])

    cs.add([
        patch_size_hp,
        emb_dim_hp,
        lr_hp,
        wd_hp,
        depth_a_hp,
        heads_a_hp,
        depth_b_hp,
        heads_b_hp,
        depth_cross_hp,
        heads_cross_hp,
        mlp_hp,
        mtf_hp,
        ca_hp,
        ia_hp,
    ])

    return cs


def cfg_from_config(config: Configuration, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    conf = dict(config)

    patch_str = conf["patch_size_zyx"]
    pz, py, px = patch_str.split("x")
    cfg["patch_size_zyx"] = (int(pz), int(py), int(px))

    cfg["emb_dim"] = int(conf["emb_dim"])
    cfg["lr"] = float(conf["lr"])
    cfg["wd"] = float(conf["wd"])

    cfg["depth_a"] = int(conf["depth_a"])
    cfg["heads_a"] = int(conf["heads_a"])

    cfg["depth_b"] = int(conf["depth_b"])
    cfg["heads_b"] = int(conf["heads_b"])

    cfg["depth_cross_attn"] = int(conf["depth_cross_attn"])
    cfg["heads_cross"] = int(conf["heads_cross"])

    cfg["mlp_layers"] = str(conf["mlp_layers"])

    cfg["min_tumor_frac"] = float(conf["min_tumor_frac"])
    cfg["clinical_aug"] = float(conf["clinical_aug"])
    cfg["imaging_aug_deg"] = float(conf["imaging_aug_deg"])

    cfg["test_center"] = TEST_CENTER
    return cfg


# =============================
# Train one trial
# =============================

def train_one_trial(
    run_dir: str,
    cfg: Dict[str, Any],
    data_paths: Dict[str, str],
    embed_num_tokens: Dict[str, int],
    largest_shape_zyx: Tuple[int, int, int],
    train_ids: List[str],
    val_ids: List[str],
    seed: int,
) -> float:
    ensure_dir(run_dir)
    seed_everything(seed)

    save_json(os.path.join(run_dir, "config.json"), cfg)

    patch_size_zyx = tuple(cfg["patch_size_zyx"])
    patch_dim = int(np.prod(patch_size_zyx))
    max_num_tokens_b = compute_max_num_tokens(largest_shape_zyx, patch_size_zyx)

    train_dl, val_dl = build_dataloaders_from_ids(
        data_paths=data_paths,
        cfg=cfg,
        train_ids=train_ids,
        val_ids=val_ids,
    )

    model = build_model(cfg, embed_num_tokens, patch_dim, max_num_tokens_b)
    losses, train_metrics, validation_metrics, best_epoch_source = make_training_elements()

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=(0.5 ** 4) * cfg["lr"],
    )
    optimizers_and_lr_sch = dict(
        optimizer=optimizer,
        lr_scheduler=dict(scheduler=lr_scheduler, monitor="validation.metrics.auc"),
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=run_dir,
        monitor="validation.metrics.auc",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="best_epoch",
        auto_insert_metric_name=False,
        verbose=True,
    )

    early_stop_cb = EarlyStopping(
        monitor="validation.metrics.auc",
        patience=EARLY_STOP_PATIENCE,
        mode="max",
        verbose=True,
    )

    csv_logger = CSVLogger(save_dir=run_dir, name=".")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

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
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        logger=csv_logger,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        callbacks=[early_stop_cb, checkpoint_cb, lr_monitor],
        gradient_clip_val=GRAD_CLIP_VAL,
        enable_progress_bar=True,
    )

    trainer.fit(pl_module, train_dl, val_dl)

    try:
        log_dir = csv_logger.log_dir
        src_metrics = os.path.join(log_dir, "metrics.csv")
        dst_metrics = os.path.join(run_dir, "metrics.csv")

        if os.path.exists(src_metrics):
            if os.path.exists(dst_metrics):
                with open(dst_metrics, "a", encoding="utf-8") as out_f, open(src_metrics, "r", encoding="utf-8") as in_f:
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
        except Exception as e:
            print(f"[WARN] Could not read metrics.csv for {run_dir}: {e}")
    else:
        print(f"[WARN] metrics.csv not found in {run_dir}, leaving val_auc=0.0")

    best_ckpt_path = os.path.join(run_dir, "best_epoch.ckpt")
    if not os.path.exists(best_ckpt_path):
        print(f"[WARN] Expected best checkpoint at {best_ckpt_path} not found.")
        best_ckpt_path = None

    save_json(
        os.path.join(run_dir, "result.json"),
        {
            "smac_config_id": cfg.get("smac_config_id"),
            "val_auc": val_auc,
            "best_ckpt": best_ckpt_path,
            "seed": seed,
        },
    )

    return val_auc


# =============================
# Checkpoint loading helper
# =============================

def load_model_from_checkpoint(
    cfg: Dict[str, Any],
    ckpt_path: str,
    embed_num_tokens: Dict[str, int],
    largest_shape_zyx: Tuple[int, int, int],
    device: torch.device,
) -> ModelMultiHead:
    patch_size_zyx = tuple(cfg["patch_size_zyx"])
    patch_dim = int(np.prod(patch_size_zyx))
    max_num_tokens_b = compute_max_num_tokens(largest_shape_zyx, patch_size_zyx)

    model = build_model(cfg, embed_num_tokens, patch_dim, max_num_tokens_b)
    model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = None

        if k.startswith("model."):
            new_key = k[len("model."):]
        elif k.startswith("_model."):
            new_key = k[len("_model."):]
        elif k.startswith("pl_module.model."):
            new_key = k[len("pl_module.model."):]
        elif k.startswith("pl_module._model."):
            new_key = k[len("pl_module._model."):]
        elif "backbone." in k:
            new_key = k[k.index("backbone."):]
        elif "heads." in k:
            new_key = k[k.index("heads."):]

        if new_key is not None:
            new_state_dict[new_key] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
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
    embed_num_tokens: Dict[str, int],
    largest_shape_zyx: Tuple[int, int, int],
    val_ids: List[str],
    fold_idx: int,
    slug: str,
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    summary: Dict[str, Any] = {"fold_idx": fold_idx, "run_dir": run_dir, "status": "ok"}
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
                        summary["best_epoch"] = int(dfm.loc[best_idx, "epoch"])
                        summary["last_epoch"] = int(last_row["epoch"])
                        summary["num_epochs_trained"] = int(last_row["epoch"]) + 1
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

    patch_size_zyx = tuple(cfg["patch_size_zyx"])
    min_tumor_frac = float(cfg["min_tumor_frac"])
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg["num_workers"])

    val_dataset = GISTDataset.dataset(
        data_dir_img=data_paths["img"],
        data_dir_seg=data_paths["seg"],
        clinical_csv_path=data_paths["csv"],

        target_spacing_zyx=TARGET_SPACING_ZYX,
        clip_range_hu=CLIP_RANGE_HU,
        target_shape_zyx=TARGET_SHAPE_ZYX,
        pad_val=PAD_VAL_FIXED,

        patch_size_zyx=patch_size_zyx,
        min_tumor_frac=min_tumor_frac,
        pad_to_multiple=PAD_TO_MULTIPLE_FIXED,
        angle_range_deg=(0.0, 0.0),
        rotation_prob=0.0,

        dropout_p=0.0,
        train=False,
        sample_ids=val_ids,
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
        embed_num_tokens=embed_num_tokens,
        largest_shape_zyx=largest_shape_zyx,
        device=device,
    )

    with torch.no_grad():
        for batch in val_dataloader:
            batch_nd = batch if isinstance(batch, NDict) else NDict(batch)
            batch_nd = ndict_to_device(batch_nd, device)

            model(batch_nd)

            probs = batch_nd[KEY_PROB].detach().cpu().view(-1)
            labels = batch_nd[KEY_TARGET].detach().cpu().view(-1)
            centers = batch_nd[KEY_CENTER]
            sids = batch_nd[KEY_SAMPLE_ID]

            centers_list = [str(c) for c in centers.cpu().tolist()] if isinstance(centers, torch.Tensor) else list(centers)
            sids_list = [str(s) for s in sids.cpu().tolist()] if isinstance(sids, torch.Tensor) else list(sids)

            for i in range(len(sids_list)):
                rows.append({
                    "sample_id": str(sids_list[i]),
                    "center": str(centers_list[i]),
                    "y_true": int(labels[i].item()),
                    "y_prob": float(probs[i].item()),
                    "fold_idx": fold_idx,
                    "smac_config_id": cfg.get("smac_config_id"),
                    "config_slug": slug,
                    "seed": seed,
                })

    summary["n_val_samples"] = len(rows)
    center_counts = Counter([r["center"] for r in rows])
    summary["val_center_counts"] = dict(center_counts)

    auc_per_center: Dict[str, Optional[float]] = {}
    for c in center_counts.keys():
        ys = [r["y_true"] for r in rows if r["center"] == c]
        ps = [r["y_prob"] for r in rows if r["center"] == c]
        auc_per_center[c] = float(roc_auc_score(ys, ps)) if len(set(ys)) >= 2 else None
    summary["auc_per_center_best"] = auc_per_center

    return summary, rows


# =============================
# Main
# =============================

def main() -> None:
    GLOBAL_SEED = 17
    seed_everything(GLOBAL_SEED)

    script_dir = Path(__file__).resolve().parent
    gist_root = script_dir.parents[1]
    data_dir = gist_root / "data"

    data_dir_img = data_dir / "img"
    data_dir_seg = data_dir / "seg"
    clinical_csv_path = data_dir / "GIST_clinical_data.csv"

    data_paths = {
        "img": str(data_dir_img),
        "seg": str(data_dir_seg),
        "csv": str(clinical_csv_path),
    }

    embed_num_tokens = {
        "0": 3, "1": 3, "2": 3, "3": 3, "4": 8, "5": 4, "6": 4, "7": 3, "8": 3,
        "9": 3, "10": 3, "11": 3, "12": 3, "13": 3, "14": 3, "15": 3
    }

    largest_shape_zyx = TARGET_SHAPE_ZYX

    base_cfg = {
        "patch_size_zyx": (8, 64, 64),
        "emb_dim": 64,

        "lr": 1e-4,
        "wd": 1e-3,

        "batch_size": 16,
        "num_workers": 10,

        "depth_a": 2,
        "heads_a": 2,

        "depth_b": 2,
        "heads_b": 4,

        "depth_cross_attn": 2,
        "heads_cross": 2,

        "mlp_layers": "single",

        # aligned to current dataset op: OpTumorPatchifyWithMask(min_tumor_frac=...)
        "min_tumor_frac": 0.3,

        "clinical_aug": 0.1,
        "imaging_aug_deg": 10,

        "test_center": TEST_CENTER,
    }

    runs_root = os.path.join(os.getcwd(), RUNS_DIR_NAME)
    ensure_dir(runs_root)

    dev_manifest, test_ids = build_dev_test_manifest_from_csv(
        data_dir_seg=data_paths["seg"],
        clinical_csv_path=data_paths["csv"],
    )
    if len(dev_manifest) == 0:
        raise RuntimeError("Dev manifest is empty (no non-EMC centers found).")

    dev_sids = [sid for (sid, ctr, lab) in dev_manifest]
    dev_centers = [ctr for (sid, ctr, lab) in dev_manifest]
    dev_labels = [lab for (sid, ctr, lab) in dev_manifest]
    strat_labels = [f"{c}_{y}" for c, y in zip(dev_centers, dev_labels)]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
    cv_splits: List[Tuple[List[str], List[str]]] = []
    for tr_idx, va_idx in skf.split(np.zeros(len(strat_labels)), strat_labels):
        train_ids = [dev_sids[i] for i in tr_idx]
        val_ids = [dev_sids[i] for i in va_idx]
        cv_splits.append((train_ids, val_ids))

    print(f"Built {len(cv_splits)} CV folds for dev set. Test center is fixed to {TEST_CENTER}.")

    cs = create_configspace(base_cfg)

    walltime_limit_seconds = 5 * 24 * 3600
    scenario = Scenario(
        cs,
        deterministic=True,
        n_trials=10 ** 9,
        walltime_limit=walltime_limit_seconds,
        seed=GLOBAL_SEED,
        output_directory=SMAC_OUTPUT_DIR,
        name=SMAC_RUN_NAME,
    )

    initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=10)
    intensifier = HyperparameterOptimizationFacade.get_intensifier(scenario, max_config_calls=1)

    smac_ref: Dict[str, Any] = {"smac": None}

    def smac_objective(config: Configuration, seed: int = 0) -> float:
        cfg = cfg_from_config(config, base_cfg)

        smac_obj = smac_ref["smac"]
        if smac_obj is None:
            raise RuntimeError("FineTuning object is not available inside objective.")

        smac_config_id = smac_obj.runhistory.config_ids.get(config, None)
        if smac_config_id is None:
            try:
                smac_config_id = smac_obj.runhistory.get_config_id(config)
            except Exception as e:
                raise RuntimeError(
                    f"Could not retrieve FineTuning internal config id for config: {config}"
                ) from e

        smac_config_id = int(smac_config_id)
        cfg["smac_config_id"] = smac_config_id
        slug = make_slug_with_smac_id(cfg, smac_config_id)

        val_aucs: List[float] = []
        fold_summaries: List[Dict[str, Any]] = []
        all_pred_rows: List[Dict[str, Any]] = []

        for fold_idx, (train_ids, val_ids) in enumerate(cv_splits):
            fold_run_dir = os.path.join(runs_root, f"{slug}__fold{fold_idx}_seed{seed}")
            print(f"\n[FineTuning] Config ID={smac_config_id} | slug={slug} | seed={seed} | CV fold={fold_idx}")
            report = {k: cfg.get(k) for k in REPORT_KEYS if k in cfg}
            print(f"[FineTuning] CONFIG -> {report}")

            val_auc = train_one_trial(
                run_dir=fold_run_dir,
                cfg=cfg,
                data_paths=data_paths,
                embed_num_tokens=embed_num_tokens,
                largest_shape_zyx=largest_shape_zyx,
                train_ids=train_ids,
                val_ids=val_ids,
                seed=seed,
            )
            print(f"[FineTuning] cfg_id={smac_config_id} | slug={slug} | seed={seed} | fold={fold_idx} | val_auc={val_auc:.4f}")
            val_aucs.append(val_auc)

            fold_summary, fold_rows = evaluate_fold_best_epoch(
                run_dir=fold_run_dir,
                cfg=cfg,
                data_paths=data_paths,
                embed_num_tokens=embed_num_tokens,
                largest_shape_zyx=largest_shape_zyx,
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

            for center, group in df_pred.groupby("center"):
                n_dev_per_center[center] = len(group)
                if group["y_true"].nunique() >= 2:
                    AUC_per_center[center] = float(roc_auc_score(group["y_true"], group["y_prob"]))
                else:
                    AUC_per_center[center] = None

            valid_center_aucs = [v for v in AUC_per_center.values() if v is not None]
            center_mean = float(np.mean(valid_center_aucs)) if len(valid_center_aucs) > 0 else None

            pred_csv_path = os.path.join(runs_root, f"{slug}__seed{seed}_dev_val_predictions.csv")
            df_pred.to_csv(pred_csv_path, index=False)
        else:
            pred_csv_path = None

        mean_val_auc = float(np.mean(val_aucs)) if len(val_aucs) > 0 else 0.0
        score = mean_val_auc
        cost = 1.0 - score

        aggregate_path = os.path.join(runs_root, f"{slug}__seed{seed}_aggregate.json")
        aggregate_payload: Dict[str, Any] = {
            "smac_config_id": smac_config_id,
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
                "formula": "1 - mean(fold_val_aucs)",
                "fold_val_aucs": val_aucs,
                "mean_val_auc": mean_val_auc,
                "score": score,
                "cost": cost,
            },
            "predictions_csv": pred_csv_path,
            "test_center": TEST_CENTER,
        }
        save_json(aggregate_path, aggregate_payload)

        print(
            f"[FineTuning] cfg_id={smac_config_id} | slug={slug} | seed={seed} | "
            f"mean_val_auc={mean_val_auc:.4f} | cost={cost:.4f}"
        )
        return cost

    smac = HyperparameterOptimizationFacade(
        scenario,
        smac_objective,
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=False,
    )
    smac_ref["smac"] = smac

    incumbent = smac.optimize()

    incumbent_cfg = cfg_from_config(incumbent, base_cfg)
    incumbent_config_id = smac.runhistory.get_config_id(incumbent)
    incumbent_cfg["smac_config_id"] = int(incumbent_config_id)
    incumbent_slug = make_slug_with_smac_id(incumbent_cfg, incumbent_config_id)

    save_json(os.path.join(runs_root, f"incumbent_{incumbent_slug}.json"), incumbent_cfg)

    print("\n[FineTuning] Optimization finished.")
    print(f"[FineTuning] Incumbent config ID: {incumbent_config_id}")
    print(f"[FineTuning] Incumbent slug: {incumbent_slug}")
    print(f"[FineTuning] Incumbent config: {incumbent_cfg}")


if __name__ == "__main__":
    main()