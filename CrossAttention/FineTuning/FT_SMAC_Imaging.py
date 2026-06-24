# ============================================================
# SMAC_LoRA_EncB_ImagingOnly.py
#
# SMAC3 HPO for downstream GIST imaging-only ablation using
# pretrained imaging encoder (enc_b) with LoRA applied to
# enc_b Attention projections "qv".
#
# Compared with the cross-attention version:
#   - removes clinical branch
#   - removes cross-attention
#   - keeps the imaging encoder branch as close as possible
#     to the original setup
#   - keeps the same HeadMLPClassifier(pooling="attention")
#
# Objective / cost:
#   cost = 1 - mean(fold_val_aucs)
#
# Keeps the same:
#   - 5-fold StratifiedKFold CV over dev (centers != TEST_CENTER)
#   - fixed external test center EMC
#   - best-epoch evaluation on validation per fold
#   - aggregate JSON saved per config
#
# FineTuning budget:
#   - n_trials = MAX_CONFIGS
# ============================================================

import sys
import os
sys.path.insert(0, '/gpfs/home4/ftohidinezhad/fuse-med-ml')

import json
import copy
import random
import shutil
import math
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

from x_transformers import Encoder

from fuse.utils.ndict import NDict
from fuse.dl.models.model_multihead import ModelMultiHead
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

from FT_Dataset import GISTDataset
from HeadMLPClassifier import HeadMLPClassifier


# =============================
# Global constants
# =============================

RUNS_DIR_NAME = "runs_smac_img_only"
TEST_CENTER = "EMC"

SMAC_OUTPUT_DIR = "output_smac_img_only"
SMAC_RUN_NAME = "gist_img_only_smac_lora"

MAX_CONFIGS = 57

# pretrained enc_b state_dict used in downstream training
PRETRAINED_ENCB_PATH = (
    r"/gpfs/home4/ftohidinezhad/fuse-med-ml/"
    r"GIST/GIST_CrossAttention/FineTuning/enc_b/enc_b_only_epoch_200.pt"
)

# Fixed LoRA settings for this round
LORA_DROPOUT_FIXED = 0.05
LORA_WEIGHT_DECAY_FIXED = 0.0
LORA_TARGET_FIXED = "qv"

# Pretraining-fixed architecture constraints
PATCH_SIZE_FIXED = (16, 64, 64)   # (Z, Y, X)
EMB_DIM_FIXED = 64
DEPTH_B_FIXED = 3
HEADS_B_FIXED = 4

# Dataset fixed/defaults
TARGET_SHAPE_ZYX = (80, 347, 498)   # used only to estimate max_num_tokens_b
ROTATION_PROB_FIXED = 0.5
PAD_TO_MULTIPLE_FIXED = True
PAD_VAL_FIXED = 0.0

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
    "depth_b", "heads_b",
    "mlp_layers",
    "min_tumor_frac", "imaging_aug_deg",
    "lora_r", "lora_ratio", "lora_alpha", "lora_target",
    "lora_dropout", "lora_weight_decay",
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
        "lr": "lr",
        "wd": "wd",
        "mlp_layers": "mlp",
        "min_tumor_frac": "mtf",
        "imaging_aug_deg": "ia",
        "lora_r": "r",
        "lora_ratio": "rr",
        "lora_target": "t",
    }
    order = [
        "lr", "wd",
        "mlp_layers", "min_tumor_frac", "imaging_aug_deg",
        "lora_r", "lora_ratio", "lora_target",
    ]

    def fmt_lr(x: float) -> str:
        if float(x) == 0.0:
            return "0"
        s = np.format_float_scientific(float(x), precision=0, unique=True, exp_digits=1)
        return s.replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")

    def fmt_mlp(v: str) -> str:
        return "S" if str(v).lower().startswith("s") else "D"

    def fmt_ratio(v: float) -> str:
        return str(int(v)) if float(v).is_integer() else str(v).replace(".", "")

    def fmt_mtf(v: float) -> str:
        return f"{int(round(float(v) * 10)):02d}"

    def clean(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9\-x\.]", "", s)

    parts: List[str] = []
    for k in order:
        if k not in cfg:
            continue
        ab = abbr.get(k, k)
        v = cfg[k]

        if k in ("lr", "wd"):
            val = fmt_lr(v)
        elif k in ("lora_r", "imaging_aug_deg"):
            val = str(int(v))
        elif k == "mlp_layers":
            val = fmt_mlp(v)
        elif k == "lora_ratio":
            val = fmt_ratio(v)
        elif k == "min_tumor_frac":
            val = fmt_mtf(v)
        elif k == "lora_target":
            val = str(v)
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


# =============================
# LoRA implementation
# =============================

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


def freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def _named_module_setattr(root: nn.Module, qualname: str, new_module: nn.Module) -> None:
    parts = qualname.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


class LoRALinear(nn.Module):
    """
    LoRA for any Linear:
      y = W x + scale * B(A(drop(x)))
    """
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(1, self.r)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        in_f = base.in_features
        out_f = base.out_features
        self.A = nn.Linear(in_f, self.r, bias=False)
        self.B = nn.Linear(self.r, out_f, bias=False)

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.B(self.A(self.drop(x)))

    @property
    def lora_parameters(self) -> List[nn.Parameter]:
        return list(self.A.parameters()) + list(self.B.parameters())


def apply_lora_to_enc_b_toq_tov(
    enc_b: nn.Module,
    emb_dim: int,
    r: int,
    alpha: float,
    dropout: float,
    target: str = "qv",
) -> Dict[str, Any]:
    """
    Wrap enc_b Attention.to_q and/or Attention.to_v with LoRALinear.
    Matches:
      - module name endswith ".to_q" / ".to_v"
      - module is nn.Linear
      - in_features == emb_dim
    """
    tgt = str(target).lower()
    report = {
        "patched_to_q": [],
        "patched_to_v": [],
        "lora_param_count": 0,
        "lora_param_tensors": 0,
    }

    named = dict(enc_b.named_modules())
    for name, mod in list(named.items()):
        if not isinstance(mod, nn.Linear):
            continue
        if int(mod.in_features) != int(emb_dim):
            continue

        if name.endswith(".to_q") and ("q" in tgt):
            _named_module_setattr(enc_b, name, LoRALinear(mod, r=r, alpha=alpha, dropout=dropout))
            report["patched_to_q"].append(name)

        if name.endswith(".to_v") and ("v" in tgt):
            _named_module_setattr(enc_b, name, LoRALinear(mod, r=r, alpha=alpha, dropout=dropout))
            report["patched_to_v"].append(name)

    lora_params: List[nn.Parameter] = []
    for m in enc_b.modules():
        if isinstance(m, LoRALinear):
            lora_params += m.lora_parameters

    for p in lora_params:
        p.requires_grad = True

    report["lora_param_count"] = int(sum(p.numel() for p in lora_params))
    report["lora_param_tensors"] = int(len(lora_params))
    return report


def collect_lora_params(enc_b: nn.Module) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for m in enc_b.modules():
        if isinstance(m, LoRALinear):
            params += m.lora_parameters

    seen = set()
    uniq: List[nn.Parameter] = []
    for p in params:
        if id(p) not in seen:
            uniq.append(p)
            seen.add(id(p))
    return uniq


# =============================
# Imaging-only encoder wrapper
# =============================

class PostEmbRMSNorm(nn.Module):
    """
    Lightweight RMSNorm-like module with parameter name `gamma`
    to match the checkpoint key: post_emb_norm.gamma
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.gamma


class PretrainedEncBWrapper(nn.Module):
    """
    Wrapper that matches the checkpoint structure:
      - token_emb.*
      - post_emb_norm.gamma
      - attn_layers.*

    This lets us load enc_b_only_epoch_200.pt into model.backbone.enc_b
    with strict=True.
    """
    def __init__(
        self,
        emb_dim: int,
        patch_dim: int,
        depth_b: int,
        heads_b: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Linear(int(patch_dim), int(emb_dim))
        self.post_emb_norm = PostEmbRMSNorm(int(emb_dim))
        self.emb_dropout = nn.Dropout(float(dropout_rate))

        self.attn_layers = Encoder(
            dim=int(emb_dim),
            depth=int(depth_b),
            heads=int(heads_b),
            attn_dropout=float(dropout_rate),
            ff_dropout=float(dropout_rate),
            ff_glu=False,
            use_rmsnorm=True,
            rotary_pos_emb=True,
        )

    def forward(
        self,
        xb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.token_emb(xb)
        x = self.emb_dropout(x)
        x = self.post_emb_norm(x)

        if mask is not None:
            x = self.attn_layers(x, mask=mask.bool())
        else:
            x = self.attn_layers(x)

        if isinstance(x, tuple):
            x = x[0]

        return x


class ImagingOnlyBackbone(nn.Module):
    """
    Imaging-only backbone.

    Exposes:
      model.backbone.enc_b

    so pretrained loading and LoRA injection work the same way as before.
    """
    def __init__(
        self,
        emb_dim: int,
        patch_dim: int,
        max_num_tokens_b: int,
        depth_b: int,
        heads_b: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.max_num_tokens_b = int(max_num_tokens_b)
        self.output_dim = int(emb_dim)

        self.enc_b = PretrainedEncBWrapper(
            emb_dim=emb_dim,
            patch_dim=patch_dim,
            depth_b=depth_b,
            heads_b=heads_b,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        xb: torch.Tensor,
        mask_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if xb.ndim != 3:
            raise ValueError(f"Expected xb to be [B, N, patch_dim], got {tuple(xb.shape)}")

        if xb.shape[1] > self.max_num_tokens_b:
            raise ValueError(
                f"Sequence length {xb.shape[1]} exceeds max_num_tokens_b={self.max_num_tokens_b}"
            )

        return self.enc_b(xb, mask=mask_b)


# =============================
# Model building
# =============================

def load_pretrained_enc_b_into_backbone(model: ModelMultiHead, encb_path: str) -> None:
    sd = torch.load(encb_path, map_location="cpu")
    missing, unexpected = model.backbone.enc_b.load_state_dict(sd, strict=True)
    if missing or unexpected:
        print(f"[WARN] enc_b load missing keys: {len(missing)}")
        print(f"[WARN] enc_b load unexpected keys: {len(unexpected)}")
    print("[INFO] Loaded pretrained enc_b weights.")


def build_model(
    cfg: Dict[str, Any],
    patch_dim: int,
    max_num_tokens_b: int,
) -> ModelMultiHead:
    if cfg.get("mlp_layers", "single") == "double":
        layers_desc = (cfg["emb_dim"], max(1, cfg["emb_dim"] // 2))
    else:
        layers_desc = (cfg["emb_dim"],)

    dropout_rate = 0.1

    backbone = ImagingOnlyBackbone(
        emb_dim=cfg["emb_dim"],
        patch_dim=patch_dim,
        max_num_tokens_b=max_num_tokens_b,
        depth_b=cfg["depth_b"],
        heads_b=cfg["heads_b"],
        dropout_rate=dropout_rate,
    )

    model = ModelMultiHead(
        backbone=backbone,
        key_out_features="model.backbone_features",
        backbone_args=[
            "data.input.img.tumor3d.patches",
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
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg["num_workers"])

    train_dataset = GISTDataset.dataset(
        data_dir_img=data_paths["img"],
        data_dir_seg=data_paths["seg"],
        clinical_csv_path=data_paths["csv"],
        train=True,
        sample_ids=train_ids,

        patch_size_zyx=patch_size_zyx,
        min_tumor_frac=min_tumor_frac,
        pad_to_multiple=PAD_TO_MULTIPLE_FIXED,
        pad_val=PAD_VAL_FIXED,
        angle_range_deg=angle_range_deg,
        rotation_prob=ROTATION_PROB_FIXED,
        dropout_p=0.0,
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

        patch_size_zyx=patch_size_zyx,
        min_tumor_frac=min_tumor_frac,
        pad_to_multiple=PAD_TO_MULTIPLE_FIXED,
        pad_val=PAD_VAL_FIXED,
        angle_range_deg=(0.0, 0.0),
        rotation_prob=0.0,
        dropout_p=0.0,
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

    lr_hp = Float("lr", (5e-5, 5e-4), default=base_cfg.get("lr", 1.5e-4), log=True)
    wd_hp = Float("wd", (1e-5, 3e-4), default=base_cfg.get("wd", 5e-5), log=True)

    mlp_hp = Categorical("mlp_layers", ["single", "double"], default=base_cfg["mlp_layers"])
    mtf_hp = Categorical("min_tumor_frac", [0.2, 0.3, 0.4], default=base_cfg["min_tumor_frac"])
    ia_hp = Categorical("imaging_aug_deg", [0, 5, 10], default=base_cfg["imaging_aug_deg"])

    lora_r_hp = Categorical("lora_r", [2, 4, 8], default=base_cfg["lora_r"])
    lora_ratio_hp = Categorical("lora_ratio", [0.5, 1.0, 2.0], default=base_cfg["lora_ratio"])

    cs.add([
        lr_hp, wd_hp,
        mlp_hp, mtf_hp, ia_hp,
        lora_r_hp, lora_ratio_hp,
    ])

    return cs


def cfg_from_config(config: Configuration, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    conf = dict(config)

    cfg["lr"] = float(conf["lr"])
    cfg["wd"] = float(conf["wd"])

    cfg["mlp_layers"] = str(conf["mlp_layers"])

    cfg["min_tumor_frac"] = float(conf["min_tumor_frac"])
    cfg["imaging_aug_deg"] = float(conf["imaging_aug_deg"])

    cfg["lora_r"] = int(conf["lora_r"])
    cfg["lora_ratio"] = float(conf["lora_ratio"])
    cfg["lora_alpha"] = float(cfg["lora_r"] * cfg["lora_ratio"])
    cfg["lora_target"] = LORA_TARGET_FIXED

    cfg["lora_dropout"] = float(LORA_DROPOUT_FIXED)
    cfg["lora_weight_decay"] = float(LORA_WEIGHT_DECAY_FIXED)

    cfg["test_center"] = TEST_CENTER
    return cfg


# =============================
# Train one trial
# =============================

def train_one_trial(
    run_dir: str,
    cfg: Dict[str, Any],
    data_paths: Dict[str, str],
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

    model = build_model(cfg, patch_dim, max_num_tokens_b)
    load_pretrained_enc_b_into_backbone(model, PRETRAINED_ENCB_PATH)

    enc_b = model.backbone.enc_b
    freeze_module(enc_b)

    lora_report = apply_lora_to_enc_b_toq_tov(
        enc_b=enc_b,
        emb_dim=int(cfg["emb_dim"]),
        r=int(cfg["lora_r"]),
        alpha=float(cfg["lora_alpha"]),
        dropout=float(cfg["lora_dropout"]),
        target=str(cfg["lora_target"]),
    )
    lora_params = collect_lora_params(enc_b)
    if len(lora_params) == 0:
        raise RuntimeError("LoRA injection produced 0 parameters. Check name matching for to_q/to_v and emb_dim.")

    other_params = [
        p for n, p in model.named_parameters()
        if (not n.startswith("backbone.enc_b.")) and p.requires_grad
    ]

    save_json(os.path.join(run_dir, "lora_report.json"), lora_report)

    losses, train_metrics, validation_metrics, best_epoch_source = make_training_elements()

    param_groups = [
        {"params": other_params, "lr": cfg["lr"], "weight_decay": cfg["wd"]},
        {"params": lora_params, "lr": cfg["lr"], "weight_decay": cfg["lora_weight_decay"]},
    ]
    optimizer = optim.AdamW(param_groups)

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
    largest_shape_zyx: Tuple[int, int, int],
    device: torch.device,
) -> ModelMultiHead:
    patch_size_zyx = tuple(cfg["patch_size_zyx"])
    patch_dim = int(np.prod(patch_size_zyx))
    max_num_tokens_b = compute_max_num_tokens(largest_shape_zyx, patch_size_zyx)

    model = build_model(cfg, patch_dim, max_num_tokens_b)
    load_pretrained_enc_b_into_backbone(model, PRETRAINED_ENCB_PATH)

    enc_b = model.backbone.enc_b
    freeze_module(enc_b)
    apply_lora_to_enc_b_toq_tov(
        enc_b=enc_b,
        emb_dim=int(cfg["emb_dim"]),
        r=int(cfg["lora_r"]),
        alpha=float(cfg["lora_alpha"]),
        dropout=float(cfg["lora_dropout"]),
        target=str(cfg["lora_target"]),
    )

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
        train=False,
        sample_ids=val_ids,

        patch_size_zyx=patch_size_zyx,
        min_tumor_frac=min_tumor_frac,
        pad_to_multiple=PAD_TO_MULTIPLE_FIXED,
        pad_val=PAD_VAL_FIXED,
        angle_range_deg=(0.0, 0.0),
        rotation_prob=0.0,
        dropout_p=0.0,
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

    largest_shape_zyx = TARGET_SHAPE_ZYX

    base_cfg = {
        "patch_size_zyx": PATCH_SIZE_FIXED,
        "emb_dim": EMB_DIM_FIXED,
        "depth_b": DEPTH_B_FIXED,
        "heads_b": HEADS_B_FIXED,

        "lr": 1.5e-4,
        "wd": 5e-5,

        "batch_size": 16,
        "num_workers": 10,

        "mlp_layers": "single",

        "min_tumor_frac": 0.3,
        "imaging_aug_deg": 0.0,

        "lora_r": 4,
        "lora_ratio": 1.0,
        "lora_target": LORA_TARGET_FIXED,

        "lora_dropout": float(LORA_DROPOUT_FIXED),
        "lora_weight_decay": float(LORA_WEIGHT_DECAY_FIXED),

        "test_center": TEST_CENTER,
    }
    base_cfg["lora_alpha"] = float(base_cfg["lora_r"] * base_cfg["lora_ratio"])

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
    print(
        f"[FIXED] patch_size_zyx={base_cfg['patch_size_zyx']} | "
        f"emb_dim={base_cfg['emb_dim']} | depth_b={base_cfg['depth_b']} | heads_b={base_cfg['heads_b']}"
    )
    print(
        f"[FIXED] LoRA target={LORA_TARGET_FIXED} | "
        f"LoRA dropout={LORA_DROPOUT_FIXED} | LoRA wd={LORA_WEIGHT_DECAY_FIXED}"
    )
    print(f"[FineTuning] Max configurations = {MAX_CONFIGS}")

    cs = create_configspace(base_cfg)

    scenario = Scenario(
        cs,
        deterministic=True,
        n_trials=MAX_CONFIGS,
        seed=GLOBAL_SEED,
        output_directory=SMAC_OUTPUT_DIR,
        name=SMAC_RUN_NAME,
    )

    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario,
        n_configs=min(10, MAX_CONFIGS),
    )
    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario,
        max_config_calls=1,
    )

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
            "pretrained_encb_path": PRETRAINED_ENCB_PATH,
            "ablation_type": "imaging_only",
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