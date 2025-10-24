import sys
import os
import json
import copy
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from collections import OrderedDict

from fuse.dl.models.model_multihead import ModelMultiHead
from fuse.dl.models.backbones.backbone_transformer import CrossAttentionTransformerEncoder
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

from HeadMLPClassifier import HeadMLPClassifier

# === Folder names to avoid interference with pooledCV ===
RUNS_DIR_NAME = "runs_locoCV"
SPLITS_DIR_NAME = "splits_locoCV"

# Import the LOCO-CV dataset module and set its splits dir to a unique name
import GIST_dataset_LOCOCV as GDL  # module import so we can patch its defaults
GDL.DEFAULT_SPLITS_DIR = SPLITS_DIR_NAME

# Use the function after patching
gist_dataloaders = GDL.gist_dataloaders  # returns (train, val, test)

# =============================
# Keys in batch dicts
# =============================
KEY_PROB = "model.prob.TKI_Classification"
KEY_LOGITS = "model.logits.TKI_Classification"
KEY_TARGET = "data.input.clinical.raw.TKIResponse"
KEY_TARGET_F = "data.input.clinical.raw.TKIResponse.f"
KEY_SAMPLE_ID = "data.sample_id"
KEY_CENTER = "data.input.clinical.raw.Center"

# What to print per run
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

def ceil_div(a, b): return (a + b - 1) // b

def compute_max_num_tokens(largest_tumor: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> int:
    zc = ceil_div(largest_tumor[0], patch_size[0])
    yc = ceil_div(largest_tumor[1], patch_size[1])
    xc = ceil_div(largest_tumor[2], patch_size[2])
    return int(zc * yc * xc)

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

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
        if float(x) == 0.0: return "0"
        s = np.format_float_scientific(float(x), precision=0, unique=True, exp_digits=1)
        return s.replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")
    def fmt_ps(ps) -> str: return "x".join(str(int(v)) for v in ps)
    def fmt_mlp(v: str) -> str: return "S" if str(v).lower().startswith("s") else "D"
    def fmt_pct10(v: float) -> str: return f"{int(round(float(v)*10)):02d}"
    def clean(s: str) -> str: return re.sub(r"[^A-Za-z0-9\-x]", "", s)

    parts = []
    for k in ORDER:
        if k not in cfg: continue
        ab = ABBR[k]; v = cfg[k]
        if k == "patch_size": val = fmt_ps(v)
        elif k in ("emb_dim","depth_a","heads_a","depth_b","heads_b","depth_cross_attn","heads_cross","imaging_aug_deg"):
            val = str(int(v))
        elif k in ("lr","wd"): val = fmt_lr(v)
        elif k == "mlp_layers": val = fmt_mlp(v)
        elif k in ("mask_pad_thresh","clinical_aug"): val = fmt_pct10(v)
        else: val = str(v)
        parts.append(f"{ab}{clean(val)}")
    slug = "-".join(parts)
    if len(slug) > max_len:
        cfg_subset = {k: cfg.get(k) for k in ORDER if k in cfg}
        h = hashlib.sha1(json.dumps(cfg_subset, sort_keys=True, default=str).encode()).hexdigest()[:8]
        slug = slug[: max_len - (len(h) + 2)] + "-h" + h
    return slug

def build_model(cfg: Dict[str, Any], embed_num_tokens: Dict[str, int], patch_dim: int, max_num_tokens_b: int) -> ModelMultiHead:
    # MLP head layout from cfg: "single"=(emb,), "double"=(emb, emb/2)
    if cfg.get("mlp_layers", "single") == "double":
        layers_desc = (cfg["emb_dim"], max(1, cfg["emb_dim"] // 2))
    else:
        layers_desc = (cfg["emb_dim"],)

    backbone = CrossAttentionTransformerEncoder(
        emb_dim=cfg["emb_dim"],
        # clinical stream
        num_tokens_a=16,
        max_seq_len_a=16,
        depth_a=cfg.get("depth_a", 2),
        heads_a=cfg.get("heads_a", 2),
        # imaging stream
        num_tokens_b=None,
        max_seq_len_b=max_num_tokens_b,
        depth_b=cfg.get("depth_b", 2),
        heads_b=cfg.get("heads_b", 4),
        # cross attention
        depth_cross_attn=cfg.get("depth_cross_attn", 2),
        output_dim=cfg["emb_dim"],
        context="both",
        kwargs_wrapper_a={'embed_num_tokens': embed_num_tokens, 'use_abs_pos_emb': False, 'emb_dropout': 0.1, 'post_emb_norm': True},
        kwargs_wrapper_b={'token_emb': nn.Linear(patch_dim, cfg["emb_dim"]), 'use_abs_pos_emb': False, 'emb_dropout': 0.1, 'post_emb_norm': True, 'return_only_embed': True},
        kwargs_encoder_a={'attn_dropout': 0.1, 'ff_dropout': 0.1, 'ff_glu': False, 'use_rmsnorm': True},
        kwargs_encoder_b={'attn_dropout': 0.1, 'ff_dropout': 0.1, 'ff_glu': False, 'use_rmsnorm': True, 'rotary_pos_emb': True},
        kwargs_cross_attn={
            'heads': cfg.get("heads_cross", 2),
            'attn_dropout': 0.1,
            'ff_dropout': 0.1,
            'use_rmsnorm': True,
            'residual_attn': False,
            'cross_residual_attn': False
        }
    )

    model = ModelMultiHead(
        backbone=backbone,
        key_out_features='model.backbone_features',
        backbone_args=[
            'data.input.clinical.vector',       # xa
            'data.input.img.tumor3d.patches',   # xb
            'model.embed_ids_a',                # embed_ids_a
            'model.embed_mask_a',               # mask for xa
            'model.embed_mask_b',               # mask for xb
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

def compute_val_auc_from_ckpt(
    run_dir: str,
    cfg: Dict[str, Any],
    ckpt_path: str,
    data_paths: Dict[str, str],
    embed_num_tokens: Dict[str, int],
    largest_tumor: Tuple[int, int, int],
    seed: int,
) -> float:
    # Quick val AUC from a saved ckpt (helper)
    seed_everything(seed)

    patch_size = cfg["patch_size"]
    patch_dim = int(np.prod(patch_size))
    max_num_tokens_b = compute_max_num_tokens(largest_tumor, patch_size)

    _, val_dl, _ = gist_dataloaders(
        data_dir_img=data_paths["img"],
        data_dir_seg=data_paths["seg"],
        clinical_csv_path=data_paths["csv"],
        test_center=cfg["test_center"],
        patch_size=patch_size,
        largest_tumor=largest_tumor,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        seed=seed,
        val_frac=0.20,
        save_split=True,
        angle_range=(-cfg["imaging_aug_deg"], cfg["imaging_aug_deg"]),
        mask_pad_threshold=cfg["mask_pad_thresh"],
        dropout_p=cfg["clinical_aug"],
    )

    model = build_model(cfg, embed_num_tokens, patch_dim, max_num_tokens_b)
    losses, train_metrics, validation_metrics, best_epoch_source = make_training_elements()
    pl_module = LightningModuleDefault(
        model_dir=run_dir,
        model=model,
        losses=losses,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=None,
    )

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
    )
    metrics = trainer.validate(pl_module, dataloaders=val_dl, ckpt_path=ckpt_path)
    metrics = metrics[0] if isinstance(metrics, list) and len(metrics) else {}
    return float(metrics.get("validation.metrics.auc", float("nan")))

def train_one_candidate(
    run_dir: str,
    cfg: Dict[str, Any],
    data_paths: Dict[str, str],
    embed_num_tokens: Dict[str, int],
    largest_tumor: Tuple[int, int, int],
    seed: int,
    ckpt_path: Optional[str] = None,   # resume from ckpt if provided
) -> Tuple[float, str]:
    ensure_dir(run_dir)
    seed_everything(seed)

    patch_size = cfg["patch_size"]
    patch_dim = int(np.prod(patch_size))
    max_num_tokens_b = compute_max_num_tokens(largest_tumor, patch_size)

    train_dl, val_dl, _ = gist_dataloaders(
        data_dir_img=data_paths["img"],
        data_dir_seg=data_paths["seg"],
        clinical_csv_path=data_paths["csv"],
        test_center=cfg["test_center"],
        patch_size=patch_size,
        largest_tumor=largest_tumor,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        seed=seed,
        val_frac=0.20,
        save_split=True,
        angle_range=(-cfg["imaging_aug_deg"], cfg["imaging_aug_deg"]),
        mask_pad_threshold=cfg["mask_pad_thresh"],
        dropout_p=cfg["clinical_aug"],
    )

    model = build_model(cfg, embed_num_tokens, patch_dim, max_num_tokens_b)
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
        save_last=True,  # save last.ckpt as well
        filename="best_epoch",
        auto_insert_metric_name=False,
        verbose=True
    )

    early_stop_cb = EarlyStopping(
        monitor="validation.metrics.auc",
        patience=15,
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

    # Train (optionally resuming from last.ckpt)
    trainer.fit(pl_module, train_dl, val_dl, ckpt_path=ckpt_path)

    # --------- FLATTEN LOGS INTO run_dir ----------
    try:
        log_dir = csv_logger.log_dir  # e.g., "<run_dir>/version_0"
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
    # ----------------------------------------------

    # Evaluate using the LAST checkpoint
    best_path = os.path.join(run_dir, "best_epoch.ckpt")
    if not os.path.exists(best_path):
        raise RuntimeError(f"best_epoch.ckpt not found in {run_dir}")

    score = compute_val_auc_from_ckpt(
        run_dir=run_dir,
        cfg=cfg,
        ckpt_path=best_path,
        data_paths=data_paths,
        embed_num_tokens=embed_num_tokens,
        largest_tumor=largest_tumor,
        seed=seed,
    )
    ckpt = best_path
    return score, ckpt


def evaluate_on_test_and_save_preds(
    run_dir: str,
    cfg: Dict[str, Any],
    best_ckpt: str,  # will be the "last.ckpt" of the winner
    data_paths: Dict[str, str],
    embed_num_tokens: Dict[str, int],
    largest_tumor: Tuple[int, int, int],
    seed: int,
) -> Dict[str, float]:
    ensure_dir(run_dir)
    seed_everything(seed)

    patch_size = cfg["patch_size"]
    patch_dim = int(np.prod(patch_size))
    max_num_tokens_b = compute_max_num_tokens(largest_tumor, patch_size)

    # dataloader
    _, _, test_dl = gist_dataloaders(
        data_dir_img=data_paths["img"],
        data_dir_seg=data_paths["seg"],
        clinical_csv_path=data_paths["csv"],
        test_center=cfg["test_center"],
        patch_size=patch_size,
        largest_tumor=largest_tumor,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        seed=seed,
        val_frac=0.20,
        save_split=True,
        angle_range=(-cfg["imaging_aug_deg"], cfg["imaging_aug_deg"]),
        mask_pad_threshold=cfg["mask_pad_thresh"],
        dropout_p=cfg["clinical_aug"],
    )

    # model
    model = build_model(cfg, embed_num_tokens, patch_dim, max_num_tokens_b)
    losses, train_metrics, validation_metrics, best_epoch_source = make_training_elements()

    pl_module = LightningModuleDefault(
        model_dir=run_dir,
        model=model,
        losses=losses,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=None,
    )

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
    )

    # Evaluate metrics
    metrics = trainer.validate(pl_module, dataloaders=test_dl, ckpt_path=best_ckpt)
    metrics = metrics[0] if isinstance(metrics, list) and len(metrics) else {}

    # --- FIX: tell pl_module what to return during predict ---
    pl_module.set_predictions_keys([KEY_PROB, KEY_TARGET_F, KEY_SAMPLE_ID, KEY_CENTER])

    # Collect predictions
    pred_batches = trainer.predict(pl_module, dataloaders=test_dl, ckpt_path=best_ckpt)
    rows = []

    def tolist(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    for batch_out in pred_batches:
        probs = tolist(batch_out.get(KEY_PROB, []))
        labels = tolist(batch_out.get(KEY_TARGET_F, []))
        sids = batch_out.get(KEY_SAMPLE_ID, [])
        centers = batch_out.get(KEY_CENTER, [])
        if not isinstance(sids, (list, tuple)):
            sids = [sids]
        if not isinstance(centers, (list, tuple)):
            centers = [centers]
        n = min(len(probs), len(labels), len(sids), len(centers))
        for i in range(n):
            rows.append((str(sids[i]), str(centers[i]), float(labels[i]), float(probs[i])))

    # Save predictions
    preds_csv = os.path.join(os.path.dirname(run_dir), "test_preds.csv")
    ensure_dir(os.path.dirname(preds_csv))
    with open(preds_csv, "w") as f:
        f.write("patient_id,center,y_true,p_hat\n")
        for sid, ctr, yt, ph in rows:
            f.write(f"{sid},{ctr},{yt},{ph}\n")

    # Save metrics
    with open(os.path.join(os.path.dirname(run_dir), "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

def generate_tier_grid() -> Dict[int, List[Dict[str, Any]]]:
    # Tier 1: patch_size × emb_dim
    tier1 = []
    for pz, py, px in [(8,32,32), (8,64,64), (16,32,32), (16, 64, 64)]:
        for emb in [32, 64, 128, 256]:
            tier1.append({"patch_size": (pz, py, px), "emb_dim": emb})

    # Tier 2: lr × wd
    #tier2 = [{"lr": lr, "wd": wd} for lr in [5e-5, 1e-4, 3e-4] for wd in [0.0, 1e-3]]
    tier2 = [{"lr": lr, "wd": 1e-3} for lr in [5e-5, 1e-4, 3e-4]]

    # Tier 3: clinical encoder [layers, heads]
    tier3 = [{"depth_a": l, "heads_a": h} for (l, h) in [(1,2), (2,2), (2,4)]]

    # Tier 4: imaging encoder [layers, heads]
    tier4 = [{"depth_b": l, "heads_b": h} for (l, h) in [(2,2), (2,4), (2,8)]]

    # Tier 5: cross attention [layers, heads]
    tier5 = [{"depth_cross_attn": l, "heads_cross": h} for (l, h) in [(1,2), (2,2), (2,4)]]

    # Tier 6: classifier MLP layers
    tier6 = [{"mlp_layers": "single"}, {"mlp_layers": "double"}]

    # Tier 7: masking 3D patches (dataset param)
    tier7 = [{"mask_pad_thresh": t} for t in [0.7, 0.8, 0.9]]

    # Tier 8: clinical augmentation strength (dataset param)
    tier8 = [{"clinical_aug": a} for a in [0.0, 0.1, 0.2]]

    # Tier 9: imaging augmentation rotation degrees (dataset param)
    tier9 = [{"imaging_aug_deg": d} for d in [0, 10, 20]]

    return {1:tier1, 2:tier2, 3:tier3, 4:tier4, 5:tier5, 6:tier6, 7:tier7, 8:tier8, 9:tier9}

def merge_cfg(base: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg.update(delta)
    return cfg

def main():
    # Orchestration (folds within tiers) with simplified resume policy:
    # - If tier has winner.json -> skip
    # - Else -> clear tier folder and run from scratch
    # - After each tier -> evaluate winner on test set
    seed_everything(17)

    # Data paths
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    data_dir_img = os.path.join(data_dir, "img")
    data_dir_seg = os.path.join(data_dir, "seg")
    clinical_csv_path = os.path.join(data_dir, "GIST_clinical_data.csv")
    data_paths = {"img": data_dir_img, "seg": data_dir_seg, "csv": clinical_csv_path}

    # Clinical token sizes
    embed_num_tokens = {
        "0": 3, "1": 3, "2": 3, "3": 3, "4": 8, "5": 4, "6": 4, "7": 3, "8": 3,
        "9": 3, "10": 3, "11": 3, "12": 3, "13": 3, "14": 3, "15": 3
    }

    largest_tumor = (80, 347, 498)
    folds = ["AVL", "EMC", "LUMC", "RUMC"]

    base_cfg = {
        "patch_size": (8, 32, 32),
        "emb_dim": 16,
        "lr": 1e-4,
        "wd": 1e-3,
        "batch_size": 8,
        "num_workers": 10,
        # encoder defaults
        "depth_a": 2, "heads_a": 2,
        "depth_b": 2, "heads_b": 4,
        "depth_cross_attn": 2, "heads_cross": 2,
        # head
        "mlp_layers": "single",
        # dataset params for tiers 7–9
        "mask_pad_thresh": 0.8,
        "clinical_aug": 0.1,
        "imaging_aug_deg": 10,
    }

    TIER_GRID = generate_tier_grid()

    # Use a namespaced runs root to avoid interference with pooledCV
    runs_root = os.path.join(os.getcwd(), RUNS_DIR_NAME)
    ensure_dir(runs_root)

    for tier in range(1, 10):
        print(f"\n=== TIER {tier} ===")
        for fold in folds:
            print(f"\n--- Fold (test_center) = {fold} ---")
            tier_dir = os.path.join(runs_root, f"fold_{fold}", f"tier_{tier}")
            winner_path = os.path.join(tier_dir, "winner.json")

            # If tier completed -> skip
            if os.path.exists(winner_path):
                with open(winner_path, "r") as f:
                    winner_info = json.load(f)
                print(f"[SKIP] Completed Tier {tier} / Fold {fold}. Winner val_auc={winner_info.get('val_auc')}")
                continue

            # If tier not completed -> clear its folder and start fresh
            if os.path.exists(tier_dir):
                print(f"[RESET] Clearing incomplete Tier {tier} / Fold {fold} at {tier_dir}")
                shutil.rmtree(tier_dir)
            ensure_dir(tier_dir)

            # Build candidates from last tier winner (if exists) or base
            prev_winner_cfg = None
            if tier > 1:
                prev_win_file = os.path.join(runs_root, f"fold_{fold}", f"tier_{tier-1}", "winner.json")
                if os.path.exists(prev_win_file):
                    with open(prev_win_file, "r") as f:
                        prev_winner_cfg = json.load(f)["cfg"]

            base_for_fold = prev_winner_cfg if prev_winner_cfg is not None else base_cfg
            base_for_fold = merge_cfg(base_for_fold, {"test_center": fold})

            grid = TIER_GRID[tier]
            candidates_cfgs = [merge_cfg(base_for_fold, delta) for delta in grid]

            best_score = -1.0
            best = {"cfg": None, "val_auc": None, "ckpt": None}

            # Fresh run for all candidates
            for i, cand in enumerate(candidates_cfgs):
                slug = config_slug(cand)
                run_dir = os.path.join(tier_dir, f"cand_{i:02d}__{slug}")
                ensure_dir(run_dir)

                report = {k: cand[k] for k in REPORT_KEYS if k in cand}
                print(f"Tier {tier} | Fold {fold} | cand {i} CONFIG -> {report}")
                with open(os.path.join(run_dir, "config.json"), "w") as f:
                    json.dump(cand, f, indent=2)

                # Train from scratch
                score, ckpt = train_one_candidate(
                    run_dir=run_dir,
                    cfg=cand,
                    data_paths=data_paths,
                    embed_num_tokens=embed_num_tokens,
                    largest_tumor=largest_tumor,
                    seed=17,
                    ckpt_path=None,
                )
                with open(os.path.join(run_dir, "result.json"), "w") as f:
                    json.dump({"val_auc": score, "ckpt": ckpt}, f, indent=2)

                print(f"Tier {tier} | Fold {fold} | cand {i} | val_auc={score:.4f}")
                if score > best_score:
                    best_score = score
                    best = {"cfg": cand, "val_auc": score, "ckpt": ckpt}

            # Sanity: ensure at least one candidate completed
            if best_score < 0:
                raise RuntimeError(
                    f"No completed candidates found for Tier {tier}, Fold {fold}. "
                    f"Please investigate run directories under {tier_dir}."
                )

            # Save winner of the tier (uses LAST checkpoint for the winner)
            with open(os.path.join(tier_dir, "winner.json"), "w") as f:
                json.dump({"cfg": best["cfg"], "val_auc": best["val_auc"], "ckpt": best["ckpt"]}, f, indent=2)

            win_report = {k: best["cfg"][k] for k in REPORT_KEYS if k in best["cfg"]}
            print(f"[WINNER] Tier {tier} | Fold {fold} | val_auc={best_score:.4f} | CONFIG -> {win_report}")

            # After each tier -> evaluate on test + write per-patient predictions
            fold_root = os.path.join(runs_root, f"fold_{fold}")
            tier_test_dir = os.path.join(fold_root, f"test_eval_tier{tier}")
            metrics = evaluate_on_test_and_save_preds(
                run_dir=tier_test_dir,
                cfg=best["cfg"],
                best_ckpt=best["ckpt"],  # last.ckpt of the winner
                data_paths=data_paths,
                embed_num_tokens=embed_num_tokens,
                largest_tumor=largest_tumor,
                seed=17,
            )
            print(f"[TEST] Tier {tier} | Fold {fold} | Test metrics: {metrics}")

    print("\nAll tiers/folds processed.")

if __name__ == "__main__":
    main()

