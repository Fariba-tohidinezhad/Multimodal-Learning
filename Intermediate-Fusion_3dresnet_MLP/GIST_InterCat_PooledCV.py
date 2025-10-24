import sys
import os
import json
import copy
import random
import shutil
import numpy as np
import hashlib, re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from collections import OrderedDict

from fuse.dl.models.model_multihead import ModelMultiHead
from fuse.dl.models.backbones.backbone_resnet_3d import BackboneResnet3D
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

from Head3DGlobalFusionMasked import Head3DGlobalFusionMasked
from GIST_dataset_PooledCV import gist_dataloaders_pooled  # returns (train, val, test)

# =============================
# Keys in batch dicts
# =============================
KEY_PROB = "model.prob.TKI_Classification"
KEY_LOGITS = "model.logits.TKI_Classification"
KEY_TARGET = "data.input.clinical.raw.TKIResponse"
KEY_TARGET_F = "data.input.clinical.raw.TKIResponse.f"
KEY_SAMPLE_ID = "data.sample_id"
KEY_CENTER = "data.input.clinical.raw.Center"

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

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def config_slug(cfg: Dict[str, Any], max_len: int = 120) -> str:
    ABBR = {
        "layers": "L",               # e.g. [2,2,2,2]
        "first_channel_dim": "C",
        "lr": "LR",
        "first_stride": "FS",
        "stem_kernel_size": "K",
        "stem_stride": "SS",
        "append_mlp": "MLP",
        "head_layers": "HD",
        "use_mask": "M",
    }
    ORDER = [
        "layers",
        "first_channel_dim",
        "lr",
        "first_stride",
        "stem_kernel_size",
        "stem_stride",
        "append_mlp",
        "head_layers",
        "use_mask",
    ]
    def fmt_list(v):
        if isinstance(v, (list, tuple)):
            return "x".join(str(int(x)) for x in v)
        return str(v)

    def fmt_lr(x: float) -> str:
        if float(x) == 0.0:
            return "0"
        s = np.format_float_scientific(float(x), precision=0, unique=True, exp_digits=1)
        return s.replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")

    def clean(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9x\-]", "", s)

    # --- build slug parts ---
    parts = []
    for k in ORDER:
        if k not in cfg:
            continue
        ab = ABBR[k]
        v = cfg[k]

        if k == "lr":
            val = fmt_lr(v)
        elif k in ("layers", "stem_kernel_size", "stem_stride", "append_mlp", "head_layers"):
            val = fmt_list(v)
        elif k == "use_mask":
            val = "T" if bool(v) else "F"
        else:
            val = str(v)

        parts.append(f"{ab}{clean(val)}")

    slug = "-".join(parts)

    # --- shorten if necessary ---
    if len(slug) > max_len:
        cfg_subset = {k: cfg.get(k) for k in ORDER if k in cfg}
        h = hashlib.sha1(json.dumps(cfg_subset, sort_keys=True, default=str).encode()).hexdigest()[:8]
        slug = slug[: max_len - (len(h) + 2)] + "-h" + h

    return slug

def build_model(cfg: Dict[str, Any]) -> ModelMultiHead:
    """
    Intermediate concatenation:
      - CT -> BackboneResnet3D (no pool)
      - Clinical -> MLP
      - Head3DGlobalFusionMasked does fusion + classification
    """

    backbone = BackboneResnet3D(
        in_channels=1,
        pool=False,
        layers=cfg.get("layers", [2, 2, 2, 2]),
        first_channel_dim=cfg.get("first_channel_dim", 64),
        first_stride=cfg.get("first_stride", 1),
        stem_kernel_size=cfg.get("stem_kernel_size", (3, 7, 7)),
        stem_stride=cfg.get("stem_stride", (1, 2, 2)),
        pretrained=False,
        name="resnet3d",
    )

    model = ModelMultiHead(
        backbone=backbone,
        conv_inputs=[("data.input.img.tumor3d.fitted", 1)],
        key_out_features="model.backbone_features",
        heads=[
            Head3DGlobalFusionMasked(
                head_name="TKI_Classification",
                mode="classification",
                num_outputs=1,
                # imaging feature maps from backbone
                conv_inputs=[("model.backbone_features", backbone.out_dim)],
                pooling=cfg.get("pooling", "avg"),  # "avg" | "max" | "avgmax"
                spatial_dropout_rate=0.0,
                layers_description=cfg.get("head_layers", (256,)),
                fused_dropout_rate=cfg.get("fused_dropout", 0.1),
                # clinical tabular input
                append_features=[("data.input.clinical.encoding", 55)],
                append_layers_description=cfg.get("append_mlp", (128, 64)),
                append_dropout_rate=cfg.get("append_dropout", 0.1),
                # masked pooling
                use_mask=cfg.get("use_mask", True),
                mask_key=cfg.get("mask_key", "data.input.seg.tumor3d.fitted"),
            )
        ],
    )

    # 🔧 ensure both attributes exist (Fuse compatibility)
    if not hasattr(model, "backbone_args"):
        model.backbone_args = None
    if not hasattr(model, "conv_inputs"):
        model.conv_inputs = [("data.input.img.tumor3d.fitted", 1)]

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
    largest_tumor: Tuple[int, int, int],
    seed: int,
) -> float:
    """
    Compute validation AUC from a saved checkpoint.
    For intermediate concatenation (3D ResNet + clinical fusion) model.
    """
    seed_everything(seed)

    # --- Data (only validation split) ---
    _, val_dl, _ = gist_dataloaders_pooled(
        data_dir_img=data_paths["img"],
        data_dir_seg=data_paths["seg"],
        clinical_csv_path=data_paths["csv"],
        fold=cfg["fold"],
        n_splits=cfg.get("n_splits", 5),
        largest_tumor=largest_tumor,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        seed=seed,
        val_frac=0.20,
        save_split=True,
        angle_range=(-cfg["imaging_aug_deg"], cfg["imaging_aug_deg"]),
        dropout_p=cfg["clinical_aug"],
    )

    # --- Model & metrics ---
    model = build_model(cfg)
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
    largest_tumor: Tuple[int, int, int],
    seed: int,
    ckpt_path: Optional[str] = None,   # resume from ckpt if provided
) -> Tuple[float, str]:
    """
    Train a single candidate configuration for the intermediate concatenation model.
    Uses BackboneResnet3D + Head3DGlobalFusionMasked (CT + clinical fusion).
    """
    ensure_dir(run_dir)
    seed_everything(seed)

    # --- Data ---
    train_dl, val_dl, _ = gist_dataloaders_pooled(
        data_dir_img=data_paths["img"],
        data_dir_seg=data_paths["seg"],
        clinical_csv_path=data_paths["csv"],
        fold=cfg["fold"],
        n_splits=cfg.get("n_splits", 5),
        largest_tumor=largest_tumor,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        seed=seed,
        val_frac=0.20,
        save_split=True,
        angle_range=(-cfg["imaging_aug_deg"], cfg["imaging_aug_deg"]),
        dropout_p=cfg["clinical_aug"],
    )

    # --- Model ---
    model = build_model(cfg)
    losses, train_metrics, validation_metrics, best_epoch_source = make_training_elements()

    # --- Optimizer & LR scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=(0.5 ** 4) * cfg["lr"],
    )
    optimizers_and_lr_sch = dict(
        optimizer=optimizer,
        lr_scheduler=dict(scheduler=lr_scheduler, monitor="validation.metrics.auc")
    )

    # --- Callbacks ---
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
        patience=15,
        mode="max",
        verbose=True,
    )
    csv_logger = CSVLogger(save_dir=run_dir, name=".")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # --- Lightning module ---
    pl_module = LightningModuleDefault(
        model_dir=run_dir,
        model=model,
        losses=losses,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=optimizers_and_lr_sch,
    )

    # --- Trainer ---
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

    # --- Train ---
    trainer.fit(pl_module, train_dl, val_dl, ckpt_path=ckpt_path)

    # --- Flatten logs (for easier access) ---
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
        shutil.rmtree(log_dir, ignore_errors=True)
    except Exception as e:
        print(f"[WARN] Failed to flatten logs for {run_dir}: {e}")

    # --- Evaluate best checkpoint ---
    best_path = os.path.join(run_dir, "best_epoch.ckpt")
    if not os.path.exists(best_path):
        raise RuntimeError(f"best_epoch.ckpt not found in {run_dir}")

    score = compute_val_auc_from_ckpt(
        run_dir=run_dir,
        cfg=cfg,
        ckpt_path=best_path,
        data_paths=data_paths,
        largest_tumor=largest_tumor,
        seed=seed,
    )
    ckpt = best_path
    return score, ckpt


def evaluate_on_test_and_save_preds(
    run_dir: str,
    cfg: Dict[str, Any],
    best_ckpt: str,  # best or last checkpoint of the winner
    data_paths: Dict[str, str],
    largest_tumor: Tuple[int, int, int],
    seed: int,
) -> Dict[str, float]:
    """
    Evaluate the trained intermediate concatenation model on the test set,
    save predictions and test metrics.
    Compatible with 3D ResNet + Head3DGlobalFusionMasked.
    """
    ensure_dir(run_dir)
    seed_everything(seed)

    # --- Dataloader ---
    _, _, test_dl = gist_dataloaders_pooled(
        data_dir_img=data_paths["img"],
        data_dir_seg=data_paths["seg"],
        clinical_csv_path=data_paths["csv"],
        fold=cfg["fold"],
        n_splits=cfg.get("n_splits", 5),
        largest_tumor=largest_tumor,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        seed=seed,
        val_frac=0.20,
        save_split=True,
        angle_range=(-cfg["imaging_aug_deg"], cfg["imaging_aug_deg"]),
        dropout_p=cfg["clinical_aug"],
    )

    # --- Model ---
    model = build_model(cfg)
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

    # --- Evaluate metrics ---
    metrics = trainer.validate(pl_module, dataloaders=test_dl, ckpt_path=best_ckpt)
    metrics = metrics[0] if isinstance(metrics, list) and len(metrics) else {}

    # --- Collect predictions ---
    pl_module.set_predictions_keys([KEY_PROB, KEY_TARGET_F, KEY_SAMPLE_ID, KEY_CENTER])
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

    # --- Save predictions ---
    preds_csv = os.path.join(os.path.dirname(run_dir), "test_preds.csv")
    ensure_dir(os.path.dirname(preds_csv))
    with open(preds_csv, "w") as f:
        f.write("patient_id,center,y_true,p_hat\n")
        for sid, ctr, yt, ph in rows:
            f.write(f"{sid},{ctr},{yt},{ph}\n")

    # --- Save metrics ---
    with open(os.path.join(os.path.dirname(run_dir), "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

def generate_tier_grid() -> Dict[int, List[Dict[str, Any]]]:
    """
    Tiered search grid for Intermediate Concatenation model
    (BackboneResnet3D + Head3DGlobalFusionMasked).
    Ordered by descending influence.
    """

    tier1 = [{"layers": [1,2,2,1]}, {"layers": [2, 2, 2, 2]}, {"layers": [3, 4, 6, 3]}]
    tier2 = [{"first_channel_dim": c} for c in [16, 32, 64]]
    tier3 = [{"lr": lr} for lr in [1e-4, 3e-4, 5e-4, 1e-3]]
    tier4 = [{"first_stride": s} for s in [1, 2]]
    tier5 = [{"stem_kernel_size": k} for k in [(3,5,5), (3,7,7), (5,9,9)]]
    tier6 = [{"stem_stride": s} for s in [(1,1,1), (1,2,2), (2,2,2)]]

    # clinical MLP
    tier7 = [{"append_mlp": l} for l in [(128,64), (64,), ()]]

    # classifier MLP
    tier8 = [{"head_layers": l} for l in [(256,), (256,128)]]

    tier9 = [{"use_mask": m} for m in [True, False]]

    return {1: tier1, 2: tier2, 3: tier3, 4: tier4, 5: tier5, 6: tier6, 7: tier7, 8: tier8, 9: tier9}

def merge_cfg(base: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg.update(delta)
    return cfg

def main():
    """
    Tiered search orchestration for Intermediate Concatenation model:
    BackboneResnet3D + Head3DGlobalFusionMasked.
    Uses pooled cross-validation (runs_PooledCV + splits_PooledCV).
    """
    seed_everything(17)

    # --- Data paths ---
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    data_dir_img = os.path.join(data_dir, "img")
    data_dir_seg = os.path.join(data_dir, "seg")
    clinical_csv_path = os.path.join(data_dir, "GIST_clinical_data.csv")
    data_paths = {"img": data_dir_img, "seg": data_dir_seg, "csv": clinical_csv_path}

    # --- Experiment setup ---
    # 90%: (42, 238, 262)
    # 95%: (50, 271, 297)
    # 98%: (55, 308, 368)
    # largest: (80, 347, 498)
    largest_tumor = (50, 271, 297)
    n_splits = 5
    folds = list(range(n_splits))
    runs_root = os.path.join(os.getcwd(), "runs_PooledCV")
    ensure_dir(runs_root)

    # --- Base configuration (defaults) ---
    base_cfg = dict(
        layers=[2, 2, 2, 2],
        first_channel_dim=32,
        lr=3e-4,
        wd=1e-4,
        first_stride=1,
        stem_kernel_size=(3, 7, 7),
        stem_stride=(1, 2, 2),
        head_layers=(256,),
        append_mlp=(128,64),
        use_mask=True,
        batch_size=8,
        num_workers=10,
        imaging_aug_deg=10,
        clinical_aug=0.1,
        n_splits=n_splits,
    )

    TIER_GRID = generate_tier_grid()

    # --- Tiered training loop ---
    for tier in range(1, 10):
        print(f"\n=== TIER {tier} ===")

        for fold in folds:
            print(f"\n--- Fold {fold} ---")

            tier_dir = os.path.join(runs_root, f"fold_{fold}", f"tier_{tier}")
            winner_path = os.path.join(tier_dir, "winner.json")

            # Skip if tier already completed
            if os.path.exists(winner_path):
                with open(winner_path, "r") as f:
                    winner_info = json.load(f)
                print(f"[SKIP] Tier {tier} Fold {fold} already completed. val_auc={winner_info.get('val_auc'):.4f}")
                continue

            # Reset old folder
            if os.path.exists(tier_dir):
                print(f"[RESET] Clearing {tier_dir}")
                shutil.rmtree(tier_dir)
            ensure_dir(tier_dir)

            # Get previous tier winner (if exists)
            prev_cfg_path = os.path.join(runs_root, f"fold_{fold}", f"tier_{tier-1}", "winner.json")
            if tier > 1 and os.path.exists(prev_cfg_path):
                with open(prev_cfg_path, "r") as f:
                    prev_winner_cfg = json.load(f)["cfg"]
            else:
                prev_winner_cfg = None

            base_for_fold = merge_cfg(prev_winner_cfg if prev_winner_cfg else base_cfg, {"fold": fold})
            grid = TIER_GRID[tier]
            candidates_cfgs = [merge_cfg(base_for_fold, delta) for delta in grid]

            # --- Train all candidates ---
            best_score, best = -1.0, {"cfg": None, "val_auc": None, "ckpt": None}

            for i, cand in enumerate(candidates_cfgs):
                slug = config_slug(cand)
                run_dir = os.path.join(tier_dir, f"cand_{i:02d}__{slug}")
                ensure_dir(run_dir)

                print(f"Tier {tier} | Fold {fold} | Candidate {i} -> {slug}")
                with open(os.path.join(run_dir, "config.json"), "w") as f:
                    json.dump(cand, f, indent=2)

                try:
                    score, ckpt = train_one_candidate(
                        run_dir=run_dir,
                        cfg=cand,
                        data_paths=data_paths,
                        largest_tumor=largest_tumor,
                        seed=17,
                        ckpt_path=None,
                    )
                    with open(os.path.join(run_dir, "result.json"), "w") as f:
                        json.dump({"val_auc": score, "ckpt": ckpt}, f, indent=2)

                    print(f"    val_auc = {score:.4f}")
                    if score > best_score:
                        best_score = score
                        best = {"cfg": cand, "val_auc": score, "ckpt": ckpt}

                except Exception as e:
                    print(f"[ERROR] Tier {tier} | Fold {fold} | Candidate {i} failed: {e}")
                    continue

            # --- Save winner ---
            if best["cfg"] is None:
                raise RuntimeError(f"No successful candidates in Tier {tier}, Fold {fold}")

            with open(winner_path, "w") as f:
                json.dump(best, f, indent=2)

            print(f"[WINNER] Tier {tier} | Fold {fold} | val_auc={best_score:.4f}")

            # --- Evaluate on test ---
            fold_root = os.path.join(runs_root, f"fold_{fold}")
            tier_test_dir = os.path.join(fold_root, f"test_eval_tier{tier}")
            metrics = evaluate_on_test_and_save_preds(
                run_dir=tier_test_dir,
                cfg=best["cfg"],
                best_ckpt=best["ckpt"],
                data_paths=data_paths,
                largest_tumor=largest_tumor,
                seed=17,
            )
            print(f"[TEST] Tier {tier} | Fold {fold} | {metrics}")

    print("\nAll tiers/folds completed.")

if __name__ == "__main__":
    main()

