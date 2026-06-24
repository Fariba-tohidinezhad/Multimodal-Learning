# =========================
# PT_MAE_Main.py
# =========================
import os
import math
import json
import time
import random
import csv
from typing import Tuple, List, Dict, Any, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from x_transformers import Encoder, TransformerWrapper

from PT_MAE_Dataset import GISTMAEPretrainDataset


# ============================================================
# CONFIG (EDIT HERE)
# ============================================================
CFG: Dict[str, Any] = {
    # --- required paths ---
    "data_dir_img": r"/gpfs/home4/ftohidinezhad/fuse-med-ml/GIST/data/DIAG_CTs",
    "data_dir_seg": r"/gpfs/home4/ftohidinezhad/fuse-med-ml/GIST/data/DIAG_CTs/Segmentations",

    # --- outputs ---
    "out_dir": r"/gpfs/home4/ftohidinezhad/fuse-med-ml/GIST/GIST_CrossAttention/PreTrain/PT_Output",
    "run_name": "mae_encb_ps16x64x64_ed64_db3_hb4",

    # --- dataset params ---
    "crop_size_zyx": (64, 256, 256),
    "patch_size_zyx": (16, 64, 64),
    "min_body_frac": 0.30,

    # --- split ---
    "val_frac": 0.10,

    # --- training ---
    "epochs": 200,
    "batch_size": 32,
    "num_workers": 10,
    "seed": 17,

    "lr": 2e-4,
    "wd": 0.05,
    "warmup_frac": 0.10,
    "grad_clip": 1.0,
    "mask_ratio": 0.75,

    # --- encoder (FineTuning winner) ---
    "emb_dim": 64,
    "depth_b": 3,
    "heads_b": 4,

    # --- decoder (lightweight) ---
    "decoder_dim": 64,
    "decoder_depth": 2,
    "decoder_heads": 4,

    # --- AMP ---
    "amp_dtype": "bf16",  # "bf16" for A100 or "fp16" for RTX2050

    # --- optional downsampling ---
    "use_downsample": False,
    "downsample_zyx": (2, 4, 4),

    # --- resume (optional) ---
    "resume_ckpt": "",

    # --- logging ---
    "print_every_steps": 50,
}
# ============================================================


# ----------------------------
# Reproducibility
# ----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ----------------------------
# CSV logging helpers
# ----------------------------
def csv_init_if_needed(csv_path: str, header: List[str]) -> None:
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()


def csv_append_row(csv_path: str, header: List[str], row: Dict[str, Any]) -> None:
    safe_row = {k: row.get(k, "") for k in header}
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writerow(safe_row)


# ----------------------------
# Stratified split by center-code (prefix before "_")
# ----------------------------
def center_code_from_sample_id(sample_id: str) -> str:
    sid = str(sample_id)
    if "_" not in sid:
        return "UNKNOWN"
    return sid.split("_", 1)[0]


def stratified_split_by_center_code(
    sample_ids: List[str],
    val_frac: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Stratify by center-code (e.g., 101/102/103/104). Keeps ~val_frac per center.
    Guarantees (if possible): for centers with >=2 samples => at least 1 in val and 1 in train.
    """
    assert 0.0 < val_frac < 1.0
    rng = random.Random(seed)

    by_center = defaultdict(list)
    for sid in sample_ids:
        by_center[center_code_from_sample_id(sid)].append(sid)

    train_ids: List[str] = []
    val_ids: List[str] = []

    for c, ids in by_center.items():
        ids = ids.copy()
        rng.shuffle(ids)

        n = len(ids)
        n_val = int(round(n * val_frac))

        if n >= 2:
            n_val = max(1, n_val)
            n_val = min(n_val, n - 1)
        else:
            n_val = 0  # singletons stay in train

        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids


# ----------------------------
# Optional patch downsampling (selectable)
# ----------------------------
def downsample_patch_tokens_avgpool3d(
    patches: torch.Tensor,
    patch_size_zyx: Tuple[int, int, int],
    downsample_zyx: Tuple[int, int, int],
) -> torch.Tensor:
    """
    patches: [B, N, pz*py*px]
    Return:  [B, N, (pz/dz)*(py/dy)*(px/dx)]
    """
    if patches.ndim != 3:
        raise ValueError(f"Expected patches [B,N,D], got shape={tuple(patches.shape)}")

    B, N, D = patches.shape
    pz, py, px = patch_size_zyx
    dz, dy, dx = downsample_zyx

    expected = pz * py * px
    if D != expected:
        raise ValueError(f"Patch dim mismatch: got {D}, expected {expected} from patch_size_zyx={patch_size_zyx}")

    if (pz % dz) or (py % dy) or (px % dx):
        raise ValueError(f"Downsample factors {downsample_zyx} must divide patch_size_zyx={patch_size_zyx}")

    x = patches.view(B * N, 1, pz, py, px)
    x = F.avg_pool3d(x, kernel_size=(dz, dy, dx), stride=(dz, dy, dx))
    x = x.flatten(1).view(B, N, -1)
    return x


# ----------------------------
# LR schedule (warmup + cosine)
# ----------------------------
def lr_at_step(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


# ----------------------------
# Build enc_b EXACTLY like downstream
# ----------------------------
def build_enc_b_downstream_compatible(
    patch_dim_in: int,
    emb_dim: int,
    depth_b: int,
    heads_b: int,
    max_seq_len_b: int,
) -> TransformerWrapper:
    enc_b = TransformerWrapper(
        num_tokens=None,
        max_seq_len=max_seq_len_b,
        token_emb=nn.Linear(patch_dim_in, emb_dim),
        use_abs_pos_emb=False,
        emb_dropout=0.1,
        post_emb_norm=True,
        return_only_embed=True,
        attn_layers=Encoder(
            dim=emb_dim,
            depth=depth_b,
            heads=heads_b,
            attn_dropout=0.1,
            ff_dropout=0.1,
            ff_glu=False,
            use_rmsnorm=True,
            rotary_pos_emb=True,
        ),
    )
    return enc_b


# ----------------------------
# MAE model (encoder = enc_b; decoder = lightweight)
# ----------------------------
class MAEWithEncB(nn.Module):
    def __init__(
        self,
        enc_b: TransformerWrapper,
        num_tokens: int,
        patch_dim_in: int,
        emb_dim: int,
        mask_ratio: float,
        decoder_dim: int,
        decoder_depth: int,
        decoder_heads: int,
    ):
        super().__init__()
        self.enc_b = enc_b
        self.num_tokens = int(num_tokens)
        self.patch_dim_in = int(patch_dim_in)
        self.emb_dim = int(emb_dim)
        self.mask_ratio = float(mask_ratio)

        self.dec_in = nn.Linear(emb_dim, decoder_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.dec_pos = nn.Parameter(torch.zeros(1, self.num_tokens, decoder_dim))
        nn.init.trunc_normal_(self.dec_pos, std=0.02)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=decoder_depth)

        self.dec_out = nn.Linear(decoder_dim, patch_dim_in)

    @torch.no_grad()
    def _make_masks(self, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        valid: [B, N] bool where True=valid token
        Returns:
          visible:   [B, N] True if token is visible to encoder (subset of valid)
          loss_mask: [B, N] True if token is masked AND valid (reconstruction loss)
        """
        B, N = valid.shape
        visible = torch.zeros((B, N), dtype=torch.bool, device=valid.device)
        loss_mask = torch.zeros((B, N), dtype=torch.bool, device=valid.device)

        for b in range(B):
            idx = torch.where(valid[b])[0]
            n_valid = int(idx.numel())
            if n_valid == 0:
                continue

            n_keep = max(1, int(round(n_valid * (1.0 - self.mask_ratio))))
            perm = idx[torch.randperm(n_valid, device=valid.device)]
            keep_idx = perm[:n_keep]
            mask_idx = perm[n_keep:]

            visible[b, keep_idx] = True
            loss_mask[b, mask_idx] = True

        return visible, loss_mask

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x:     [B, N, patch_dim_in]
        valid: [B, N] bool (True=valid token)
        """
        if x.ndim != 3 or valid.ndim != 2:
            raise ValueError(f"Bad shapes: x={tuple(x.shape)} valid={tuple(valid.shape)}")

        B, N, D = x.shape
        if N != self.num_tokens:
            raise ValueError(f"Expected N={self.num_tokens}, got {N}")
        if D != self.patch_dim_in:
            raise ValueError(f"Expected patch_dim_in={self.patch_dim_in}, got {D}")
        if valid.shape[0] != B or valid.shape[1] != N:
            raise ValueError(f"valid must be [B,N]; got {tuple(valid.shape)} vs B={B},N={N}")

        visible, loss_mask = self._make_masks(valid)
        enc_mask = visible & valid  # True = keep/attend

        z = self.enc_b(x, mask=enc_mask, return_embeddings=True)  # [B, N, emb_dim]

        d = self.dec_in(z)  # [B, N, decoder_dim]
        mask_tok = self.mask_token.expand(B, N, -1)
        d = torch.where(enc_mask.unsqueeze(-1), d, mask_tok)
        d = d + self.dec_pos

        dec_key_padding = ~valid  # True = ignore
        d = self.decoder(d, src_key_padding_mask=dec_key_padding)

        pred = self.dec_out(d)  # [B, N, patch_dim_in]

        denom = loss_mask.sum().clamp(min=1).float()
        per_token = ((pred - x) ** 2).mean(dim=-1)  # [B, N]
        loss = (per_token * loss_mask.float()).sum() / denom

        dbg = {
            "mask_ratio_actual": (loss_mask.sum().float() / valid.sum().clamp(min=1).float()).detach(),
            "num_valid": valid.sum().detach(),
            "valid_frac": (valid.float().mean()).detach(),
        }
        return loss, dbg


# ----------------------------
# Collate (Fuse NDict -> tensors)
# ----------------------------
def fuse_collate(batch: List[Any]) -> Dict[str, Any]:
    sids: List[str] = []
    patches: List[torch.Tensor] = []
    valids: List[torch.Tensor] = []

    for sample in batch:
        sids.append(str(sample["data.sample_id"]))
        patches.append(sample["data.input.img.patches"])
        valids.append(sample["model.embed_mask_b"])

    patches_t = torch.stack(patches, dim=0)
    valids_t = torch.stack(valids, dim=0)
    return {"sample_id": sids, "patches": patches_t, "valid": valids_t}


# ----------------------------
# Checkpoint helpers
# ----------------------------
def save_ckpt(path: str, model: nn.Module, opt: torch.optim.Optimizer, epoch: int, step: int, cfg: dict) -> None:
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "step": step, "cfg": cfg}, path)


def load_ckpt(path: str, model: nn.Module, opt: torch.optim.Optimizer, device: str) -> Tuple[int, int, dict]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    opt.load_state_dict(ckpt["opt"])
    return int(ckpt["epoch"]), int(ckpt["step"]), ckpt.get("cfg", {})


def save_enc_b_only(path: str, enc_b: TransformerWrapper) -> None:
    torch.save(enc_b.state_dict(), path)


# ----------------------------
# Validation epoch
# ----------------------------
@torch.no_grad()
def run_val_epoch(
    model: nn.Module,
    dl_val: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    use_down: bool,
    patch_size_zyx: Tuple[int, int, int],
    downsample_zyx: Tuple[int, int, int],
) -> float:
    model.eval()

    pz, py, px = patch_size_zyx
    loss_sum = 0.0
    n_batches = 0

    for batch in dl_val:
        patches = batch["patches"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)

        if use_down:
            patches = downsample_patch_tokens_avgpool3d(
                patches=patches,
                patch_size_zyx=(pz, py, px),
                downsample_zyx=downsample_zyx,
            )

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss, _ = model(patches, valid)
        else:
            loss, _ = model(patches, valid)

        loss_sum += float(loss.item())
        n_batches += 1

    model.train()
    return loss_sum / max(1, n_batches)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    for k in ("data_dir_img", "data_dir_seg"):
        if not os.path.isdir(CFG[k]):
            raise RuntimeError(f"{k} does not exist or is not a folder: {CFG[k]}")

    seed_everything(int(CFG["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if str(CFG["amp_dtype"]).lower() == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda") if (use_amp and amp_dtype == torch.float16) else None

    run_dir = os.path.join(CFG["out_dir"], CFG["run_name"])
    ensure_dir(run_dir)
    save_json(os.path.join(run_dir, "config.json"), CFG)

    # ---- CSV log setup ----
    csv_path = os.path.join(run_dir, "train_log.csv")
    CSV_HEADER = [
        "record_type",
        "timestamp_unix",
        "epoch",
        "global_step",
        "steps_per_epoch",
        "total_steps",
        "warmup_steps",
        "lr",
        "loss",  # step loss (train) or last batch loss (epoch row)
        "mask_ratio_actual",
        "num_valid",
        "valid_frac",
        "epoch_loss_mean",      # train
        "val_loss_mean",        # val recon
        "epoch_mask_mean",
        "epoch_valid_frac_mean",
        "epoch_num_valid_mean",
        "ckpt_path",
        "enc_b_path",
        "grad_clip",
        "use_amp",
        "amp_dtype",
        "split_train_n",
        "split_val_n",
    ]
    csv_init_if_needed(csv_path, CSV_HEADER)

    # ---- split ids (10% per center-code) ----
    all_ids = list(GISTMAEPretrainDataset.sample_ids(CFG["data_dir_img"]))
    train_ids, val_ids = stratified_split_by_center_code(
        sample_ids=all_ids,
        val_frac=float(CFG["val_frac"]),
        seed=int(CFG["seed"]),
    )
    print(f"[SPLIT] total={len(all_ids)} train={len(train_ids)} val={len(val_ids)}")

    # log split once
    csv_append_row(
        csv_path,
        CSV_HEADER,
        {
            "record_type": "split",
            "timestamp_unix": int(time.time()),
            "split_train_n": len(train_ids),
            "split_val_n": len(val_ids),
        },
    )

    # ---- datasets ----
    ds_train = GISTMAEPretrainDataset.dataset(
        data_dir_img=CFG["data_dir_img"],
        data_dir_seg=CFG["data_dir_seg"],
        sample_ids=train_ids,
        crop_size_zyx=tuple(CFG["crop_size_zyx"]),
        patch_size_zyx=tuple(CFG["patch_size_zyx"]),
        min_body_frac=float(CFG["min_body_frac"]),
    )

    ds_val = GISTMAEPretrainDataset.dataset(
        data_dir_img=CFG["data_dir_img"],
        data_dir_seg=CFG["data_dir_seg"],
        sample_ids=val_ids,
        crop_size_zyx=tuple(CFG["crop_size_zyx"]),
        patch_size_zyx=tuple(CFG["patch_size_zyx"]),
        min_body_frac=float(CFG["min_body_frac"]),
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=int(CFG["batch_size"]),
        shuffle=True,
        num_workers=int(CFG["num_workers"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=(int(CFG["num_workers"]) > 0),
        collate_fn=fuse_collate,
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=int(CFG["batch_size"]),
        shuffle=False,
        num_workers=int(CFG["num_workers"]),
        pin_memory=True,
        drop_last=False,
        persistent_workers=(int(CFG["num_workers"]) > 0),
        collate_fn=fuse_collate,
    )

    # ---- infer N and patch dim ----
    one = next(iter(dl_train))
    patches_raw = one["patches"]  # [B, N, Draw]
    _, N, Draw = patches_raw.shape

    pz, py, px = tuple(CFG["patch_size_zyx"])
    expected_raw = int(pz * py * px)
    if int(Draw) != expected_raw:
        raise RuntimeError(
            f"Dataset patch dim={int(Draw)} but expected {expected_raw} from patch_size_zyx={CFG['patch_size_zyx']}"
        )

    use_down = bool(CFG["use_downsample"])
    if use_down:
        dz, dy, dx = tuple(CFG["downsample_zyx"])
        if (pz % dz) or (py % dy) or (px % dx):
            raise RuntimeError(
                f"downsample_zyx={CFG['downsample_zyx']} must divide patch_size_zyx={CFG['patch_size_zyx']}"
            )
        patch_dim_in = int((pz // dz) * (py // dy) * (px // dx))
    else:
        patch_dim_in = expected_raw

    # ---- build model ----
    enc_b = build_enc_b_downstream_compatible(
        patch_dim_in=patch_dim_in,
        emb_dim=int(CFG["emb_dim"]),
        depth_b=int(CFG["depth_b"]),
        heads_b=int(CFG["heads_b"]),
        max_seq_len_b=int(N),
    )

    model = MAEWithEncB(
        enc_b=enc_b,
        num_tokens=int(N),
        patch_dim_in=int(patch_dim_in),
        emb_dim=int(CFG["emb_dim"]),
        mask_ratio=float(CFG["mask_ratio"]),
        decoder_dim=int(CFG["decoder_dim"]),
        decoder_depth=int(CFG["decoder_depth"]),
        decoder_heads=int(CFG["decoder_heads"]),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG["lr"]), weight_decay=float(CFG["wd"]))

    # ---- resume ----
    start_epoch = 0
    global_step = 0
    if CFG["resume_ckpt"] and os.path.exists(CFG["resume_ckpt"]):
        start_epoch, global_step, _ = load_ckpt(CFG["resume_ckpt"], model, opt, device=str(device))
        print(f"[RESUME] epoch={start_epoch} step={global_step} from {CFG['resume_ckpt']}")
        csv_append_row(
            csv_path,
            CSV_HEADER,
            {
                "record_type": "resume",
                "timestamp_unix": int(time.time()),
                "epoch": start_epoch,
                "global_step": global_step,
                "ckpt_path": CFG["resume_ckpt"],
            },
        )

    # ---- schedule ----
    steps_per_epoch = len(dl_train)
    total_steps = int(CFG["epochs"]) * steps_per_epoch
    warmup_steps = int(round(float(CFG["warmup_frac"]) * total_steps))

    print(f"[INFO] device={device} amp={use_amp} amp_dtype={CFG['amp_dtype']}")
    print(f"[INFO] N_tokens={N} raw_patch_dim={Draw} patch_dim_in={patch_dim_in} use_downsample={use_down}")
    print(f"[INFO] train_steps/epoch={steps_per_epoch} total_steps={total_steps} warmup_steps={warmup_steps}")
    print(f"[INFO] run_dir={run_dir}")
    print(f"[INFO] csv_log={csv_path}")

    # ---- training loop ----
    model.train()
    t_print = time.time()

    epochs = int(CFG["epochs"])
    save_epochs = set(range(10, epochs + 1, 10))
    base_lr = float(CFG["lr"])
    grad_clip = float(CFG["grad_clip"])
    print_every = int(CFG.get("print_every_steps", 50))

    last_lr: float = 0.0
    last_loss: float = 0.0
    last_dbg: Dict[str, float] = {"mask_ratio_actual": 0.0, "num_valid": 0.0, "valid_frac": 0.0}

    for epoch in range(start_epoch, epochs):
        epoch_loss_sum = 0.0
        epoch_mask_sum = 0.0
        epoch_valid_frac_sum = 0.0
        epoch_num_valid_sum = 0.0
        epoch_batches = 0

        for batch in dl_train:
            patches = batch["patches"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            if use_down:
                patches = downsample_patch_tokens_avgpool3d(
                    patches=patches,
                    patch_size_zyx=(pz, py, px),
                    downsample_zyx=tuple(CFG["downsample_zyx"]),
                )

            lr = lr_at_step(global_step, total_steps, base_lr, warmup_steps)
            for pg in opt.param_groups:
                pg["lr"] = lr

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss, dbg = model(patches, valid)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
            else:
                loss, dbg = model(patches, valid)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            global_step += 1

            epoch_batches += 1
            loss_val = float(loss.detach().item())
            mask_val = float(dbg["mask_ratio_actual"].detach().item())
            valid_frac_val = float(dbg["valid_frac"].detach().item())
            num_valid_val = float(dbg["num_valid"].detach().item())

            epoch_loss_sum += loss_val
            epoch_mask_sum += mask_val
            epoch_valid_frac_sum += valid_frac_val
            epoch_num_valid_sum += num_valid_val

            last_lr = float(lr)
            last_loss = loss_val
            last_dbg = {
                "mask_ratio_actual": mask_val,
                "num_valid": num_valid_val,
                "valid_frac": valid_frac_val,
            }

            if (global_step % print_every) == 0:
                dt = (time.time() - t_print) / 60.0
                print(
                    f"epoch={epoch+1:03d}/{epochs} step={global_step:07d} "
                    f"loss={loss_val:.6f} lr={lr:.2e} "
                    f"mask={mask_val:.3f} valid={int(num_valid_val)} valid_frac={valid_frac_val:.3f} "
                    f"time={dt:.1f}m"
                )
                t_print = time.time()

                csv_append_row(
                    csv_path,
                    CSV_HEADER,
                    {
                        "record_type": "step",
                        "timestamp_unix": int(time.time()),
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "steps_per_epoch": steps_per_epoch,
                        "total_steps": total_steps,
                        "warmup_steps": warmup_steps,
                        "lr": last_lr,
                        "loss": last_loss,
                        "mask_ratio_actual": mask_val,
                        "num_valid": int(num_valid_val),
                        "valid_frac": valid_frac_val,
                        "grad_clip": grad_clip,
                        "use_amp": bool(use_amp),
                        "amp_dtype": str(CFG["amp_dtype"]),
                        "split_train_n": len(train_ids),
                        "split_val_n": len(val_ids),
                    },
                )

        # ---- epoch means (train) ----
        denom = max(1, epoch_batches)
        epoch_loss_mean = epoch_loss_sum / denom
        epoch_mask_mean = epoch_mask_sum / denom
        epoch_valid_frac_mean = epoch_valid_frac_sum / denom
        epoch_num_valid_mean = epoch_num_valid_sum / denom

        # ---- val recon loss ----
        val_loss_mean = run_val_epoch(
            model=model,
            dl_val=dl_val,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            use_down=use_down,
            patch_size_zyx=tuple(CFG["patch_size_zyx"]),
            downsample_zyx=tuple(CFG["downsample_zyx"]),
        )
        print(f"[VAL] epoch={epoch+1:03d} val_recon_loss={val_loss_mean:.6f}")

        # ---- epoch end checkpoints ----
        ckpt_last_path = os.path.join(run_dir, "ckpt_last.pt")
        enc_last_path = os.path.join(run_dir, "enc_b_only.pt")

        save_ckpt(ckpt_last_path, model, opt, epoch + 1, global_step, CFG)
        save_enc_b_only(enc_last_path, model.enc_b)

        # ---- epoch CSV row ----
        csv_append_row(
            csv_path,
            CSV_HEADER,
            {
                "record_type": "epoch",
                "timestamp_unix": int(time.time()),
                "epoch": epoch + 1,
                "global_step": global_step,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
                "lr": last_lr,
                "loss": last_loss,  # last batch loss
                "mask_ratio_actual": last_dbg["mask_ratio_actual"],
                "num_valid": int(last_dbg["num_valid"]),
                "valid_frac": last_dbg["valid_frac"],
                "epoch_loss_mean": epoch_loss_mean,
                "val_loss_mean": val_loss_mean,
                "epoch_mask_mean": epoch_mask_mean,
                "epoch_valid_frac_mean": epoch_valid_frac_mean,
                "epoch_num_valid_mean": epoch_num_valid_mean,
                "ckpt_path": ckpt_last_path,
                "enc_b_path": enc_last_path,
                "grad_clip": grad_clip,
                "use_amp": bool(use_amp),
                "amp_dtype": str(CFG["amp_dtype"]),
                "split_train_n": len(train_ids),
                "split_val_n": len(val_ids),
            },
        )

        # ---- milestone checkpoints every 10 epochs ----
        if (epoch + 1) in save_epochs:
            ckpt_e_path = os.path.join(run_dir, f"ckpt_epoch_{epoch+1:03d}.pt")
            enc_e_path = os.path.join(run_dir, f"enc_b_only_epoch_{epoch+1:03d}.pt")

            save_ckpt(ckpt_e_path, model, opt, epoch + 1, global_step, CFG)
            save_enc_b_only(enc_e_path, model.enc_b)

            csv_append_row(
                csv_path,
                CSV_HEADER,
                {
                    "record_type": "epoch_ckpt",
                    "timestamp_unix": int(time.time()),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "steps_per_epoch": steps_per_epoch,
                    "total_steps": total_steps,
                    "warmup_steps": warmup_steps,
                    "epoch_loss_mean": epoch_loss_mean,
                    "val_loss_mean": val_loss_mean,
                    "epoch_mask_mean": epoch_mask_mean,
                    "epoch_valid_frac_mean": epoch_valid_frac_mean,
                    "epoch_num_valid_mean": epoch_num_valid_mean,
                    "ckpt_path": ckpt_e_path,
                    "enc_b_path": enc_e_path,
                    "split_train_n": len(train_ids),
                    "split_val_n": len(val_ids),
                },
            )

    print("[DONE] Training complete.")
    print(f"[SAVED] Encoder-only weights: {os.path.join(run_dir, 'enc_b_only.pt')}")
    print(f"[LOG] CSV: {csv_path}")


if __name__ == "__main__":
    main()
