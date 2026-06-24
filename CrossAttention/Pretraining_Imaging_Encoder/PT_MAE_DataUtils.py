import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nibabel as nib
from fuse.data.ops.op_base import OpBase, OpReversibleBase
from fuse.utils import NDict
import torch
import torch.nn as nn
import random
from typing import Tuple, Optional, Sequence, List, Dict, Union
from scipy.ndimage import rotate
from einops import rearrange



class OpIDPath(OpReversibleBase):
    """
    Derive CT and segmentation file paths from sample_id.

    Assumes:
        CT:    {sample_id}.nii.gz
        Body:  {sample_id}_body.nii.gz
        Liver: {sample_id}_liver.nii.gz
    """

    def __call__(self, sample_dict: NDict, op_id: Optional[str] = None) -> NDict:
        sid = sample_dict["data.sample_id"]

        # CT
        sample_dict["data.input.img_path"] = f"{sid}.nii.gz"

        # Segmentations
        sample_dict["data.input.body_path"] = f"{sid}_body.nii.gz"
        sample_dict["data.input.liver_path"] = f"{sid}_liver.nii.gz"

        return sample_dict


class OpLoadImage(OpBase):
    """
    Load a NIfTI image file (.nii.gz) from a given directory and convert it to [Z, Y, X] format.

    Stores:
      - array at key_out
      - original spacing at f"{key_out}.original_spacing"  (in [Z,Y,X])
      - original size    at f"{key_out}.original_size"     (in [Z,Y,X])

    Usage in pipeline:
      (OpGISTLoadImage(dir_path=CT_DIR, is_mask=False), dict(key_in="data.input.img_path", key_out="data.input.img"))
      (OpGISTLoadImage(dir_path=SEG_DIR, is_mask=True),  dict(key_in="data.input.body_seg_path", key_out="data.input.body"))
      (OpGISTLoadImage(dir_path=SEG_DIR, is_mask=True),  dict(key_in="data.input.liver_seg_path", key_out="data.input.liver"))
    """

    def __init__(self, dir_path: str, is_mask: bool = False):
        super().__init__()
        self._dir_path = dir_path
        self._is_mask = is_mask

    def __call__(self, sample_dict: NDict, key_in: str, key_out: str) -> NDict:
        rel_path = sample_dict[key_in]
        full_path = os.path.join(self._dir_path, rel_path)

        if not full_path.endswith(".nii.gz"):
            raise ValueError(f"Expected a .nii.gz file, but got: {full_path}")
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")

        img = nib.load(full_path)

        # [X, Y, Z] as stored in NIfTI
        arr_xyz = np.asanyarray(img.dataobj)

        # Convert to float32 for CT, uint8 for masks
        if self._is_mask:
            arr_xyz = (arr_xyz > 0).astype(np.uint8)
        else:
            arr_xyz = arr_xyz.astype(np.float32, copy=False)

        # -> [Z, Y, X]
        arr_zyx = np.transpose(arr_xyz, (2, 1, 0))

        # spacing: nib gives (X, Y, Z) -> convert to (Z, Y, X)
        spacing_xyz = img.header.get_zooms()[:3]
        spacing_zyx = np.asarray(spacing_xyz[::-1], dtype=np.float32)

        size_zyx = np.asarray(arr_zyx.shape, dtype=np.int32)

        sample_dict[key_out] = arr_zyx
        sample_dict[f"{key_out}.original_spacing"] = spacing_zyx
        sample_dict[f"{key_out}.original_size"] = size_zyx

        return sample_dict


class OpResample(OpBase):
    """
    Resample CT + body mask + liver mask to target spacing (Z, Y, X).
    Assumes all arrays are in [Z, Y, X] format.
    """

    def __init__(self, target_spacing_z, target_spacing_y, target_spacing_x):
        super().__init__()
        self._target_spacing_zyx = np.array(
            (target_spacing_z, target_spacing_y, target_spacing_x), dtype=np.float32
        )

    def __call__(self, sample_dict: NDict) -> NDict:
        # --- Resample CT (linear) ---
        img_rs, img_spacing_rs, img_size_rs = self._resample_from_keys(
            sample_dict=sample_dict,
            key_vol="data.input.img",
            key_spacing="data.input.img.original_spacing",
            key_size="data.input.img.original_size",
            interpolator=sitk.sitkLinear,
        )
        sample_dict["data.input.img.resampled"] = img_rs

        # --- Resample body mask (nearest) ---
        body_rs, _, _ = self._resample_from_keys(
            sample_dict=sample_dict,
            key_vol="data.input.body",
            key_spacing="data.input.body.original_spacing",
            key_size="data.input.body.original_size",
            interpolator=sitk.sitkNearestNeighbor,
        )
        sample_dict["data.input.body.resampled"] = body_rs.astype(np.uint8, copy=False)

        # --- Resample liver mask (nearest) ---
        liver_rs, _, _ = self._resample_from_keys(
            sample_dict=sample_dict,
            key_vol="data.input.liver",
            key_spacing="data.input.liver.original_spacing",
            key_size="data.input.liver.original_size",
            interpolator=sitk.sitkNearestNeighbor,
        )
        sample_dict["data.input.liver.resampled"] = liver_rs.astype(np.uint8, copy=False)

        # (Optional) store resampled spacing/size once (same target for all)
        sample_dict["data.input.resampled_spacing"] = self._target_spacing_zyx.copy()

        return sample_dict

    def _resample_from_keys(
        self,
        sample_dict: NDict,
        key_vol: str,
        key_spacing: str,
        key_size: str,
        interpolator,
    ):
        if key_vol not in sample_dict:
            raise KeyError(f"Missing volume key: {key_vol}")
        if key_spacing not in sample_dict or key_size not in sample_dict:
            raise KeyError(f"Missing metadata keys: {key_spacing} / {key_size}")

        vol_zyx = sample_dict[key_vol]  # [Z,Y,X]
        original_spacing_zyx = np.asarray(sample_dict[key_spacing], dtype=np.float32)
        original_size_zyx = np.asarray(sample_dict[key_size], dtype=np.int32)

        target_spacing_zyx = self._target_spacing_zyx

        # SimpleITK expects spacing/size in [X,Y,Z]
        vol_sitk = sitk.GetImageFromArray(vol_zyx)
        vol_sitk.SetSpacing(original_spacing_zyx[::-1].tolist())

        new_size_zyx = np.round(original_size_zyx * (original_spacing_zyx / target_spacing_zyx)).astype(int)

        vol_rs_sitk = self._resample_image(
            image=vol_sitk,
            target_spacing_xyz=target_spacing_zyx[::-1],
            new_size_xyz=new_size_zyx[::-1],
            interpolator=interpolator,
        )

        vol_rs_zyx = sitk.GetArrayFromImage(vol_rs_sitk).copy()
        spacing_rs_zyx = target_spacing_zyx.copy()
        size_rs_zyx = np.asarray(vol_rs_zyx.shape, dtype=np.int32)

        return vol_rs_zyx, spacing_rs_zyx, size_rs_zyx

    @staticmethod
    def _resample_image(image, target_spacing_xyz, new_size_xyz, interpolator):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetOutputSpacing([float(s) for s in target_spacing_xyz])
        resampler.SetSize([int(s) for s in new_size_xyz])
        resampler.SetInterpolator(interpolator)
        return resampler.Execute(image)


class OpClipHU(OpBase):
    """
    Clip CT values to a fixed HU range (no normalization).
    """

    def __init__(self, clip_range: Tuple[float, float] = (-150, 250)):
        super().__init__()
        self.clip_range = clip_range

    def __call__(self, sample_dict: NDict, key_in: str, key_out: str) -> NDict:
        ct = sample_dict[key_in].astype(np.float32, copy=False)  # [Z,Y,X]
        sample_dict[key_out] = np.clip(ct, self.clip_range[0], self.clip_range[1]).astype(np.float32, copy=False)
        return sample_dict


class OpAbdominalRandomCrop(OpBase):
    """
    Random abdominal crop using Body + Liver masks.

    Robust to volumes smaller than crop_size_zyx by padding up to the minimum crop size
    (CT padded with -150.0, masks with 0). This prevents crashes on short-Z scans.

    Assumption:
      - Larger Z index = more superior (toward head/shoulders).
    Preferred constraint (when feasible):
      - Crop superior end (z1-1) <= liver superior slice (z_liver_max).
    Fallback:
      - If liver constraint is impossible, sample from lower part of body (fallback_body_lower_frac),
        and if still impossible sample anywhere valid.

    Uses global randomness (non-deterministic).
    """

    def __init__(
        self,
        crop_size_zyx: Tuple[int, int, int] = (64, 256, 256),
        max_tries: int = 20,
        fallback_body_lower_frac: float = 0.70,
        store_debug: bool = False,
        pad_ct_val: float = -150.0,   # <-- NEW
    ):
        super().__init__()
        self.crop_size_zyx = tuple(int(v) for v in crop_size_zyx)
        self.max_tries = int(max_tries)
        self.fallback_body_lower_frac = float(fallback_body_lower_frac)
        self.store_debug = bool(store_debug)
        self.pad_ct_val = float(pad_ct_val)

        if not (0.0 <= self.fallback_body_lower_frac <= 1.0):
            raise ValueError("fallback_body_lower_frac must be in [0,1]")

    def __call__(
        self,
        sample_dict: NDict,
        key_in: Tuple[str, str, str],
        key_out: Tuple[str, str, str],
    ) -> NDict:

        key_img, key_body, key_liver = key_in
        key_out_img, key_out_body, key_out_liver = key_out

        ct = sample_dict[key_img]
        body = sample_dict[key_body]
        liver = sample_dict[key_liver]

        if ct.shape != body.shape or ct.shape != liver.shape:
            raise ValueError(f"Shape mismatch: ct={ct.shape}, body={body.shape}, liver={liver.shape}")

        cz, cy, cx = self.crop_size_zyx

        # ---- NEW: pad to at least crop size (so we never crash on small volumes) ----
        ct, body, liver, pad_zyx = self._pad_to_min_crop(ct, body, liver, self.crop_size_zyx)

        # write back padded arrays (so downstream ops see consistent shapes)
        sample_dict[key_img] = ct
        sample_dict[key_body] = body
        sample_dict[key_liver] = liver

        Z, Y, X = ct.shape
        hz, hy, hx = cz // 2, cy // 2, cx // 2

        # Now crop size is guaranteed to fit
        # (still good to keep assert-like safety)
        if cz > Z or cy > Y or cx > X:
            raise ValueError(f"Crop size {self.crop_size_zyx} larger than (padded) volume {ct.shape}")

        # --- Z landmarks ---
        z_body_min, z_body_max = self._first_last_nonzero_z(body)
        z_liver_min, z_liver_max = self._first_last_nonzero_z(liver)

        if z_body_min is None or z_body_max is None:
            # If padding created a fully-empty mask (shouldn't happen), just allow anywhere
            z_body_min, z_body_max = 0, Z - 1

        # Preferred: constrain crop to not extend above liver superior boundary
        used_fallback = False
        if z_liver_min is not None and z_liver_max is not None:
            z_upper_bound = int(z_liver_max)
        else:
            used_fallback = True
            z_upper_bound = int(np.floor(
                z_body_min + self.fallback_body_lower_frac * (z_body_max - z_body_min)
            ))

        # Primary constraint:
        # z1-1 <= z_upper_bound  ->  z0 <= z_upper_bound - (cz - 1)
        z0_min = max(0, z_body_min)
        z0_max = min(Z - cz, z_upper_bound - (cz - 1))

        # If impossible, fall back to a lower-body window
        if z0_min > z0_max:
            used_fallback = True
            lower_body_upper = int(np.floor(
                z_body_min + self.fallback_body_lower_frac * (z_body_max - z_body_min)
            ))
            z0_min = max(0, z_body_min)
            z0_max = min(Z - cz, lower_body_upper)

            # Still impossible? last resort: anywhere valid
            if z0_min > z0_max:
                z0_min = 0
                z0_max = Z - cz

        # ---- sample attempts ----
        for _ in range(self.max_tries):

            z0 = int(np.random.randint(z0_min, z0_max + 1))
            z1 = z0 + cz
            zc = z0 + hz  # mid-slice for choosing y/x inside body

            coords = np.argwhere(body[zc] > 0)
            if coords.size == 0:
                # if body slice is empty (rare), try again
                continue

            ys = coords[:, 0]
            xs = coords[:, 1]
            keep = (ys >= hy) & (ys < (Y - hy)) & (xs >= hx) & (xs < (X - hx))
            coords = coords[keep]
            if coords.size == 0:
                continue

            yi, xi = coords[np.random.randint(0, coords.shape[0])]
            yc, xc = int(yi), int(xi)

            y0, y1 = yc - hy, yc + hy
            x0, x1 = xc - hx, xc + hx

            ct_crop = ct[z0:z1, y0:y1, x0:x1]
            if ct_crop.shape != (cz, cy, cx):
                continue

            sample_dict[key_out_img] = ct_crop.astype(np.float32, copy=False)
            sample_dict[key_out_body] = body[z0:z1, y0:y1, x0:x1].astype(np.uint8, copy=False)
            sample_dict[key_out_liver] = liver[z0:z1, y0:y1, x0:x1].astype(np.uint8, copy=False)

            if self.store_debug:
                sample_dict["data.crop.used_fallback_body"] = np.bool_(used_fallback)
                sample_dict["data.crop.center_zyx"] = np.array([zc, yc, xc], dtype=np.int32)
                sample_dict["data.crop.z0_range"] = np.array([z0_min, z0_max], dtype=np.int32)
                sample_dict["data.crop.pad_zyx"] = np.array(pad_zyx, dtype=np.int32)

            return sample_dict

        raise RuntimeError("Failed to sample valid abdominal crop.")

    def _pad_to_min_crop(self, ct, body, liver, crop_size_zyx):
        """Pad arrays at the END of each axis so shape >= crop size."""
        Z, Y, X = ct.shape
        cz, cy, cx = crop_size_zyx

        pad_z = max(0, cz - Z)
        pad_y = max(0, cy - Y)
        pad_x = max(0, cx - X)

        if pad_z == 0 and pad_y == 0 and pad_x == 0:
            return ct, body, liver, (0, 0, 0)

        pad_width = ((0, pad_z), (0, pad_y), (0, pad_x))

        ct_p = np.pad(ct, pad_width, mode="constant", constant_values=self.pad_ct_val).astype(np.float32, copy=False)
        body_p = np.pad(body, pad_width, mode="constant", constant_values=0).astype(np.uint8, copy=False)
        liver_p = np.pad(liver, pad_width, mode="constant", constant_values=0).astype(np.uint8, copy=False)

        return ct_p, body_p, liver_p, (pad_z, pad_y, pad_x)

    @staticmethod
    def _first_last_nonzero_z(mask_zyx) -> Tuple[Optional[int], Optional[int]]:
        idx = np.where(mask_zyx > 0)
        if idx[0].size == 0:
            return None, None
        return int(idx[0].min()), int(idx[0].max())


class OpMinMaxNormalize(OpBase):
    """
    Fixed-range min-max normalization to [0, 1].

    Intended usage (recommended for CT):
      - Apply AFTER OpClipHU so values are in [min_val, max_val]
      - Then normalize: (x - min_val) / (max_val - min_val)

    Notes:
      - Keeps float32
      - Optionally clips again for safety (in case something upstream didn't clip)
    """

    def __init__(
        self,
        min_val: float = -150.0,
        max_val: float = 250.0,
        eps: float = 1e-8,
        clip_before_norm: bool = True,
    ):
        super().__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.eps = float(eps)
        self.clip_before_norm = bool(clip_before_norm)

        if not (self.max_val > self.min_val):
            raise ValueError(f"max_val must be > min_val, got {self.min_val}, {self.max_val}")

    def __call__(self, sample_dict: NDict, key_in: str, key_out: str) -> NDict:
        x = sample_dict[key_in].astype(np.float32, copy=False)

        if self.clip_before_norm:
            x = np.clip(x, self.min_val, self.max_val)

        denom = (self.max_val - self.min_val) + self.eps
        x = (x - self.min_val) / denom

        # Safety (numerical / upstream issues)
        x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)

        sample_dict[key_out] = x
        return sample_dict


class OpCTPatchifyWithMask(OpBase):
    """
    Patchify a 3D abdominal CT crop [Z,Y,X] into non-overlapping patch tokens and create a VALID-TOKEN mask.

    This Op produces a *validity mask* (embed_mask) used by the encoder to ignore invalid tokens
    (padding / outside-body patches). It does NOT implement MAE random masking (e.g., 75%),
    which should happen later in the MAE module or in a dedicated Op.

    Inputs (typical for our pretraining pipeline):
      - CT crop:   data.input.img.crop          float32 [Z,Y,X]
      - Body crop: data.input.body.crop         uint8   [Z,Y,X]  (recommended)
      - Liver crop (optional): data.input.liver.crop  uint8 [Z,Y,X]

    Outputs:
      - key_out:               torch.FloatTensor [N, patch_dim]
      - model.embed_mask_b:    torch.BoolTensor  [N]  (True = valid token)
      - optionally:            grid shape and body fraction per patch

    Valid token rules (configurable):
      1) If pad_val is provided and padding exists (or pad voxels exist), we can invalidate patches
         that contain too much padding via mask_pad_threshold.
      2) If body mask exists, we can invalidate patches with body_fraction < min_body_frac.
      3) If liver mask exists and you want it, you can optionally require liver_fraction >= min_liver_frac
         (off by default; for pretraining I usually do NOT require liver, because abdomen includes more).

    Note:
      - For our current crop design (e.g., (64,256,256)) and chosen patch sizes, shapes are typically divisible,
        so padding is usually not needed. But this Op can optionally pad.
    """

    def __init__(
        self,
        patch_size_zyx: Tuple[int, int, int],
        # --- validity masking knobs ---
        min_body_frac: float = 0.30,            # keep patch if >= this fraction is inside body
        use_liver_frac: bool = False,           # usually False for pretraining
        min_liver_frac: float = 0.01,           # only used if use_liver_frac=True
        # --- padding support ---
        pad_to_multiple: bool = False,          # if True, pad volume to multiples of patch size
        pad_val: Optional[float] = None,        # e.g., -150.0 if your crop uses -150 padding; or 0.0 if normalized
        mask_pad_threshold: float = 1.0,        # 1.0 disables pad-based masking; 0.3 => mask if >=30% padding
        # --- misc ---
        store_debug: bool = False,
    ):
        super().__init__()
        self.patch_size_zyx = tuple(int(x) for x in patch_size_zyx)

        self.min_body_frac = float(min_body_frac)
        self.use_liver_frac = bool(use_liver_frac)
        self.min_liver_frac = float(min_liver_frac)

        self.pad_to_multiple = bool(pad_to_multiple)
        self.pad_val = pad_val
        self.mask_pad_threshold = float(mask_pad_threshold)

        self.store_debug = bool(store_debug)

        if any(p <= 0 for p in self.patch_size_zyx):
            raise ValueError(f"patch_size_zyx must be positive, got {self.patch_size_zyx}")
        if not (0.0 <= self.min_body_frac <= 1.0):
            raise ValueError("min_body_frac must be in [0,1]")
        if not (0.0 <= self.mask_pad_threshold <= 1.0):
            raise ValueError("mask_pad_threshold must be in [0,1]")
        if self.pad_to_multiple and self.pad_val is None:
            # You can still pad with 0.0 if normalized; but require explicitness to avoid silent bugs.
            raise ValueError("pad_to_multiple=True requires pad_val to be set explicitly.")

    def __call__(
        self,
        sample_dict: "NDict",
        key_in: Tuple[str, str, str],
        key_out: Tuple[str, str, str],
    ) -> "NDict":
        key_in_img, key_in_body, key_in_liver = key_in
        key_out_patches, key_out_mask, key_out_grid = key_out

        vol = sample_dict[key_in_img]
        if not isinstance(vol, np.ndarray):
            vol = np.asarray(vol)
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume [Z,Y,X], got shape {vol.shape}")

        body = None
        if key_in_body and (key_in_body in sample_dict):
            body = sample_dict[key_in_body]
            if not isinstance(body, np.ndarray):
                body = np.asarray(body)
            if body.shape != vol.shape:
                raise ValueError(f"Body shape {body.shape} != volume shape {vol.shape}")
            body = (body > 0).astype(np.uint8, copy=False)

        liver = None
        if self.use_liver_frac and key_in_liver and (key_in_liver in sample_dict):
            liver = sample_dict[key_in_liver]
            if not isinstance(liver, np.ndarray):
                liver = np.asarray(liver)
            if liver.shape != vol.shape:
                raise ValueError(f"Liver shape {liver.shape} != volume shape {vol.shape}")
            liver = (liver > 0).astype(np.uint8, copy=False)

        # Optionally pad to multiples of patch size
        vol_p, body_p, liver_p, pad_info = self._pad_if_needed(vol, body, liver)

        z, y, x = vol_p.shape
        pz, py, px = self.patch_size_zyx

        if (z % pz) or (y % py) or (x % px):
            raise ValueError(
                f"Volume shape {vol_p.shape} not divisible by patch size {self.patch_size_zyx}. "
                f"Either fix upstream crop sizing or set pad_to_multiple=True."
            )

        gz, gy, gx = z // pz, y // py, x // px
        num_patches = gz * gy * gx

        # Patchify CT
        patches = rearrange(
            vol_p,
            "(gz pz) (gy py) (gx px) -> (gz gy gx) (pz py px)",
            gz=gz, gy=gy, gx=gx, pz=pz, py=py, px=px
        )
        patches_t = torch.as_tensor(patches, dtype=torch.float32)

        # Build valid token mask
        valid = torch.ones((num_patches,), dtype=torch.bool)

        # (A) Padding-based validity (optional, only if pad_val is provided and threshold < 1.0)
        if (self.pad_val is not None) and (self.mask_pad_threshold < 1.0):
            pad_val = float(self.pad_val)
            total_vox = patches_t.shape[1]
            num_pad = (patches_t == pad_val).sum(dim=1)
            pad_frac = num_pad.float() / float(total_vox)
            # invalidate if pad_frac >= threshold
            valid = valid & (pad_frac < self.mask_pad_threshold)

        # (B) Body-fraction validity (recommended)
        body_frac_t = None
        if body_p is not None:
            body_patches = rearrange(
                body_p,
                "(gz pz) (gy py) (gx px) -> (gz gy gx) (pz py px)",
                gz=gz, gy=gy, gx=gx, pz=pz, py=py, px=px
            )
            body_patches_t = torch.as_tensor(body_patches, dtype=torch.float32)
            body_frac_t = body_patches_t.mean(dim=1)  # fraction in [0,1]
            valid = valid & (body_frac_t >= self.min_body_frac)

        # (C) Optional liver-fraction validity (usually OFF for pretraining)
        liver_frac_t = None
        if liver_p is not None:
            liver_patches = rearrange(
                liver_p,
                "(gz pz) (gy py) (gx px) -> (gz gy gx) (pz py px)",
                gz=gz, gy=gy, gx=gx, pz=pz, py=py, px=px
            )
            liver_patches_t = torch.as_tensor(liver_patches, dtype=torch.float32)
            liver_frac_t = liver_patches_t.mean(dim=1)
            valid = valid & (liver_frac_t >= self.min_liver_frac)

        # Store outputs
        sample_dict[key_out_patches] = patches_t
        sample_dict[key_out_mask] = valid

        if key_out_grid:
            sample_dict[key_out_grid] = np.asarray([gz, gy, gx], dtype=np.int32)

        if self.store_debug:
            sample_dict["data.patchify.patch_size_zyx"] = np.asarray([pz, py, px], dtype=np.int32)
            sample_dict["data.patchify.num_patches"] = int(num_patches)
            if pad_info is not None:
                sample_dict["data.patchify.pad_zyx"] = np.asarray(pad_info, dtype=np.int32)
            if body_frac_t is not None:
                sample_dict["data.patchify.body_frac"] = body_frac_t.cpu().numpy().astype(np.float32)
            if liver_frac_t is not None:
                sample_dict["data.patchify.liver_frac"] = liver_frac_t.cpu().numpy().astype(np.float32)

        return sample_dict

    def _pad_if_needed(self, vol, body, liver):
        if not self.pad_to_multiple:
            return vol, body, liver, None

        z, y, x = vol.shape
        pz, py, px = self.patch_size_zyx

        def _pad_amount(n, p):
            r = n % p
            return 0 if r == 0 else (p - r)

        pad_z = _pad_amount(z, pz)
        pad_y = _pad_amount(y, py)
        pad_x = _pad_amount(x, px)

        if pad_z == 0 and pad_y == 0 and pad_x == 0:
            return vol, body, liver, (0, 0, 0)

        # pad at the end of each axis (simple + consistent)
        pad_width = ((0, pad_z), (0, pad_y), (0, pad_x))
        vol_p = np.pad(vol, pad_width, mode="constant", constant_values=float(self.pad_val)).astype(np.float32, copy=False)

        body_p = None
        if body is not None:
            body_p = np.pad(body, pad_width, mode="constant", constant_values=0).astype(np.uint8, copy=False)

        liver_p = None
        if liver is not None:
            liver_p = np.pad(liver, pad_width, mode="constant", constant_values=0).astype(np.uint8, copy=False)

        return vol_p, body_p, liver_p, (pad_z, pad_y, pad_x)

