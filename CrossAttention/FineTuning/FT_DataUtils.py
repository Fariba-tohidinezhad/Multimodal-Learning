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
    Derive CT and tumor segmentation RELATIVE file paths from sample_id.

    Downstream folder structure:
        CT:  img/{sample_id}.nii.gz
        SEG: seg/{sample_id}.nii.gz

    IMPORTANT:
      - This Op stores only filenames (relative paths), because OpLoadImage
        will join them with its dir_path.
    """

    def __call__(self, sample_dict: NDict, op_id: Optional[str] = None) -> NDict:
        sid = sample_dict["data.sample_id"]

        # Store RELATIVE filenames (NOT absolute paths)
        sample_dict["data.input.img_path"] = f"{sid}.nii.gz"
        sample_dict["data.input.seg_path"] = f"{sid}.nii.gz"

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
    Resample CT + tumor segmentation to target spacing (Z, Y, X).
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
        sample_dict["data.input.img.resampled_spacing"] = img_spacing_rs
        sample_dict["data.input.img.resampled_size"] = img_size_rs

        # --- Resample tumor seg (nearest) ---
        seg_rs, seg_spacing_rs, seg_size_rs = self._resample_from_keys(
            sample_dict=sample_dict,
            key_vol="data.input.seg",
            key_spacing="data.input.seg.original_spacing",
            key_size="data.input.seg.original_size",
            interpolator=sitk.sitkNearestNeighbor,
        )
        sample_dict["data.input.seg.resampled"] = seg_rs.astype(np.uint8, copy=False)
        sample_dict["data.input.seg.resampled_spacing"] = seg_spacing_rs
        sample_dict["data.input.seg.resampled_size"] = seg_size_rs

        # (Optional) store once (same target for all)
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

        new_size_zyx = np.round(
            original_size_zyx * (original_spacing_zyx / target_spacing_zyx)
        ).astype(int)

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


class OpClipAndMinMaxNormalize(OpBase):
    """
    Clip CT values to a fixed HU range and min-max normalize to [0, 1] in a single Op.

    Typical CT usage:
      - Clip to [min_val, max_val] in HU (e.g., [-150, 250])
      - Normalize: (x - min_val) / (max_val - min_val)

    Notes:
      - Outputs float32
      - Includes optional safety clipping to [0, 1]
    """

    def __init__(
        self,
        clip_range: Tuple[float, float] = (-150.0, 250.0),
        eps: float = 1e-8,
        safety_clip_01: bool = True,
    ):
        super().__init__()
        self.min_val = float(clip_range[0])
        self.max_val = float(clip_range[1])
        self.eps = float(eps)
        self.safety_clip_01 = bool(safety_clip_01)

        if not (self.max_val > self.min_val):
            raise ValueError(f"clip_range invalid: {clip_range}. Need max_val > min_val.")

    def __call__(self, sample_dict: NDict, key_in: str, key_out: str) -> NDict:
        x = sample_dict[key_in].astype(np.float32, copy=False)  # [Z,Y,X]

        # 1) Clip HU
        x = np.clip(x, self.min_val, self.max_val)

        # 2) Min-max normalize to [0, 1]
        denom = (self.max_val - self.min_val) + self.eps
        x = (x - self.min_val) / denom

        # Optional safety
        if self.safety_clip_01:
            x = np.clip(x, 0.0, 1.0)

        sample_dict[key_out] = x.astype(np.float32, copy=False)
        return sample_dict


class OpCropOrPadToFixedShapeUsingSeg(OpBase):
    """
    Crop/pad to a fixed target shape using the tumor segmentation to define the crop window,
    AND set everything outside the tumor to padding (pad_val).

    - Uses tumor seg to find a crop center (bbox center by default).
    - Crops/pads BOTH image and seg to the same target_shape_zyx.
    - After fitting, applies tumor mask: img_out[seg_out == 0] = pad_val
      so "everything outside the tumor becomes padding".

    Assumes volumes are in [Z, Y, X].

    Inputs:
      key_in:  (img_key, seg_key)
    Outputs:
      key_out: (img_fitted_key, seg_fitted_key)

    Notes:
      - If seg is empty, falls back to a centered crop/pad and does NOT erase everything
        (because "outside tumor" is undefined). You can change that behavior if you prefer.
      - pad_val should match the value domain:
          * HU-clipped CT: pad_val = -150.0 (recommended)
          * normalized [0,1]: pad_val = 0.0
          * if you explicitly want -1 padding: pad_val = -1.0 (but be consistent downstream)
    """

    def __init__(
        self,
        target_shape_zyx: Tuple[int, int, int],
        pad_val: float = 0.0,
        center_mode: str = "bbox",   # "bbox" or "com"
        store_debug: bool = False,
    ):
        super().__init__()
        self.target_shape = tuple(int(v) for v in target_shape_zyx)
        self.pad_val = float(pad_val)
        self.center_mode = str(center_mode)
        self.store_debug = bool(store_debug)

        if self.center_mode not in ("bbox", "com"):
            raise ValueError("center_mode must be 'bbox' or 'com'")

    def __call__(
        self,
        sample_dict: NDict,
        key_in: Tuple[str, str],
        key_out: Tuple[str, str],
    ) -> NDict:
        key_in_img, key_in_seg = key_in
        key_out_img, key_out_seg = key_out

        img = sample_dict[key_in_img]
        seg = sample_dict[key_in_seg]

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
        if not isinstance(seg, np.ndarray):
            seg = np.asarray(seg)

        if img.ndim != 3 or seg.ndim != 3:
            raise ValueError(f"Expected 3D [Z,Y,X]. Got img={img.shape}, seg={seg.shape}")
        if img.shape != seg.shape:
            raise ValueError(f"Shape mismatch: img={img.shape}, seg={seg.shape}")

        seg_bin = (seg > 0).astype(np.uint8, copy=False)
        has_tumor = bool(seg_bin.any())

        # --- choose crop center from seg (or fallback to volume center if empty) ---
        if has_tumor:
            center = self._compute_center(seg_bin)
        else:
            center = np.array(img.shape, dtype=int) // 2

        # --- crop/pad both to fixed target ---
        img_fit, seg_fit = self._crop_or_pad_centered(
            img, seg_bin, center_zyx=center, target_shape_zyx=self.target_shape, pad_val=self.pad_val
        )

        # --- crucial: set everything outside tumor to padding ---
        # Only do this if tumor exists (otherwise you'd wipe the whole image).
        if has_tumor:
            img_fit = img_fit.astype(np.float32, copy=False)
            img_fit[seg_fit == 0] = self.pad_val

        sample_dict[key_out_img] = img_fit.astype(np.float32, copy=False)
        sample_dict[key_out_seg] = seg_fit.astype(np.uint8, copy=False)

        if self.store_debug:
            sample_dict["data.fitted.target_shape_zyx"] = np.asarray(self.target_shape, dtype=np.int32)
            sample_dict["data.fitted.pad_val"] = np.float32(self.pad_val)
            sample_dict["data.fitted.has_tumor"] = np.bool_(has_tumor)
            sample_dict["data.fitted.center_zyx"] = np.asarray(center, dtype=np.int32)
            # tumor bbox in fitted coords (if exists)
            if has_tumor:
                zz, yy, xx = np.where(seg_fit > 0)
                sample_dict["data.fitted.tumor_bbox_zyx_min"] = np.asarray([zz.min(), yy.min(), xx.min()], dtype=np.int32)
                sample_dict["data.fitted.tumor_bbox_zyx_max"] = np.asarray([zz.max(), yy.max(), xx.max()], dtype=np.int32)

        return sample_dict

    def _compute_center(self, seg_bin: np.ndarray) -> np.ndarray:
        if self.center_mode == "bbox":
            zz, yy, xx = np.where(seg_bin > 0)
            z0, z1 = int(zz.min()), int(zz.max())
            y0, y1 = int(yy.min()), int(yy.max())
            x0, x1 = int(xx.min()), int(xx.max())
            return np.array([(z0 + z1) // 2, (y0 + y1) // 2, (x0 + x1) // 2], dtype=int)

        # center of mass (rounded)
        coords = np.argwhere(seg_bin > 0).astype(np.float32)
        com = coords.mean(axis=0)  # [z,y,x]
        return np.round(com).astype(int)

    @staticmethod
    def _crop_or_pad_centered(
        img: np.ndarray,
        seg: np.ndarray,
        center_zyx: np.ndarray,
        target_shape_zyx: Tuple[int, int, int],
        pad_val: float,
    ):
        Z, Y, X = img.shape
        tz, ty, tx = target_shape_zyx
        cz, cy, cx = (int(center_zyx[0]), int(center_zyx[1]), int(center_zyx[2]))

        # desired window [start, end)
        z0 = cz - tz // 2
        y0 = cy - ty // 2
        x0 = cx - tx // 2
        z1 = z0 + tz
        y1 = y0 + ty
        x1 = x0 + tx

        # padding needed if out of bounds
        pbz = max(0, -z0); paz = max(0, z1 - Z)
        pby = max(0, -y0); pay = max(0, y1 - Y)
        pbx = max(0, -x0); pax = max(0, x1 - X)

        # clamp crop window
        z0c = max(0, z0); z1c = min(Z, z1)
        y0c = max(0, y0); y1c = min(Y, y1)
        x0c = max(0, x0); x1c = min(X, x1)

        img_crop = img[z0c:z1c, y0c:y1c, x0c:x1c]
        seg_crop = seg[z0c:z1c, y0c:y1c, x0c:x1c]

        pad_width = ((pbz, paz), (pby, pay), (pbx, pax))

        img_out = np.pad(
            img_crop.astype(np.float32, copy=False),
            pad_width,
            mode="constant",
            constant_values=float(pad_val),
        )
        seg_out = np.pad(
            seg_crop.astype(np.uint8, copy=False),
            pad_width,
            mode="constant",
            constant_values=0,
        )

        if img_out.shape != target_shape_zyx or seg_out.shape != target_shape_zyx:
            raise RuntimeError(
                f"Output shape mismatch: img_out={img_out.shape}, seg_out={seg_out.shape}, expected={target_shape_zyx}"
            )

        return img_out, seg_out


class OpTumorRandomRotation(OpBase):
    """
    Random in-plane rotation of tumor-only fitted volumes (shape preserved).

    Inputs are assumed to already be fitted to target shape and tumor-only:
      - img has pad_val outside tumor
      - seg is binary tumor mask

    This op:
      - samples random angle each call (with probability `prob`)
      - rotates img+seg together with reshape=False to keep shape fixed
      - pads missing regions with pad_val (img) / 0 (seg)
      - re-applies tumor-only rule: img[seg==0] = pad_val
    """

    def __init__(
        self,
        angle_range_deg: Tuple[float, float] = (-10.0, 10.0),
        prob: float = 0.5,
        pad_val: float = 0.0,  # 0.0 if normalized, -150.0 if HU-clipped
        axes_options: Optional[Sequence[Tuple[int, int]]] = None,
        store_debug: bool = False,
    ):
        super().__init__()
        self.angle_range_deg = (float(angle_range_deg[0]), float(angle_range_deg[1]))
        self.prob = float(prob)
        self.pad_val = float(pad_val)
        # default: axial-only rotation (Y,X) to avoid anisotropic Z artifacts
        self.axes_options = list(axes_options) if axes_options is not None else [(1, 2)]
        self.store_debug = bool(store_debug)

        if not (0.0 <= self.prob <= 1.0):
            raise ValueError("prob must be in [0,1]")
        for ax in self.axes_options:
            if ax not in ((0, 1), (0, 2), (1, 2)):
                raise ValueError(f"Invalid axes {ax} for [Z,Y,X]")

    def __call__(
        self,
        sample_dict: NDict,
        key_in: Tuple[str, str],
        key_out: Tuple[str, str],
    ) -> NDict:
        key_img_in, key_seg_in = key_in
        key_img_out, key_seg_out = key_out

        # copy=True to avoid mutating cached/static arrays by accident
        img = np.asarray(sample_dict[key_img_in], dtype=np.float32).copy()
        seg = (np.asarray(sample_dict[key_seg_in]) > 0).astype(np.uint8, copy=True)

        if img.shape != seg.shape:
            raise ValueError(f"Shape mismatch: img={img.shape}, seg={seg.shape}")
        if img.ndim != 3:
            raise ValueError(f"Expected 3D [Z,Y,X], got {img.shape}")

        # apply augmentation with probability prob
        if np.random.rand() >= self.prob:
            sample_dict[key_img_out] = img
            sample_dict[key_seg_out] = seg
            if self.store_debug:
                sample_dict["data.aug.rotate.applied"] = np.bool_(False)
            return sample_dict

        # sample random angle and axes EACH CALL
        angle = float(np.random.uniform(self.angle_range_deg[0], self.angle_range_deg[1]))
        axes = self.axes_options[np.random.randint(0, len(self.axes_options))]

        # rotate - keep shape fixed
        img_r = rotate(
            img,
            angle=angle,
            axes=axes,
            reshape=False,
            order=1,                 # linear for image
            mode="constant",
            cval=self.pad_val,
            prefilter=False,
        ).astype(np.float32, copy=False)

        seg_r = rotate(
            seg,
            angle=angle,
            axes=axes,
            reshape=False,
            order=0,                 # nearest for mask
            mode="constant",
            cval=0,
            prefilter=False,
        )
        seg_r = (seg_r > 0).astype(np.uint8, copy=False)

        # re-enforce tumor-only padding (important after interpolation)
        img_r[seg_r == 0] = self.pad_val

        sample_dict[key_img_out] = img_r
        sample_dict[key_seg_out] = seg_r

        if self.store_debug:
            sample_dict["data.aug.rotate.applied"] = np.bool_(True)
            sample_dict["data.aug.rotate.angle_deg"] = np.float32(angle)
            sample_dict["data.aug.rotate.axes"] = np.asarray(axes, dtype=np.int32)

        return sample_dict


class OpTumorPatchifyWithMask(OpBase):
    """
    Patchify a 3D tumor-focused CT volume [Z,Y,X] into non-overlapping patch tokens
    and create a VALID-TOKEN mask based on tumor segmentation fraction per patch.

    Inputs:
      - CT volume:  key_in_img (float32) [Z,Y,X]  (typically normalized to [0,1])
      - Tumor seg:  key_in_seg (uint8)   [Z,Y,X]  (binary)

    Outputs:
      - patches:          torch.FloatTensor [N, patch_dim]
      - model.embed_mask_b: torch.BoolTensor [N] (True = valid token)
      - grid:             np.int32 [gz,gy,gx]

    Valid token rule (recommended for downstream):
      - valid if tumor_fraction_in_patch >= min_tumor_frac

    Optional:
      - If pad_to_multiple=True, pads at END of each axis to multiples of patch size.
        CT padded with pad_val (default 0.0), seg padded with 0.
    """

    def __init__(
        self,
        patch_size_zyx: Tuple[int, int, int],
        min_tumor_frac: float = 0.001,      # e.g. >0 voxels (very permissive); tune if needed
        pad_to_multiple: bool = True,       # recommended because (80,347,498) is not divisible for many patch sizes
        pad_val: float = 0.0,               # aligned with your MAE pretraining normalized padding
        store_debug: bool = False,
    ):
        super().__init__()
        self.patch_size_zyx = tuple(int(x) for x in patch_size_zyx)
        self.min_tumor_frac = float(min_tumor_frac)
        self.pad_to_multiple = bool(pad_to_multiple)
        self.pad_val = float(pad_val)
        self.store_debug = bool(store_debug)

        if any(p <= 0 for p in self.patch_size_zyx):
            raise ValueError(f"patch_size_zyx must be positive, got {self.patch_size_zyx}")
        if not (0.0 <= self.min_tumor_frac <= 1.0):
            raise ValueError("min_tumor_frac must be in [0,1]")

    def __call__(
        self,
        sample_dict: "NDict",
        key_in: Tuple[str, str],
        key_out: Tuple[str, str, str],
    ) -> "NDict":
        key_in_img, key_in_seg = key_in
        key_out_patches, key_out_mask, key_out_grid = key_out

        vol = np.asarray(sample_dict[key_in_img], dtype=np.float32)
        seg = (np.asarray(sample_dict[key_in_seg]) > 0).astype(np.uint8, copy=False)

        if vol.ndim != 3 or seg.ndim != 3:
            raise ValueError(f"Expected 3D [Z,Y,X]. Got vol={vol.shape}, seg={seg.shape}")
        if vol.shape != seg.shape:
            raise ValueError(f"Shape mismatch: vol={vol.shape}, seg={seg.shape}")

        vol_p, seg_p, pad_info = self._pad_if_needed(vol, seg)

        z, y, x = vol_p.shape
        pz, py, px = self.patch_size_zyx

        if (z % pz) or (y % py) or (x % px):
            raise ValueError(
                f"Volume shape {vol_p.shape} not divisible by patch size {self.patch_size_zyx}. "
                f"Set pad_to_multiple=True or fix upstream sizing."
            )

        gz, gy, gx = z // pz, y // py, x // px
        num_patches = gz * gy * gx

        # Patchify CT -> tokens
        patches = rearrange(
            vol_p,
            "(gz pz) (gy py) (gx px) -> (gz gy gx) (pz py px)",
            gz=gz, gy=gy, gx=gx, pz=pz, py=py, px=px,
        )
        patches_t = torch.as_tensor(patches, dtype=torch.float32)

        # Patchify seg -> tumor fraction per patch
        seg_patches = rearrange(
            seg_p,
            "(gz pz) (gy py) (gx px) -> (gz gy gx) (pz py px)",
            gz=gz, gy=gy, gx=gx, pz=pz, py=py, px=px,
        )
        seg_patches_t = torch.as_tensor(seg_patches, dtype=torch.float32)
        tumor_frac_t = seg_patches_t.mean(dim=1)  # [N] in [0,1]

        valid = tumor_frac_t >= self.min_tumor_frac  # [N] bool

        # Store outputs
        sample_dict[key_out_patches] = patches_t
        sample_dict[key_out_mask] = valid
        sample_dict[key_out_grid] = np.asarray([gz, gy, gx], dtype=np.int32)

        if self.store_debug:
            sample_dict["data.patchify.patch_size_zyx"] = np.asarray([pz, py, px], dtype=np.int32)
            sample_dict["data.patchify.grid_zyx"] = np.asarray([gz, gy, gx], dtype=np.int32)
            sample_dict["data.patchify.num_patches"] = int(num_patches)
            sample_dict["data.patchify.min_tumor_frac"] = np.float32(self.min_tumor_frac)
            sample_dict["data.patchify.tumor_frac"] = tumor_frac_t.cpu().numpy().astype(np.float32)
            if pad_info is not None:
                sample_dict["data.patchify.pad_zyx"] = np.asarray(pad_info, dtype=np.int32)

        return sample_dict

    def _pad_if_needed(self, vol: np.ndarray, seg: np.ndarray):
        if not self.pad_to_multiple:
            return vol, seg, None

        z, y, x = vol.shape
        pz, py, px = self.patch_size_zyx

        def _pad_amount(n, p):
            r = n % p
            return 0 if r == 0 else (p - r)

        pad_z = _pad_amount(z, pz)
        pad_y = _pad_amount(y, py)
        pad_x = _pad_amount(x, px)

        if pad_z == 0 and pad_y == 0 and pad_x == 0:
            return vol, seg, (0, 0, 0)

        pad_width = ((0, pad_z), (0, pad_y), (0, pad_x))
        vol_p = np.pad(vol, pad_width, mode="constant", constant_values=self.pad_val).astype(np.float32, copy=False)
        seg_p = np.pad(seg, pad_width, mode="constant", constant_values=0).astype(np.uint8, copy=False)

        return vol_p, seg_p, (pad_z, pad_y, pad_x)



# Clinical Ops
class OpCastLabelToFloat(OpBase):
    """
    Casts an integer label (e.g., 0 or 1) to float (0.0 or 1.0).
    Required for binary classification with BCEWithLogitsLoss.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, sample_dict: NDict, key_in: str, key_out: Optional[str] = None) -> NDict:
        if key_out is None:
            key_out = key_in
        label = sample_dict[key_in]
        sample_dict[key_out] = float(label)
        return sample_dict


class OpClinicalPreprocess(OpBase):
    """
    Prepares clinical features for transformer input:
    - Binarizes continuous variables using thresholds
    - Converts categorical strings to integer indices
    - Assigns masked token index to missing values
    - Outputs: a flat vector of all features
    """

    def __init__(
        self,
        thresholds: Dict[str, float],
        categorical_mappings: Dict[str, Dict[str, int]]
    ):
        super().__init__()
        self.thresholds = thresholds
        self.cat_map = categorical_mappings

        # Ordered feature list for embedding extraction
        self.feature_names = list(thresholds.keys()) + list(categorical_mappings.keys())

    def __call__(self, sample_dict, key_in: str, key_out: str):
        raw = sample_dict[key_in]
        vector = []

        for feature in self.feature_names:
            if feature in self.thresholds:  # Binarize continuous
                raw_value = raw.get(feature, None)
                if pd.isna(raw_value):
                    category = 2  # masked
                else:
                    category = 0 if raw_value <= self.thresholds[feature] else 1
            else:  # Encode categorical
                raw_value = raw.get(feature, None)
                num_categories = len(self.cat_map[feature])
                if pd.isna(raw_value):
                    category = num_categories  # masked
                else:
                    category = self.cat_map[feature][raw_value]

            vector.append(category)

        sample_dict[key_out] = torch.tensor(vector, dtype=torch.long)
        return sample_dict


class OpClinicalAugmentation(OpBase):
    """
    Applies augmentation to categorical clinical vectors:
    - Randomly masks categorical values using reserved index
    """

    def __init__(
        self,
        dropout_p: float = 0.1,
        feature_names: List[str] = None,
        mask_token_index: Dict[str, int] = None,
    ):
        """
        :param dropout_p: probability of masking a token
        :param feature_names: list of all categorical features (including binarized continuous ones)
        :param mask_token_index: dict mapping each feature name to its reserved mask token index
        """
        super().__init__()
        self.dropout_p = dropout_p
        self.feature_names = feature_names
        self.mask_token_index = mask_token_index

    def __call__(self, sample_dict: NDict, key_in: str, key_out: Optional[str] = None) -> NDict:
        vec = sample_dict[key_in].clone()
        if key_out is None:
            key_out = key_in

        for i, feature in enumerate(self.feature_names):
            if random.random() < self.dropout_p:
                vec[i] = self.mask_token_index[feature]

        sample_dict[key_out] = vec
        return sample_dict


class OpClinicalMask(OpBase):
    """
    Generates a binary mask for clinical input (xa) based on mask token index.
    `True` where value is not the mask token, `False` where it is.
    """
    def __init__(self, feature_names: List[str], mask_token_index: Dict[str, int]):
        super().__init__()
        self.feature_names = feature_names
        self.mask_token_index = mask_token_index

    def __call__(self, sample_dict: NDict, key_in='data.input.clinical.vector', key_out='model.embed_mask_a') -> NDict:
        vec = sample_dict[key_in]  # shape: [num_features]
        mask = torch.ones_like(vec, dtype=torch.bool)

        for i, feat in enumerate(self.feature_names):
            if vec[i].item() == self.mask_token_index[feat]:
                mask[i] = False

        sample_dict[key_out] = mask # shape: [num_features]
        return sample_dict


class OpClinicalEmbedID(OpBase):
    """
    Generates per-feature embed_ids for TransformerWrapper, using string keys.

    Assumes input vector shape: [num_features] (each index is a token/category).
    Produces: model.embed_ids_a as a dict[str, Tensor] for each feature index as string.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def __call__(self, sample_dict, key_in='data.input.clinical.vector', key_out='model.embed_ids_a'):
        vec = sample_dict[key_in]  # shape: [num_features]
        embed_ids = {}

        for i in range(self.num_features):
            feature_key = str(i)
            embed_ids[feature_key] = vec[i].unsqueeze(0)  # shape: [1]

        sample_dict[key_out] = embed_ids
        return sample_dict

