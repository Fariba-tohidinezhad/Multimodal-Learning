import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nibabel as nib
from fuse.data.ops.op_base import OpBase
from fuse.utils import NDict
import torch
import torch.nn as nn
import random
from typing import Tuple, Optional, Sequence, List, Dict, Union
from scipy.ndimage import rotate
from einops import rearrange


class OpGISTLoadImage(OpBase):
    """
    Load a NIfTI image file (.nii.gz) from a given directory and convert it to [Z, Y, X] format.
    Stores the image array, original spacing, and original size in sample_dict.

    - key_in: str → filename key (e.g., 'file.nii.gz')
    - key_out: str → key under which the [Z, Y, X] array is stored
    """

    def __init__(self, dir_path: str):
        super().__init__()
        self._dir_path = dir_path

    def __call__(self, sample_dict: NDict, key_in: str, key_out: str) -> NDict:
        img_filename = os.path.join(self._dir_path, sample_dict[key_in])

        if not img_filename.endswith('.nii.gz'):
            raise ValueError(f"Expected a .nii.gz file, but got: {img_filename}")

        # --- Load image in [X, Y, Z] ---
        img = nib.load(img_filename)
        img_np = img.get_fdata()  # shape: [X, Y, Z]

        # --- Transpose to [Z, Y, X] ---
        img_np = np.transpose(img_np, axes=(2, 1, 0))  # shape: [Z, Y, X]

        # --- Extract spacing and size in [Z, Y, X] ---
        spacing_xyz = img.header.get_zooms()  # [X, Y, Z]
        original_spacing = np.array(spacing_xyz[::-1], dtype=np.float32)  # [Z, Y, X]
        original_size = np.array(img_np.shape, dtype=np.int32)            # [Z, Y, X]

        # --- Save into sample_dict ---
        sample_dict[key_out] = img_np.copy()
        sample_dict["data.input.img.original_spacing"] = original_spacing
        sample_dict["data.input.img.original_size"] = original_size

        return sample_dict


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


class GISTDataUtils:
    def __init__(self, dataset, img_dir, seg_dir):
        """
        Initialize the utility class with a dataset and data directories.

        Args:
            dataset (DatasetDefault): The dataset containing CT scan data.
            img_dir (str): Path to the directory containing CT scan images.
            seg_dir (str): Path to the directory containing segmentation masks.
        """
        self.dataset = dataset
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.unique_spacings = []

    def show_mid_slice(self, patient_id, ct_key="data.input.img", seg_key="data.input.seg"):
        """
        Display the mid-slice of a CT scan for a given patient ID, rotated by -90 degrees,
        and overlay the segmentation mask.

        Args:
            patient_id (str): The sample ID of the patient.
            ct_key (str): The key corresponding to the CT scan in the dataset dictionary.
            seg_key (str): The key corresponding to the segmentation mask in the dataset dictionary.
        """
        sample = self.dataset.getitem(patient_id)
        ct_scan = sample[ct_key]
        seg_mask = sample[seg_key]

        if hasattr(ct_scan, "numpy"):
            ct_scan = ct_scan.numpy()
        if hasattr(seg_mask, "numpy"):
            seg_mask = seg_mask.numpy()

        mid_slice = ct_scan.shape[2] // 2
        ct_slice = np.rot90(ct_scan[:, :, mid_slice], k=-1)
        seg_slice = np.rot90(seg_mask[:, :, mid_slice], k=-1)

        plt.figure(figsize=(6, 6))
        plt.imshow(ct_slice, cmap="gray")
        plt.imshow(seg_slice, cmap="jet", alpha=0.3)
        plt.show()

    def check_ct_seg_integrity(self):
        """
        Perform integrity checks on the dataset including voxel spacing, orientation,
        shape consistency, and presence of non-empty tumor masks.
        """
        unique_orientations = set()

        for sample in self.dataset:
            ct_img_path = sample["data.input.img_path"]
            seg_img_path = sample["data.input.seg_path"]

            ct_img = sitk.ReadImage(os.path.join(self.img_dir, ct_img_path))
            seg_img = sitk.ReadImage(os.path.join(self.seg_dir, seg_img_path))

            ct_spacing = ct_img.GetSpacing()
            seg_spacing = seg_img.GetSpacing()

            ct_direction = ct_img.GetDirection()
            seg_direction = seg_img.GetDirection()

            self.unique_spacings.append(ct_spacing)
            unique_orientations.add(ct_direction)

            if ct_spacing != seg_spacing:
                print(f"⚠️ Mismatch in voxel spacing for {ct_img_path}: {ct_spacing} vs {seg_spacing}")

            if ct_direction != seg_direction:
                print(f"⚠️ Mismatch in orientation for {ct_img_path}: {ct_direction} vs {seg_direction}")

            if ct_img.GetSize() != seg_img.GetSize():
                print(
                    f"⚠️ Shape mismatch: {ct_img_path} has size {ct_img.GetSize()}, but segmentation is {seg_img.GetSize()}")

            seg_array = sitk.GetArrayFromImage(seg_img)
            if np.sum(seg_array) == 0:
                print(f"⚠️ Empty mask detected: {seg_img_path}")

        print("\n✅ Unique voxel spacings found:", set(self.unique_spacings))
        print("✅ Unique orientations found:", unique_orientations)


    def find_mean_median_spacing(self):
        """
        Calculate the mean and median voxel spacing from all CT scans in the dataset.
        """
        spacings = []

        for sample in self.dataset:
            ct_img_path = sample["data.input.img_path"]
            ct_img = sitk.ReadImage(os.path.join(self.img_dir, ct_img_path))
            spacings.append(ct_img.GetSpacing())

        spacings = np.array(spacings)  # Convert to NumPy array for easy calculations
        mean_spacing = np.mean(spacings, axis=0)
        median_spacing = np.median(spacings, axis=0)

        min_spacing = np.min(spacings, axis=0)
        max_spacing = np.max(spacings, axis=0)

        print(f"Min Spacing:{min_spacing}, Max Spacing:{max_spacing}")
        print(f"Mean Spacing: {mean_spacing}")
        print(f"Median Spacing: {median_spacing}")


class OpGISTResample(OpBase):
    """
    Resamples CT scans and segmentation masks to a specified spacing (Z, Y, X).
    Assumes all arrays are in [Z, Y, X] format and returns resampled arrays in the same format.
    """

    def __init__(self, target_spacing_z, target_spacing_y, target_spacing_x):
        super().__init__()
        # Store as tuple to be converted later
        self._target_spacing = (target_spacing_z, target_spacing_y, target_spacing_x)

    def __call__(self, sample_dict: NDict) -> NDict:
        # Convert spacing to NumPy array (Z, Y, X)
        target_spacing = np.array(self._target_spacing, dtype=np.float32)

        # --- Check required metadata ---
        if "data.input.img.original_spacing" not in sample_dict or \
           "data.input.img.original_size" not in sample_dict:
            raise ValueError("Missing original spacing or size in metadata.")

        ct_np = sample_dict["data.input.img"]        # shape: [Z, Y, X]
        seg_np = sample_dict["data.input.seg"]       # shape: [Z, Y, X]

        original_spacing = sample_dict["data.input.img.original_spacing"]  # [Z, Y, X]
        original_size = sample_dict["data.input.img.original_size"]        # [Z, Y, X]

        # --- Convert to SimpleITK Images ---
        ct_sitk = sitk.GetImageFromArray(ct_np)
        seg_sitk = sitk.GetImageFromArray(seg_np)

        ct_sitk.SetSpacing(original_spacing[::-1].tolist())  # to [X, Y, Z]
        seg_sitk.SetSpacing(original_spacing[::-1].tolist())

        # --- Compute new size ---
        new_size = np.round(original_size * (original_spacing / target_spacing)).astype(int)

        # --- Resample CT ---
        resampled_ct = self.resample_image(
            ct_sitk,
            target_spacing=target_spacing[::-1],  # to [X, Y, Z]
            new_size=new_size[::-1],
            interpolator=sitk.sitkLinear
        )

        # --- Resample segmentation ---
        resampled_seg = self.resample_image(
            seg_sitk,
            target_spacing=target_spacing[::-1],
            new_size=new_size[::-1],
            interpolator=sitk.sitkNearestNeighbor
        )

        # --- Convert back to NumPy (still [Z, Y, X]) ---
        sample_dict["data.input.img.resampled"] = sitk.GetArrayFromImage(resampled_ct).copy()
        sample_dict["data.input.seg.resampled"] = sitk.GetArrayFromImage(resampled_seg).copy()

        return sample_dict

    def resample_image(self, image, target_spacing, new_size, interpolator):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetOutputSpacing([float(s) for s in target_spacing])
        resampler.SetSize([int(s) for s in new_size])
        resampler.SetInterpolator(interpolator)
        return resampler.Execute(image)


class OpClipAndNormalizeMasked(OpBase):
    """
    Normalize CT values within the tumor region only, using min-max normalization.
    Assumes input CT and segmentation volumes are in [Z, Y, X] format.

    - Voxels outside the tumor (seg == 0) are set to `background_val`
    - Tumor voxels are clipped to `clip_range`, then normalized to `to_range`
    - Outputs both normalized CT and segmentation to key_out = (img_out, seg_out)
    """

    def __init__(
        self,
        clip_range: Tuple[float, float] = (-150, 250),
        to_range: Tuple[float, float] = (0.0, 1.0),
        background_val: float = -1.0,
    ):
        super().__init__()
        self.clip_range = clip_range
        self.to_range = to_range
        self.background_val = background_val

    def __call__(
        self,
        sample_dict: NDict,
        key_in: Tuple[str, str],
        key_out: Tuple[str, str],
    ) -> NDict:
        key_img, key_seg = key_in
        key_img_out, key_seg_out = key_out

        ct = sample_dict[key_img]    # expected shape: [Z, Y, X]
        seg = sample_dict[key_seg]   # expected shape: [Z, Y, X]

        assert ct.shape == seg.shape, \
            f"CT and segmentation shape mismatch: {ct.shape} vs {seg.shape}"

        # Create binary tumor mask
        mask = seg > 0

        # Output initialized to background everywhere
        out_ct = np.full_like(ct, fill_value=self.background_val, dtype=np.float32)

        # Normalize only tumor voxels
        tumor_voxels = ct[mask]
        clipped = np.clip(tumor_voxels, self.clip_range[0], self.clip_range[1])
        scale = (self.to_range[1] - self.to_range[0]) / (self.clip_range[1] - self.clip_range[0])
        normalized = (clipped - self.clip_range[0]) * scale + self.to_range[0]
        out_ct[mask] = normalized

        # Store both outputs
        sample_dict[key_img_out] = out_ct.copy().astype(np.float32)
        sample_dict[key_seg_out] = seg.copy().astype(np.float32)

        return sample_dict


class OpTumorCrop(OpBase):
    """
    Crop both CT and segmentation arrays using the 3D bounding box of the tumor.
    Assumes inputs are in [Z, Y, X] format.

    Input:
        - key_in: Tuple[str, str] → (ct_key, seg_key)
        - key_out: Tuple[str, str] → (cropped_ct_key, cropped_seg_key)
    """

    def __init__(self):
        super().__init__()

    def __call__(self, sample_dict: NDict, key_in: Tuple[str, str], key_out: Tuple[str, str]) -> NDict:
        key_ct_in, key_seg_in = key_in
        key_ct_out, key_seg_out = key_out

        ct = sample_dict[key_ct_in]    # [Z, Y, X]
        seg = sample_dict[key_seg_in]  # [Z, Y, X]

        assert ct.shape == seg.shape, \
            f"CT and segmentation shape mismatch: {ct.shape} vs {seg.shape}"

        # Find tumor bounding box (non-zero region)
        nonzero_indices = np.argwhere(seg > 0)
        if nonzero_indices.size == 0:
            raise ValueError(f"No tumor region found in segmentation for sample: {sample_dict.get('data.sample_id', 'UNKNOWN')}")

        z_min, y_min, x_min = nonzero_indices.min(axis=0)
        z_max, y_max, x_max = nonzero_indices.max(axis=0) + 1  # inclusive upper bound

        # Apply crop
        ct_crop = ct[z_min:z_max, y_min:y_max, x_min:x_max].copy()
        seg_crop = seg[z_min:z_max, y_min:y_max, x_min:x_max].copy()

        # Store output
        sample_dict[key_ct_out] = ct_crop
        sample_dict[key_seg_out] = seg_crop

        return sample_dict


class OpTumorRandomRotation(OpBase):
    """
    Applies a random 3D rotation to both CT and segmentation volumes in [Z, Y, X] format.

    - Rotates around randomly chosen axes (e.g., (Z,Y), (Z,X), (Y,X))
    - Uses cubic interpolation for CT, nearest for segmentation
    - Pads missing regions with cval: -1024 for CT, 0 for segmentation
    """

    def __init__(
        self,
        angle_range: Tuple[float, float] = (-10.0, 10.0),
        axes_options: Optional[Sequence[Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.angle_range = angle_range
        self.axes_options = axes_options if axes_options is not None else [(0, 1), (0, 2), (1, 2)]  # (Z, Y), (Z, X), (Y, X)

    def __call__(self, sample_dict: NDict, key_in: Tuple[str, str], key_out: Tuple[str, str]) -> NDict:
        key_ct_in, key_seg_in = key_in
        key_ct_out, key_seg_out = key_out

        ct = sample_dict[key_ct_in]  # [Z, Y, X]
        seg = sample_dict[key_seg_in]  # [Z, Y, X]

        assert ct.shape == seg.shape, \
            f"CT and segmentation shape mismatch: {ct.shape} vs {seg.shape}"

        # Sample random rotation
        angle = random.uniform(*self.angle_range)
        axes = random.choice(self.axes_options)

        # Apply rotation
        ct_rot = rotate(
            ct, angle=angle, axes=axes,
            reshape=True, order=3, mode='constant', cval=-1024.0
        )
        seg_rot = rotate(
            seg, angle=angle, axes=axes,
            reshape=True, order=0, mode='constant', cval=0
        )

        sample_dict[key_ct_out] = ct_rot.copy()
        sample_dict[key_seg_out] = seg_rot.copy()

        # Optional debug info
        sample_dict['data.input.img.rotated.info'] = {
            'angle_deg': angle,
            'axes': axes
        }

        return sample_dict


class OpPadOrCropToFixedShape(OpBase):
    """
    Pads or center-crops a 3D tumor volume and segmentation to a fixed shape [Z, Y, X].

    - Pads with -1.0 for the image and 0 for the segmentation if input is smaller
    - Center-crops if input is larger (e.g., due to rotation)
    - Returns both processed image and binary segmentation mask
    """

    def __init__(self, largest_tumor: Tuple[int, int, int]):
        """
        :param largest_tumor: target fixed shape [Z, Y, X] for the network input
        """
        super().__init__()
        self.largest_tumor = largest_tumor
        self.pad_val_img = -1.0
        self.pad_val_seg = 0.0

    def __call__(
        self,
        sample_dict: NDict,
        key_in: Tuple[str, str] = ('data.input.img.tumor3d.norm', 'data.input.seg.tumor3d.norm'),
        key_out: Tuple[str, str] = ('data.input.img.tumor3d.fitted', 'data.input.seg.tumor3d.fitted')
    ) -> NDict:
        key_img_in, key_seg_in = key_in
        key_img_out, key_seg_out = key_out

        img = sample_dict[key_img_in]  # [Z, Y, X]
        seg = sample_dict[key_seg_in]  # [Z, Y, X]
        assert img.shape == seg.shape, f"Shape mismatch: {img.shape} vs {seg.shape}"

        target_shape = self.largest_tumor
        fitted_img = img.copy()
        fitted_seg = (seg > 0).astype(np.float32)  # ensure binary

        for dim in range(3):  # Z, Y, X
            input_dim = fitted_img.shape[dim]
            target_dim = target_shape[dim]

            if input_dim < target_dim:
                # Pad symmetrically
                total_pad = target_dim - input_dim
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                pad_cfg = [(0, 0), (0, 0), (0, 0)]
                pad_cfg[dim] = (pad_before, pad_after)
                fitted_img = np.pad(
                    fitted_img, pad_cfg, mode='constant', constant_values=self.pad_val_img
                )
                fitted_seg = np.pad(
                    fitted_seg, pad_cfg, mode='constant', constant_values=self.pad_val_seg
                )

            elif input_dim > target_dim:
                # Center-crop
                excess = input_dim - target_dim
                start = excess // 2
                end = start + target_dim
                fitted_img = np.take(fitted_img, range(start, end), axis=dim)
                fitted_seg = np.take(fitted_seg, range(start, end), axis=dim)

        assert fitted_img.shape == target_shape and fitted_seg.shape == target_shape, \
            f"Expected shape {target_shape}, got {fitted_img.shape} and {fitted_seg.shape}"

        # Convert both to tensors
        sample_dict[key_img_out] = torch.tensor(fitted_img, dtype=torch.float32).unsqueeze(0)
        sample_dict[key_seg_out] = torch.tensor(fitted_seg, dtype=torch.float32).unsqueeze(0)

        return sample_dict


class OpEncodeMetaDataGIST(OpBase):
    """
    Encode GIST clinical meta-data into one-hot vectors (as torch tensors).

    Reads all clinical features from sample_dict[key_in],
    encodes them, and writes one-hot vectors under:
        data.input.clinical.encoding
    """

    def __init__(
        self,
        feature_names: list[str],
        categorical_mappings: Dict[str, Dict[str, int]],
        mask_token_index: Dict[str, int],
        thresholds: Dict[str, float],
    ):
        """
        :param feature_names: list of all features to encode
        :param categorical_mappings: mapping for categorical variables
        :param mask_token_index: dict mapping each variable to its N/A (mask) index
        :param thresholds: dict with thresholds for continuous variables
        """
        super().__init__()
        self.feature_names = feature_names
        self.categorical_mappings = categorical_mappings
        self.mask_token_index = mask_token_index
        self.thresholds = thresholds

    def __call__(
        self,
        sample_dict: NDict,
        key_in: str = "data.input.clinical.raw",
    ) -> NDict:
        """
        Reads clinical features from sample_dict[key_in] and stores encoded
        torch tensors under sample_dict["data.input.clinical.encoding"].
        """
        raw_data = sample_dict[key_in]
        encodings = {}

        for feature in self.feature_names:
            value = raw_data.get(feature, None)
            mask_idx = self.mask_token_index.get(feature, None)

            # --- Continuous variables ---
            if feature in self.thresholds:
                thr = self.thresholds[feature]
                one_hot = np.zeros(3, dtype=np.float32)  # [≤thr, >thr, missing]

                # treat "-1" or "N/A" as missing
                if (
                    value is None
                    or (isinstance(value, float) and np.isnan(value))
                    or str(value).strip() in ["-1", "N/A", ""]
                ):
                    one_hot[mask_idx] = 1
                else:
                    try:
                        v = float(value)
                        one_hot[0 if v <= thr else 1] = 1
                    except (ValueError, TypeError):
                        one_hot[mask_idx] = 1

                encodings[feature] = torch.tensor(one_hot, dtype=torch.float32)

            # --- Categorical variables ---
            elif feature in self.categorical_mappings:
                mapping = self.categorical_mappings[feature]
                one_hot = np.zeros(len(mapping) + 1, dtype=np.float32)  # +1 for mask token

                # treat "N/A", "-1", or empty as missing
                if (
                    value is None
                    or str(value).strip() in ["N/A", "-1", ""]
                ):
                    one_hot[mask_idx] = 1
                elif value in mapping:
                    idx = mapping[value]
                    one_hot[idx] = 1
                else:
                    one_hot[mask_idx] = 1

                encodings[feature] = torch.tensor(one_hot, dtype=torch.float32)

            # --- Unknown variable ---
            else:
                raise ValueError(
                    f"Feature '{feature}' not found in thresholds or categorical mappings."
                )

        sample_dict["data.input.clinical.encoding"] = encodings
        return sample_dict




class OpAugOneHotGIST(OpBase):
    """
    Apply feature-wise one-hot dropout augmentation for GIST clinical encodings.

    With probability `dropout_p`, replaces the current one-hot vector
    with the mask token (N/A) index specified in mask_token_index[feature].

    Reads and writes augmented features directly under sample_dict[key_in].
    """

    def __init__(
        self,
        dropout_p: float,
        feature_names: list[str],
        mask_token_index: dict,
    ):
        """
        :param dropout_p: probability of dropping (masking) a feature's one-hot encoding
        :param feature_names: list of clinical feature names to augment
        :param mask_token_index: mapping {feature_name: mask_index}
        """
        super().__init__()
        self.dropout_p = dropout_p
        self.feature_names = feature_names
        self.mask_token_index = mask_token_index

    def __call__(
        self,
        sample_dict: NDict,
        key_in: str = "data.input.clinical.encoding",
    ) -> NDict:
        """
        Apply one-hot dropout augmentation to each feature in feature_names.
        Overwrites the augmented result directly on key_in.
        """
        input_encodings = sample_dict[key_in]

        for feature in self.feature_names:
            one_hot = input_encodings[feature]
            assert isinstance(one_hot, torch.Tensor), \
                f"Expected tensor for {feature}, got {type(one_hot)}"

            one_hot_aug = one_hot.clone()

            # With probability dropout_p, replace with mask token one-hot
            if random.random() < self.dropout_p:
                mask_idx = self.mask_token_index[feature]
                one_hot_aug = torch.zeros_like(one_hot)
                one_hot_aug[mask_idx] = 1.0

            # overwrite directly
            input_encodings[feature] = one_hot_aug

        sample_dict[key_in] = input_encodings
        return sample_dict

