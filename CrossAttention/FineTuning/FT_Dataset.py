# =========================
# GIST_Dataset_Downstream.py
# (updated imaging pipeline -> NEW Ops)
# clinical pipeline stays the same
# =========================

from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.utils.ndict import NDict
from fuse.data.ops.op_base import OpReversibleBase
from fuse.data.ops.ops_common import OpKeepKeypaths
from typing import Optional, Sequence, Tuple
import pandas as pd
import os

# --- NEW imaging ops (your downstream ops) ---
from FT_DataUtils import (
    OpIDPath,
    OpLoadImage,
    OpResample,
    OpClipAndMinMaxNormalize,
    OpCropOrPadToFixedShapeUsingSeg,
    OpTumorRandomRotation,
    OpTumorPatchifyWithMask,
    OpCastLabelToFloat,
    OpClinicalPreprocess,
    OpClinicalAugmentation,
    OpClinicalMask,
    OpClinicalEmbedID,
)


class GISTDataset:
    @staticmethod
    def sample_ids(data_dir_seg: str) -> Sequence[str]:
        # IMPORTANT: ".nii.gz" -> remove 7 chars
        return sorted([f[:-7] for f in os.listdir(data_dir_seg) if f.endswith(".nii.gz")])

    @staticmethod
    def setup_clinical_preprocessing(df_clinical: pd.DataFrame, sample_ids: Sequence[str]):
        df_filtered = df_clinical[df_clinical["sample_id"].isin(sample_ids)]

        thresholds = {
            "Age_at_Imatinib": 65,
            "TumorSize": 100,
            "MIT": 5,
        }
        categorical_mappings = {
            "Gender": {"Male": 0, "Female": 1},
            "PrimaryTumorSite": {
                "Colon": 0,
                "Duodenal": 1,
                "Esophagus": 2,
                "Gastric": 3,
                "Rectum": 4,
                "Small bowel": 5,
                "Other": 6,
            },
            "StatusatDiagnosis": {"Localized disease": 0, "Locally advanced": 1, "Metastasized": 2},
            "Histology": {"Spindle cell": 0, "Epitheloid": 1, "Mixed type": 2},
            "CD117": {"Positive": 1, "Negative": 0},
            "DOG1": {"Positive": 1, "Negative": 0},
            "KIT": {"Present": 1, "Absent": 0},
            "PDGFR": {"Present": 1, "Absent": 0},
            "BRAF": {"Present": 1, "Absent": 0},
            "Diabetes": {"Yes": 1, "No": 0},
            "Hypertension": {"Yes": 1, "No": 0},
            "Hypercholesterolemia": {"Yes": 1, "No": 0},
            "OtherCancer": {"Yes": 1, "No": 0},
        }

        feature_names = [
            "Age_at_Imatinib",
            "TumorSize",
            "MIT",
            "Gender",
            "PrimaryTumorSite",
            "StatusatDiagnosis",
            "Histology",
            "CD117",
            "DOG1",
            "KIT",
            "PDGFR",
            "BRAF",
            "Diabetes",
            "Hypertension",
            "Hypercholesterolemia",
            "OtherCancer",
        ]

        mask_token_index = {
            "Age_at_Imatinib": 2,
            "TumorSize": 2,
            "MIT": 2,
            "Gender": 2,
            "PrimaryTumorSite": 7,
            "StatusatDiagnosis": 3,
            "Histology": 3,
            "CD117": 2,
            "DOG1": 2,
            "KIT": 2,
            "PDGFR": 2,
            "BRAF": 2,
            "Diabetes": 2,
            "Hypertension": 2,
            "Hypercholesterolemia": 2,
            "OtherCancer": 2,
        }

        return thresholds, categorical_mappings, feature_names, mask_token_index

    @staticmethod
    def static_pipeline(
        data_dir_img: str,
        data_dir_seg: str,
        df_clinical: pd.DataFrame,
        thresholds,
        categorical_mappings,
        # --- imaging knobs (static part) ---
        target_spacing_zyx: Tuple[float, float, float] = (5.0, 0.7636719, 0.7636719),
        clip_range_hu: Tuple[float, float] = (-150.0, 250.0),
        target_shape_zyx: Tuple[int, int, int] = (80, 347, 498),
        pad_val: float = 0.0,  # normalized padding
    ) -> PipelineDefault:
        return PipelineDefault(
            "static",
            [
                # ---------- IMAGING (NEW OPS, same order as defined) ----------
                (OpIDPath(), dict()),
                (OpLoadImage(dir_path=data_dir_img, is_mask=False), dict(key_in="data.input.img_path", key_out="data.input.img")),
                (OpLoadImage(dir_path=data_dir_seg, is_mask=True), dict(key_in="data.input.seg_path", key_out="data.input.seg")),
                (
                    OpResample(
                        target_spacing_z=float(target_spacing_zyx[0]),
                        target_spacing_y=float(target_spacing_zyx[1]),
                        target_spacing_x=float(target_spacing_zyx[2]),
                    ),
                    dict(),
                ),
                (
                    OpClipAndMinMaxNormalize(clip_range=clip_range_hu),
                    dict(key_in="data.input.img.resampled", key_out="data.input.img.clipnorm"),
                ),
                (
                    OpCropOrPadToFixedShapeUsingSeg(target_shape_zyx=target_shape_zyx, pad_val=pad_val),
                    dict(
                        key_in=("data.input.img.clipnorm", "data.input.seg.resampled"),
                        key_out=("data.input.img.fitted", "data.input.seg.fitted"),
                    ),
                ),

                # ---------- CLINICAL (UNCHANGED) ----------
                (
                    OpReadDataframe(
                        data=df_clinical,
                        columns_to_extract=[
                            "sample_id",
                            "Center",
                            "Age_at_Imatinib",
                            "Gender",
                            "StatusatDiagnosis",
                            "PrimaryTumorSite",
                            "TumorSize",
                            "Histology",
                            "MIT",
                            "CD117",
                            "DOG1",
                            "KIT",
                            "PDGFR",
                            "BRAF",
                            "Diabetes",
                            "Hypertension",
                            "Hypercholesterolemia",
                            "OtherCancer",
                            "TKIResponse",
                        ],
                        key_column="sample_id",
                        key_name="data.sample_id",
                    ),
                    dict(prefix="data.input.clinical.raw"),
                ),
                (
                    OpClinicalPreprocess(thresholds=thresholds, categorical_mappings=categorical_mappings),
                    dict(key_in="data.input.clinical.raw", key_out="data.input.clinical.vector"),
                ),
            ],
        )

    @staticmethod
    def dynamic_pipeline(
        # --- imaging knobs (dynamic part) ---
        patch_size_zyx: Tuple[int, int, int] = (16, 64, 64),
        min_tumor_frac: float = 0.3,
        pad_to_multiple: bool = True,
        pad_val: float = 0.0,  # normalized padding
        angle_range_deg: Tuple[float, float] = (-10.0, 10.0),
        rotation_prob: float = 0.5,
        train: bool = False,
        # --- clinical knobs (unchanged) ---
        feature_names=None,
        mask_token_index=None,
        dropout_p: float = 0.1,
    ) -> PipelineDefault:
        steps = []

        # ---------- IMAGING AUG (NEW ROTATION) ----------
        # NOTE: rotate AFTER fitting (exactly like your check script)
        if train:
            steps.append(
                (
                    OpTumorRandomRotation(
                        angle_range_deg=angle_range_deg,
                        prob=rotation_prob,
                        pad_val=pad_val,
                        axes_options=[(1, 2)],
                        store_debug=False,
                    ),
                    dict(
                        key_in=("data.input.img.fitted", "data.input.seg.fitted"),
                        key_out=("data.input.img.fitted", "data.input.seg.fitted"),  # overwrite in-place
                    ),
                )
            )

        # ---------- IMAGING PATCHIFY (NEW PATCHIFY WITH MASK) ----------
        steps.append(
            (
                OpTumorPatchifyWithMask(
                    patch_size_zyx=patch_size_zyx,
                    min_tumor_frac=min_tumor_frac,
                    pad_to_multiple=pad_to_multiple,
                    pad_val=pad_val,
                    store_debug=False,
                ),
                dict(
                    key_in=("data.input.img.fitted", "data.input.seg.fitted"),
                    key_out=("data.input.img.tumor3d.patches", "model.embed_mask_b", "data.input.img.patches_grid_zyx"),
                ),
            )
        )

        # ---------- CLINICAL AUG (UNCHANGED) ----------
        if train:
            steps.append(
                (
                    OpClinicalAugmentation(
                        dropout_p=dropout_p,
                        feature_names=feature_names,
                        mask_token_index=mask_token_index,
                    ),
                    dict(key_in="data.input.clinical.vector"),
                )
            )

        # Clinical mask for TransformerWrapper
        steps.append(
            (
                OpClinicalMask(feature_names=feature_names, mask_token_index=mask_token_index),
                dict(key_in="data.input.clinical.vector"),
            )
        )

        # Clinical embed_ids
        steps.append((OpClinicalEmbedID(num_features=16), dict()))

        # Cast label to float
        steps.append(
            (
                OpCastLabelToFloat(),
                dict(
                    key_in="data.input.clinical.raw.TKIResponse",
                    key_out="data.input.clinical.raw.TKIResponse.f",
                ),
            )
        )

        keep_keys = [
            "data.sample_id",
            "data.input.clinical.raw.Center",
            "data.input.clinical.vector",
            "model.embed_ids_a",
            "model.embed_mask_a",
            "data.input.img.tumor3d.patches",
            "model.embed_mask_b",
            "data.input.img.patches_grid_zyx",
            "data.input.clinical.raw.TKIResponse",
            "data.input.clinical.raw.TKIResponse.f",
        ]
        steps.append((OpKeepKeypaths(), {"keep_keypaths": keep_keys}))

        return PipelineDefault("dynamic", steps)

    @staticmethod
    def dataset(
        data_dir_img: str,
        data_dir_seg: str,
        clinical_csv_path: str,
        # imaging knobs
        target_spacing_zyx: Tuple[float, float, float] = (5.0, 0.7636719, 0.7636719),
        clip_range_hu: Tuple[float, float] = (-150.0, 250.0),
        target_shape_zyx: Tuple[int, int, int] = (80, 347, 498),
        pad_val: float = 0.0,
        patch_size_zyx: Tuple[int, int, int] = (16, 64, 64),
        min_tumor_frac: float = 0.3,
        pad_to_multiple: bool = True,
        angle_range_deg: Tuple[float, float] = (-10.0, 10.0),
        rotation_prob: float = 0.5,
        # clinical knobs
        dropout_p: float = 0.1,
        # control
        train: bool = False,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> DatasetDefault:
        df_clinical = pd.read_csv(clinical_csv_path)

        if sample_ids is None:
            sample_ids = GISTDataset.sample_ids(data_dir_seg)

        thresholds, categorical_mappings, feature_names, mask_token_index = GISTDataset.setup_clinical_preprocessing(
            df_clinical, sample_ids
        )

        static_pipeline = GISTDataset.static_pipeline(
            data_dir_img=data_dir_img,
            data_dir_seg=data_dir_seg,
            df_clinical=df_clinical,
            thresholds=thresholds,
            categorical_mappings=categorical_mappings,
            target_spacing_zyx=target_spacing_zyx,
            clip_range_hu=clip_range_hu,
            target_shape_zyx=target_shape_zyx,
            pad_val=pad_val,
        )

        dynamic_pipeline = GISTDataset.dynamic_pipeline(
            patch_size_zyx=patch_size_zyx,
            min_tumor_frac=min_tumor_frac,
            pad_to_multiple=pad_to_multiple,
            pad_val=pad_val,
            angle_range_deg=angle_range_deg,
            rotation_prob=rotation_prob,
            train=train,
            feature_names=feature_names,
            mask_token_index=mask_token_index,
            dropout_p=dropout_p,
        )

        dataset = DatasetDefault(
            sample_ids=sample_ids,
            static_pipeline=static_pipeline,
            dynamic_pipeline=dynamic_pipeline,
        )
        dataset.create()
        return dataset
