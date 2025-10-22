from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.utils.ndict import NDict
from fuse.data.ops.op_base import OpReversibleBase
from fuse.data.ops.ops_common import OpKeepKeypaths, OpOverrideNaN
from typing import Optional, Sequence, Tuple
import pandas as pd
import os

from GISTDataUtils import (
    OpGISTLoadImage, OpCastLabelToFloat, OpGISTResample, OpTumorCrop,
    OpClipAndNormalizeMasked, OpTumorRandomRotation, OpPadOrCropToFixedShape,
    OpEncodeMetaDataGIST, OpAugOneHotGIST
)

class OpGISTSampleIDDecode(OpReversibleBase):
    # derive file paths from sample_id
    def __call__(self, sample_dict: NDict, op_id: Optional[str] = None) -> NDict:
        sid = sample_dict["data.sample_id"]
        sample_dict["data.input.img_path"] = f"{sid}.nii.gz"
        sample_dict["data.input.seg_path"] = f"{sid}.nii.gz"
        return sample_dict

class GISTDataset:
    def sample_ids(data_dir_seg: str) -> Sequence[str]:
        return [
            f.split(".")[0]
            for f in os.listdir(data_dir_seg)
            if f.endswith(".nii.gz")
        ]

    def setup_clinical_preprocessing(df_clinical: pd.DataFrame, sample_ids: Sequence[str]):
        df_filtered = df_clinical[df_clinical['sample_id'].isin(sample_ids)]
        thresholds = {
            'Age_at_Imatinib': 65,
            'TumorSize': 100,
            'MIT': 5,
        }
        categorical_mappings = {
            'Gender': {'Male': 0, 'Female': 1},
            'PrimaryTumorSite': {
                'Colon': 0, 'Duodenal': 1, 'Esophagus': 2, 'Gastric': 3,
                'Rectum': 4, 'Small bowel': 5, 'Other': 6
            },
            'StatusatDiagnosis': {'Localized disease': 0, 'Locally advanced': 1, 'Metastasized': 2},
            'Histology': {'Spindle cell': 0, 'Epitheloid': 1, 'Mixed type': 2},
            'CD117': {'Positive': 1, 'Negative': 0},
            'DOG1': {'Positive': 1, 'Negative': 0},
            'KIT': {'Present': 1, 'Absent': 0},
            'PDGFR': {'Present': 1, 'Absent': 0},
            'BRAF': {'Present': 1, 'Absent': 0},
            'Diabetes': {'Yes': 1, 'No': 0},
            'Hypertension': {'Yes': 1, 'No': 0},
            'Hypercholesterolemia': {'Yes': 1, 'No': 0},
            'OtherCancer': {'Yes': 1, 'No': 0}
        }
        feature_names = [
            "Age_at_Imatinib", "TumorSize", "MIT",
            "Gender", "PrimaryTumorSite", "StatusatDiagnosis", "Histology",
            "CD117", "DOG1", "KIT", "PDGFR", "BRAF",
            "Diabetes", "Hypertension", "Hypercholesterolemia", "OtherCancer"
        ]
        mask_token_index = {
            "Age_at_Imatinib": 2, "TumorSize": 2, "MIT": 2,
            "Gender": 2, "PrimaryTumorSite": 7, "StatusatDiagnosis": 3, "Histology": 3,
            "CD117": 2, "DOG1": 2, "KIT": 2, "PDGFR": 2, "BRAF": 2,
            "Diabetes": 2, "Hypertension": 2, "Hypercholesterolemia": 2, "OtherCancer": 2
        }
        return thresholds, categorical_mappings, feature_names, mask_token_index

    def static_pipeline(
        data_dir_img: str,
        data_dir_seg: str,
        df_clinical: pd.DataFrame,
        thresholds,
        categorical_mappings,
        feature_names,
        mask_token_index
    ) -> PipelineDefault:

        return PipelineDefault("static", [
            (OpGISTSampleIDDecode(), dict()),
            (OpGISTLoadImage(data_dir_img), dict(key_in="data.input.img_path", key_out="data.input.img")),
            (OpGISTLoadImage(data_dir_seg), dict(key_in="data.input.seg_path", key_out="data.input.seg")),
            (OpReadDataframe(
                data=df_clinical,
                columns_to_extract=["sample_id", "Center", "Age_at_Imatinib", "Gender",
                                    "StatusatDiagnosis", "PrimaryTumorSite", "TumorSize",
                                    "Histology", "MIT", "CD117", "DOG1", "KIT", "PDGFR", "BRAF",
                                    "Diabetes", "Hypertension", "Hypercholesterolemia", "OtherCancer",
                                    "TKIResponse"],
                key_column="sample_id",
                key_name="data.sample_id"
            ), dict(prefix="data.input.clinical.raw")),

            (OpOverrideNaN(), dict(key="data.input.clinical.raw.TumorSize", value_to_fill="-1")),
            (OpOverrideNaN(), dict(key="data.input.clinical.raw.Histology", value_to_fill="N/A")),
            (OpOverrideNaN(), dict(key="data.input.clinical.raw.MIT", value_to_fill="-1")),
            (OpOverrideNaN(), dict(key="data.input.clinical.raw.CD117", value_to_fill="N/A")),
            (OpOverrideNaN(), dict(key="data.input.clinical.raw.DOG1", value_to_fill="N/A")),
            (OpOverrideNaN(), dict(key="data.input.clinical.raw.KIT", value_to_fill="N/A")),
            (OpOverrideNaN(), dict(key="data.input.clinical.raw.PDGFR", value_to_fill="N/A")),
            (OpOverrideNaN(), dict(key="data.input.clinical.raw.BRAF", value_to_fill="N/A")),

            (OpEncodeMetaDataGIST(feature_names=feature_names, categorical_mappings=categorical_mappings,
                                  mask_token_index=mask_token_index, thresholds=thresholds
            ), dict(key_in="data.input.clinical.raw"))
        ])

    def dynamic_pipeline(
        largest_tumor: Tuple[int, int, int] = (80, 347, 498),
        train: bool = False,
        feature_names=None,
        mask_token_index=None,
        angle_range: Tuple[float, float] = (-10, 10),     # NEW: imaging augmentation range (degrees)
        dropout_p: float = 0.1                            # NEW: clinical augmentation dropout prob
    ) -> PipelineDefault:

        steps = []

        # Resample
        steps.append((
            OpGISTResample(
                target_spacing_x=0.7636719,
                target_spacing_y=0.7636719,
                target_spacing_z=5.0
            ), dict()
        ))

        # Augmentation (rotation) — only in training
        if train:
            steps.append((
                OpTumorRandomRotation(angle_range=angle_range, axes_options=[(1, 2)]), dict(
                    key_in=("data.input.img.resampled", "data.input.seg.resampled"),
                    key_out=("data.input.img.rotated", "data.input.seg.rotated")
                )
            ))
            crop_input_keys = ("data.input.img.rotated", "data.input.seg.rotated")
        else:
            crop_input_keys = ("data.input.img.resampled", "data.input.seg.resampled")

        # Tumor crop
        steps.append((
            OpTumorCrop(), dict(
                key_in=crop_input_keys,
                key_out=("data.input.img.tumor3d", "data.input.seg.tumor3d")
            )
        ))

        # Normalize
        steps.append((
            OpClipAndNormalizeMasked(), dict(
                key_in=("data.input.img.tumor3d", "data.input.seg.tumor3d"),
                key_out=("data.input.img.tumor3d.norm", "data.input.seg.tumor3d.norm")
            )
        ))

        # Pad/Crop to fixed shape
        steps.append((
            OpPadOrCropToFixedShape(
                largest_tumor=largest_tumor
            ),
            dict(
                key_in=('data.input.img.tumor3d.norm', 'data.input.seg.tumor3d.norm'),
                key_out=('data.input.img.tumor3d.fitted', 'data.input.seg.tumor3d.fitted')
            )
        ))

        # Clinical augmentation (training only) with configurable dropout
        if train:
            steps.append((
                OpAugOneHotGIST(dropout_p=dropout_p, feature_names=feature_names, mask_token_index=mask_token_index),
                dict(key_in="data.input.clinical.encoding")
            ))

        # Cast label to float
        steps.append((
            OpCastLabelToFloat(), dict(
                key_in="data.input.clinical.raw.TKIResponse",
                key_out="data.input.clinical.raw.TKIResponse.f"
            )
        ))

        keep_keys = [
            "data.sample_id",
            "data.input.clinical.raw.Center",
            "data.input.clinical.encoding",
            "data.input.img.tumor3d.fitted",
            "data.input.seg.tumor3d.fitted", # mask_key in Head3DGlobalFusionMasked
            "data.input.clinical.raw.TKIResponse",
            "data.input.clinical.raw.TKIResponse.f",
        ]
        steps.append((OpKeepKeypaths(), {'keep_keypaths': keep_keys}))

        return PipelineDefault("dynamic", steps)

    def dataset(
        data_dir_img: str,
        data_dir_seg: str,
        clinical_csv_path: str,
        largest_tumor: Tuple[int, int, int] = (80, 347, 498),
        train: bool = False,
        sample_ids: Optional[Sequence[str]] = None,
        angle_range: Tuple[float, float] = (-10, 10),
        dropout_p: float = 0.1
    ) -> DatasetDefault:
        """
        Build GIST dataset with configurable augmentation & masking knobs.
        """

        df_clinical = pd.read_csv(clinical_csv_path)

        if sample_ids is None:
            sample_ids = GISTDataset.sample_ids(data_dir_seg)

        thresholds, categorical_mappings, feature_names, mask_token_index = GISTDataset.setup_clinical_preprocessing(
            df_clinical, sample_ids)

        static_pipeline = GISTDataset.static_pipeline(
            data_dir_img=data_dir_img,
            data_dir_seg=data_dir_seg,
            df_clinical=df_clinical,
            thresholds=thresholds,
            categorical_mappings=categorical_mappings,
            feature_names=feature_names,
            mask_token_index=mask_token_index,
        )

        dynamic_pipeline = GISTDataset.dynamic_pipeline(
            largest_tumor=largest_tumor,
            train=train,
            feature_names=feature_names,
            mask_token_index=mask_token_index,
            angle_range=angle_range,
            dropout_p=dropout_p
        )

        dataset = DatasetDefault(
            sample_ids=sample_ids,
            static_pipeline=static_pipeline,
            dynamic_pipeline=dynamic_pipeline,
        )
        dataset.create()
        return dataset
