# =========================
# PT_MAE_Dataset.py
# =========================
import os
from typing import Optional, Sequence, Tuple, Dict

from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.utils.ndict import NDict
from fuse.data.ops.op_base import OpReversibleBase
from fuse.data.ops.ops_common import OpKeepKeypaths

from PT_MAE_DataUtils import (
    OpLoadImage,
    OpResample,
    OpClipHU,
    OpAbdominalRandomCrop,
    OpCTPatchifyWithMask,
    OpMinMaxNormalize,
)


class OpMAESampleIDDecode(OpReversibleBase):
    """Derive file paths from sample_id for MAE pretraining."""
    def __call__(self, sample_dict: NDict, op_id: Optional[str] = None) -> NDict:
        sid = str(sample_dict["data.sample_id"])
        sample_dict["data.input.img_path"] = f"{sid}.nii.gz"
        sample_dict["data.input.body_seg_path"] = f"{sid}_body.nii.gz"
        sample_dict["data.input.liver_seg_path"] = f"{sid}_liver.nii.gz"
        return sample_dict


class GISTMAEPretrainDataset:
    """
    Notes on center coding in sample_id:
      - sample_id format: "<center_code>_<patient_number>" e.g., "101_1000"
      - center_code mapping:
          101 -> AVL
          102 -> EMC
          103 -> LUMC
          104 -> RUMC
    """

    CENTER_CODE_TO_NAME: Dict[str, str] = {
        "101": "AVL",
        "102": "EMC",
        "103": "LUMC",
        "104": "RUMC",
    }

    @staticmethod
    def get_center_code(sample_id: str) -> str:
        sid = str(sample_id)
        if "_" not in sid:
            return "UNKNOWN"
        return sid.split("_", 1)[0]

    @classmethod
    def get_center_name(cls, sample_id: str) -> str:
        code = cls.get_center_code(sample_id)
        return cls.CENTER_CODE_TO_NAME.get(code, code)

    @staticmethod
    def sample_ids(data_dir_img: str) -> Sequence[str]:
        return [
            f.replace(".nii.gz", "")
            for f in os.listdir(data_dir_img)
            if f.endswith(".nii.gz")
        ]

    @staticmethod
    def static_pipeline(
        data_dir_img: str,
        data_dir_seg: str,
        target_spacing_zyx: Tuple[float, float, float] = (5.0, 0.7636719, 0.7636719),
        clip_range: Tuple[float, float] = (-150.0, 250.0),
    ) -> PipelineDefault:
        """static: decode paths + load CT/body/liver + resample + clip"""
        return PipelineDefault(
            "static",
            [
                (OpMAESampleIDDecode(), {}),

                # load
                (OpLoadImage(dir_path=data_dir_img, is_mask=False),
                 dict(key_in="data.input.img_path", key_out="data.input.img")),
                (OpLoadImage(dir_path=data_dir_seg, is_mask=True),
                 dict(key_in="data.input.body_seg_path", key_out="data.input.body")),
                (OpLoadImage(dir_path=data_dir_seg, is_mask=True),
                 dict(key_in="data.input.liver_seg_path", key_out="data.input.liver")),

                # resample
                (OpResample(*target_spacing_zyx), {}),

                # clip
                (OpClipHU(clip_range=clip_range),
                 dict(key_in="data.input.img.resampled", key_out="data.input.img.resampled.clipped")),
            ],
        )

    @staticmethod
    def dynamic_pipeline(
        crop_size_zyx: Tuple[int, int, int] = (64, 256, 256),
        patch_size_zyx: Tuple[int, int, int] = (16, 64, 64),
        min_body_frac: float = 0.30,
        store_debug: bool = False,
    ) -> PipelineDefault:
        """dynamic: abdominal random crop → min-max norm → patchify(valid mask) → keep keys"""
        steps = []

        # crop from resampled+clipped CT and resampled masks
        steps.append(
            (
                OpAbdominalRandomCrop(
                    crop_size_zyx=crop_size_zyx,
                    store_debug=store_debug,
                ),
                dict(
                    key_in=("data.input.img.resampled.clipped", "data.input.body.resampled", "data.input.liver.resampled"),
                    key_out=("data.input.img.crop", "data.input.body.crop", "data.input.liver.crop"),
                ),
            )
        )

        # normalize crop to [0,1]
        steps.append(
            (
                OpMinMaxNormalize(min_val=-150.0, max_val=250.0),
                dict(key_in="data.input.img.crop", key_out="data.input.img.crop.norm"),
            )
        )

        # patchify normalized crop
        steps.append(
            (
                OpCTPatchifyWithMask(
                    patch_size_zyx=patch_size_zyx,
                    min_body_frac=min_body_frac,
                    use_liver_frac=False,
                    pad_to_multiple=False,
                    pad_val=None,
                    mask_pad_threshold=1.0,
                    store_debug=store_debug,
                ),
                dict(
                    key_in=("data.input.img.crop.norm", "data.input.body.crop", "data.input.liver.crop"),
                    key_out=("data.input.img.patches", "model.embed_mask_b", "data.input.grid"),
                ),
            )
        )

        keep_keys = [
            "data.sample_id",
            "data.input.img.patches",   # torch [N, pz*py*px] (or normalized values)
            "model.embed_mask_b",       # torch [N] bool
            "data.input.grid",          # np [gz,gy,gx]
        ]
        steps.append((OpKeepKeypaths(), {"keep_keypaths": keep_keys}))

        return PipelineDefault("dynamic", steps)

    @staticmethod
    def dataset(
        data_dir_img: str,
        data_dir_seg: str,
        sample_ids: Optional[Sequence[str]] = None,
        target_spacing_zyx: Tuple[float, float, float] = (5.0, 0.7636719, 0.7636719),
        clip_range: Tuple[float, float] = (-150.0, 250.0),
        crop_size_zyx: Tuple[int, int, int] = (64, 256, 256),
        patch_size_zyx: Tuple[int, int, int] = (16, 64, 64),
        min_body_frac: float = 0.30,
        store_debug: bool = False,
    ) -> DatasetDefault:

        if sample_ids is None:
            sample_ids = GISTMAEPretrainDataset.sample_ids(data_dir_img)

        static_pl = GISTMAEPretrainDataset.static_pipeline(
            data_dir_img=data_dir_img,
            data_dir_seg=data_dir_seg,
            target_spacing_zyx=target_spacing_zyx,
            clip_range=clip_range,
        )

        dynamic_pl = GISTMAEPretrainDataset.dynamic_pipeline(
            crop_size_zyx=crop_size_zyx,
            patch_size_zyx=patch_size_zyx,
            min_body_frac=min_body_frac,
            store_debug=store_debug,
        )

        ds = DatasetDefault(
            sample_ids=list(sample_ids),
            static_pipeline=static_pl,
            dynamic_pipeline=dynamic_pl,
        )
        ds.create()
        return ds
