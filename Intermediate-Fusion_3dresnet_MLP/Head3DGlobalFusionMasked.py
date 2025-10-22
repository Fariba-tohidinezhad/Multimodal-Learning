"""
Head3DGlobalFusionMasked
- True 3D global pooling (avg, max, or avg+max)
- Optional *masked* pooling over tumor voxels (and optional validity mask)
- Clinical (tabular) MLP branch -> concat -> ClassifierMLP (instead of 1x1x1 Conv3D)
- Outputs logits [B,K] and probs [B,K] for classification, or [B,dim] for regression
- Adds explainability tensors saved under "model.explain.*" keys

Author: Fariba (modified to include explainability and vector-based classifier)
"""

from collections.abc import Sequence
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fuse.utils.ndict import NDict
from fuse.dl.models.heads.common import ClassifierMLP


class Head3DGlobalFusionMasked(nn.Module):
    def __init__(
        self,
        *,
        head_name: str = "head_0",
        mode: str = "classification",                 # "classification" | "regression"
        num_outputs: int = 2,                         # classes (classification) or dims (regression)
        # ---- imaging feature maps ----
        conv_inputs: Sequence[Tuple[str, int]],
        pooling: str = "avg",                         # "avg" | "max" | "avgmax"
        spatial_dropout_rate: float = 0.0,            # Dropout3d on feature maps BEFORE pooling
        layers_description: Sequence[int] = (256,),   # MLP hidden layers for classifier
        fused_dropout_rate: float = 0.1,              # dropout inside classifier MLP
        # ---- clinical branch ----
        append_features: Sequence[Tuple[str, int]] | None = None,
        append_layers_description: Sequence[int] = (128, 64),
        append_dropout_rate: float = 0.2,
        # ---- masked pooling ----
        use_mask: bool = False,
        mask_key: Optional[str] = None,               # batch_dict key for segmentation mask
        valid_mask_key: Optional[str] = None,         # optional validity mask (exclude padding, etc.)
        mask_resize_mode: str = "nearest",            # "nearest" | "trilinear"
        mask_threshold: float = 0.5,                  # used if trilinear
        empty_mask_fallback: str = "global_avg",      # "global_avg" | "global_max" | "skip"
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert mode in ("classification", "regression")
        assert pooling in ("avg", "max", "avgmax")
        assert conv_inputs is not None and len(conv_inputs) > 0, "conv_inputs must be provided"

        # ----- store
        self.head_name = head_name
        self.mode = mode
        self.num_outputs = num_outputs
        self.pooling = pooling
        self.conv_inputs = conv_inputs

        # ----- spatial dropout before pooling
        self.spatial_do = nn.Dropout3d(spatial_dropout_rate) if spatial_dropout_rate > 0 else nn.Identity()

        # ----- compute imaging channel count
        self.imaging_ch = sum(ch for _, ch in conv_inputs)
        pooled_imaging_ch = self.imaging_ch * (2 if pooling == "avgmax" else 1)

        # ----- clinical branch (optional)
        self.append_features = append_features
        if append_features is not None and len(append_features) > 0:
            tab_in = sum(ch for _, ch in append_features)
            if len(append_layers_description) == 0:
                self.tabular_module = nn.Identity()
                tab_out = tab_in
            else:
                self.tabular_module = ClassifierMLP(
                    in_ch=tab_in,
                    num_classes=None,
                    layers_description=append_layers_description,
                    dropout_rate=append_dropout_rate,
                )
                tab_out = append_layers_description[-1]
        else:
            self.tabular_module = None
            tab_out = 0

        fused_in_ch = pooled_imaging_ch + tab_out

        # ----- MLP classifier head (replaces ClassifierFCN3D)
        self.classifier_mlp = ClassifierMLP(
            in_ch=fused_in_ch,
            num_classes=num_outputs if mode == "classification" else None,
            layers_description=layers_description,
            dropout_rate=fused_dropout_rate,
        )

        # ----- masked pooling config
        self.use_mask = use_mask
        self.mask_key = mask_key
        self.valid_mask_key = valid_mask_key
        assert not use_mask or (mask_key is not None), "use_mask=True requires mask_key"
        assert mask_resize_mode in ("nearest", "trilinear")
        self.mask_resize_mode = mask_resize_mode
        self.mask_threshold = mask_threshold
        self.empty_mask_fallback = empty_mask_fallback
        self.eps = eps

    # ----------------- helpers -----------------

    @staticmethod
    def _lift_to_5d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(2)
        return x

    def _cat_conv_inputs(self, batch_dict: NDict) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        for key, _ in self.conv_inputs:
            x = batch_dict[key]
            x = self._lift_to_5d(x)
            assert x.dim() == 5, f"{key} must be 5D [B,C,D,H,W]. Got {tuple(x.shape)}"
            feats.append(x)
        return torch.cat(feats, dim=1)

    @staticmethod
    def _resize_mask(mask: torch.Tensor, target: torch.Size, mode: str, thr: float) -> torch.Tensor:
        d, h, w = target[-3:]
        if mode == "nearest":
            m = F.interpolate(mask.float(), size=(d, h, w), mode="nearest")
            return (m > 0.5).float()
        else:
            m = F.interpolate(mask.float(), size=(d, h, w), mode="trilinear", align_corners=False)
            return (m > thr).float()

    def _masked_global_avg(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        num = (feat * mask).sum(dim=(2, 3, 4))
        den = mask.sum(dim=(2, 3, 4)).clamp_min(self.eps)
        return num / den

    def _masked_global_max(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        very_neg = torch.finfo(feat.dtype).min
        masked = torch.where(mask.bool(), feat, torch.full_like(feat, very_neg))
        return masked.amax(dim=(2, 3, 4))

    def _global_pool_unmasked(self, feat: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == "avg":
            out = F.adaptive_avg_pool3d(feat, 1)
        elif kind == "max":
            out = F.adaptive_max_pool3d(feat, 1)
        else:
            raise ValueError(f"Unsupported pooling kind: {kind}")
        return out.view(out.size(0), out.size(1))

    def _pooled_imaging(self, feat: torch.Tensor, mask_resized: Optional[torch.Tensor]) -> torch.Tensor:
        feat = self.spatial_do(feat)
        if self.use_mask and mask_resized is not None:
            mask_sum = mask_resized.sum(dim=(2, 3, 4))
            empty = (mask_sum <= self.eps).squeeze(1)

            if self.pooling == "avg":
                pooled = self._masked_global_avg(feat, mask_resized)
                if empty.any():
                    pooled[empty] = self._global_pool_unmasked(feat[empty], "avg")
                return pooled
            elif self.pooling == "max":
                pooled = self._masked_global_max(feat, mask_resized)
                if empty.any():
                    pooled[empty] = self._global_pool_unmasked(feat[empty], "max")
                return pooled
            else:
                avg = self._masked_global_avg(feat, mask_resized)
                max_ = self._masked_global_max(feat, mask_resized)
                if empty.any():
                    avg[empty] = self._global_pool_unmasked(feat[empty], "avg")
                    max_[empty] = self._global_pool_unmasked(feat[empty], "max")
                return torch.cat([avg, max_], dim=1)

        # unmasked pooling
        if self.pooling == "avg":
            return self._global_pool_unmasked(feat, "avg")
        elif self.pooling == "max":
            return self._global_pool_unmasked(feat, "max")
        else:
            a = self._global_pool_unmasked(feat, "avg")
            m = self._global_pool_unmasked(feat, "max")
            return torch.cat([a, m], dim=1)

    def _tabular_branch(self, batch_dict: NDict) -> Optional[torch.Tensor]:
        if not self.append_features:
            return None
        tabs = []
        for key, _ in self.append_features:
            value = batch_dict[key]
            if isinstance(value, (dict, NDict)):
                keys_sorted = sorted(value.keys())
                tensors = [value[k] for k in keys_sorted]
                value = torch.cat(tensors, dim=1 if tensors[0].dim() == 2 else 0)
            tabs.append(value)
        tab = torch.cat(tabs, dim=1)
        return self.tabular_module(tab) if self.tabular_module else tab

    # ----------------- forward -----------------

    def forward(self, batch_dict: NDict) -> Dict:
        # --- 1. imaging feature extraction
        feat = self._cat_conv_inputs(batch_dict)  # [B,C,D,H,W]
        batch_dict["model.explain.feat_img"] = feat

        if torch.is_grad_enabled() and feat.requires_grad:
            feat.retain_grad()

        # --- 2. prepare mask (optional)
        mask_resized = None
        if self.use_mask:
            mask = self._lift_to_5d(batch_dict[self.mask_key])
            if self.valid_mask_key and self.valid_mask_key in batch_dict:
                vmask = self._lift_to_5d(batch_dict[self.valid_mask_key])
                mask = (mask > 0.5).float() * (vmask > 0.5).float()
            mask_resized = self._resize_mask(mask, feat.shape, self.mask_resize_mode, self.mask_threshold)
            batch_dict["model.explain.mask_resized"] = mask_resized

        # --- 3. masked/unmasked pooling
        pooled_img = self._pooled_imaging(feat, mask_resized)  # [B,C_pool]
        batch_dict["model.explain.pooled_img"] = pooled_img

        # --- 4. clinical tabular branch
        tab_feat = self._tabular_branch(batch_dict)  # [B,H_tab] or None
        if tab_feat is not None:
            batch_dict["model.explain.tabular_feat"] = tab_feat

        # --- 5. fuse and classify
        fused = torch.cat([pooled_img, tab_feat], dim=1) if tab_feat is not None else pooled_img
        batch_dict["model.explain.fused_feat"] = fused

        logits = self.classifier_mlp(fused)  # [B, num_outputs] or [B, dim]

        # --- 6. outputs
        if self.mode == "regression":
            batch_dict[f"model.prob.{self.head_name}"] = logits
            return batch_dict

        if self.num_outputs == 1:
            probs = torch.sigmoid(logits).squeeze(1)
        else:
            probs = F.softmax(logits, dim=1)

        batch_dict[f"model.logits.{self.head_name}"] = logits
        batch_dict[f"model.prob.{self.head_name}"] = probs

        return batch_dict
