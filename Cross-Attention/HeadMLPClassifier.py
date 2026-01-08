import torch
from torch import nn
from fuse.utils import NDict
from typing import Sequence


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, N, D]
        attn_weights = torch.softmax(self.attn(x), dim=1)  # [B, N, 1]
        pooled = torch.sum(attn_weights * x, dim=1)        # [B, D]
        return pooled, attn_weights.squeeze(-1)            # also return attention weights: [B, N]


class HeadMLPClassifier(nn.Module):
    def __init__(
        self,
        input_key: str,
        prob_key: str,
        logits_key: str,
        in_ch: int,
        num_classes: int,
        layers_description: Sequence[int] = (256,),
        dropout_rate: float = 0.1,
        pooling: str = 'avg',  # 'avg', 'max', or 'attention'
    ):
        super().__init__()
        self.input_key = input_key
        self.prob_key = prob_key
        self.logits_key = logits_key
        self.pooling_type = pooling

        if pooling == 'avg':
            self.pool = lambda x: (torch.mean(x, dim=1), None)
        elif pooling == 'max':
            self.pool = lambda x: (torch.max(x, dim=1).values, None)
        elif pooling == 'attention':
            self.attn_pool = AttentionPooling(in_ch)
            self.pool = self.attn_pool
        else:
            raise ValueError("Invalid pooling type. Choose 'avg', 'max', or 'attention'.")

        layer_list = []
        last_dim = in_ch
        for dim in layers_description:
            layer_list.append(nn.Linear(last_dim, dim))
            layer_list.append(nn.ReLU())
            if dropout_rate > 0:
                layer_list.append(nn.Dropout(dropout_rate))
            last_dim = dim
        layer_list.append(nn.Linear(last_dim, num_classes))

        self.classifier = nn.Sequential(*layer_list)

    def forward(self, batch_dict: NDict) -> NDict:
        x = batch_dict[self.input_key]  # [B, N, D]
        pooled, attn_weights = self.pool(x)  # pooled: [B, D], attn_weights: [B, N] or None

        logits = self.classifier(pooled)  # [B, 1] for binary, [B, C] for multi-class
        batch_dict[self.logits_key] = logits

        if logits.shape[1] == 1:
            # Binary classification case
            prob = torch.sigmoid(logits)  # [B, 1]
            prob_scalar = prob.squeeze(-1)  # [B] — needed for thresholding, confusion, AUC
            batch_dict[self.prob_key] = prob_scalar  # e.g., model.prob.TKI_Classification

            # For BSS metric — requires 2-class probs: [B, 2]
            batch_dict['model.prob.TKI_Classification.bss'] = torch.stack([1 - prob_scalar, prob_scalar], dim=1)

        else:
            # Multi-class case
            prob = torch.softmax(logits, dim=1)  # [B, C]
            batch_dict[self.prob_key] = prob
            batch_dict['model.prob.TKI_Classification.bss'] = prob

        # Optional: save attention weights for inspection
        if self.pooling_type == 'attention':
            batch_dict['model.attn.pooling_weights'] = attn_weights  # [B, N]

        return batch_dict
