"""Composite loss for binding affinity prediction.

L_total = L_huber + lambda_rank * L_pairwise_ranking
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CompositeLoss(nn.Module):
    """Huber loss + pairwise ranking loss.

    The Huber loss provides robust regression (less sensitive to outliers).
    The pairwise ranking loss encourages correct relative ordering,
    which directly improves ranking metrics (Spearman, Kendall).
    """

    def __init__(
        self,
        lambda_rank: float = 0.1,
        huber_delta: float = 1.0,
        rank_margin: float = 0.5,
        rank_sample_size: int = 64,
    ):
        super().__init__()
        self.lambda_rank = lambda_rank
        self.rank_margin = rank_margin
        self.rank_sample_size = rank_sample_size
        self.huber = nn.SmoothL1Loss(beta=huber_delta)

    def pairwise_ranking_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Sampled pairwise margin ranking loss with adaptive margin."""
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        B = pred.shape[0]

        if B < 2:
            return torch.tensor(0.0, device=pred.device)

        n_pairs = min(self.rank_sample_size, B * (B - 1) // 2)
        idx_i = torch.randint(0, B, (n_pairs,), device=pred.device)
        idx_j = torch.randint(0, B, (n_pairs,), device=pred.device)

        # Ensure different indices
        mask_diff = idx_i != idx_j
        idx_i = idx_i[mask_diff]
        idx_j = idx_j[mask_diff]

        if idx_i.shape[0] == 0:
            return torch.tensor(0.0, device=pred.device)

        target_diff = target[idx_i] - target[idx_j]
        pred_diff = pred[idx_i] - pred[idx_j]
        sign = torch.sign(target_diff)

        # Adaptive margin scaled by affinity difference
        adaptive_margin = self.rank_margin * target_diff.abs().clamp(min=0.1)

        loss = torch.relu(adaptive_margin - sign * pred_diff)
        return loss.mean()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute composite loss.

        Args:
            pred: [B, 1] predicted pK (scaled)
            target: [B, 1] true pK (scaled)

        Returns:
            total_loss: scalar tensor
            loss_dict: {'huber': float, 'rank': float, 'total': float}
        """
        l_huber = self.huber(pred, target)
        l_rank = self.pairwise_ranking_loss(pred, target)

        total = l_huber + self.lambda_rank * l_rank

        return total, {
            "huber": l_huber.item(),
            "rank": l_rank.item(),
            "total": total.item(),
        }
