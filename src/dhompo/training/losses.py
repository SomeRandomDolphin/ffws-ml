"""Composite loss for Tier-A: peak-weighted main + auxiliary station heads.

ARCHITECTURE.md §3.4 specifies main loss × 1.0 plus 0.1 per auxiliary station
head, summed. The main loss reuses the W1_moderate sample-weighting scheme
from ``training/run_peak_weighted_experiment.py`` so the new training pipeline
inherits the diagnosed flood-peak fix.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError(
        "Install via: pip install -r requirements-torch.txt"
    ) from exc


@dataclass(frozen=True)
class LossConfig:
    aux_weight: float = 0.1
    elevated_threshold: float = 7.0
    flood_threshold: float = 9.0
    elevated_weight: float = 3.0
    flood_weight: float = 10.0


def peak_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    config: LossConfig | None = None,
) -> torch.Tensor:
    """Peak-weighted MSE over a (batch, horizons) prediction tensor.

    The W1_moderate scheme from the existing peak-weighted experiment:
    weight = 1 normally, 3 in the elevated band, 10 inside the flood band.
    """
    if config is None:
        config = LossConfig()
    sq = (pred - target) ** 2
    w = torch.ones_like(target)
    w = torch.where(target >= config.elevated_threshold,
                    torch.full_like(w, config.elevated_weight), w)
    w = torch.where(target >= config.flood_threshold,
                    torch.full_like(w, config.flood_weight), w)
    return (w * sq).mean()


class CompositeLoss(nn.Module):
    """Sum of peak-weighted main loss and unweighted auxiliary station loss."""

    def __init__(self, config: LossConfig | None = None) -> None:
        super().__init__()
        self.config = config or LossConfig()

    def forward(
        self,
        horizon_pred: torch.Tensor,
        horizon_target: torch.Tensor,
        aux_pred: torch.Tensor,
        aux_target: torch.Tensor,
        aux_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        main = peak_weighted_mse(horizon_pred, horizon_target, self.config)

        aux_sq = (aux_pred - aux_target) ** 2
        if aux_mask is not None:
            aux_mask_f = aux_mask.float()
            denom = aux_mask_f.sum().clamp(min=1.0)
            aux = (aux_sq * aux_mask_f).sum() / denom
        else:
            aux = aux_sq.mean()

        total = main + self.config.aux_weight * aux
        return {"total": total, "main": main, "aux": aux}
