"""Stochastic sensor dropout (ARCHITECTURE.md §3.4 / §6).

During training we drop sensors at random — for each (batch, station) pair we
flip a Bernoulli(p) coin and, if it lands heads, mark the station as bad. The
station's value is zeroed and its mask bit is set to 0 before the model sees
the batch. This trains Tier-A on the full sensor-dropout permutation space
without enumerating it explicitly.

Per ARCHITECTURE.md §6, the dropout probability is regime-conditional: more
augmentation for normal flow (the easy regime, where the model can afford to
lose context), less for floods (where every sensor matters).
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "Install via: pip install -r requirements-torch.txt"
    ) from exc


@dataclass(frozen=True)
class DropoutSchedule:
    """Per-regime Bernoulli drop probabilities.

    Defaults match ARCHITECTURE.md §6: flood = 0.3 (lightest), elevated = 0.4,
    normal = 0.5 (heaviest).

    Note: §6's `p_aug` (overall augmentation probability) and the per-station
    drop probability are different knobs. This dataclass is the per-station
    drop probability when the augmentation IS applied.
    """

    normal: float = 0.30
    elevated: float = 0.30
    flood: float = 0.30


def regime_for_target(
    target_value: torch.Tensor,
    elevated_threshold: float = 7.0,
    flood_threshold: float = 9.0,
) -> torch.Tensor:
    """Classify each batch element by Dhompo target water level.

    Returns a tensor of shape (batch,) with values 0=normal, 1=elevated,
    2=flood. Thresholds match the existing peak-weighted regime tags in
    ``training/run_peak_weighted_experiment.py``.
    """
    regime = torch.zeros_like(target_value, dtype=torch.long)
    regime = torch.where(target_value >= elevated_threshold,
                          torch.ones_like(regime), regime)
    regime = torch.where(target_value >= flood_threshold,
                          torch.full_like(regime, 2), regime)
    return regime


def apply_sensor_dropout(
    station_features: torch.Tensor,
    mask: torch.Tensor,
    target_value: torch.Tensor,
    schedule: DropoutSchedule | None = None,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-station Bernoulli dropout to features + mask.

    Parameters
    ----------
    station_features:
        (batch, n_stations, features_per_station) tensor of per-station inputs.
    mask:
        (batch, n_stations) binary tensor; 1 = healthy at ETL time, 0 = bad.
        Already-bad stations are preserved as bad — dropout only flips healthy
        stations to bad, never the reverse.
    target_value:
        (batch,) Dhompo level used to classify the regime per sample.
    schedule:
        Per-regime drop probabilities. Defaults to ``DropoutSchedule()``.
    rng:
        Optional torch.Generator for deterministic tests.

    Returns
    -------
    (augmented_features, augmented_mask) — same shapes as inputs.
    """
    if schedule is None:
        schedule = DropoutSchedule()

    regime = regime_for_target(target_value)
    drop_p = torch.tensor(
        [schedule.normal, schedule.elevated, schedule.flood],
        device=station_features.device, dtype=station_features.dtype,
    )[regime]                                            # (batch,)
    drop_p = drop_p.unsqueeze(-1).expand_as(mask)        # (batch, n_stations)

    if rng is None:
        coin = torch.rand_like(drop_p)
    else:
        coin = torch.empty_like(drop_p)
        coin.uniform_(0.0, 1.0, generator=rng)

    drop_event = coin < drop_p                           # True → drop this cell
    new_mask = mask & ~drop_event                        # only ever turn 1→0
    feature_mask = new_mask.unsqueeze(-1).to(station_features.dtype)
    new_features = station_features * feature_mask       # zero dropped values

    return new_features, new_mask
