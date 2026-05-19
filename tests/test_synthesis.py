"""Tests for the basin-coherent synthetic augmenter."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dhompo.data.synthesis import (  # noqa: E402
    AugmentationConfig,
    AugmentationSchedule,
    AugmentationStats,
    SyntheticAugmenter,
    regime_indices,
)


N_STATIONS = 14
N_FEATS = 7
AR_DIM = 6


def _augmenter(rng_seed: int = 0, **config_overrides) -> SyntheticAugmenter:
    g = torch.Generator()
    g.manual_seed(rng_seed)
    station_std = torch.full((N_STATIONS,), 0.5)
    training_max = torch.full((N_STATIONS,), 4.0)
    config = AugmentationConfig(**config_overrides)
    return SyntheticAugmenter(
        station_std=station_std,
        training_max=training_max,
        schedule=AugmentationSchedule(),
        config=config,
        rng=g,
    )


def _batch(batch_size: int = 8, value: float = 1.5, target: float = 1.5):
    feats = torch.full((batch_size, N_STATIONS, N_FEATS), value)
    ar = torch.full((batch_size, AR_DIM), value)
    targets = torch.full((batch_size,), target)
    return feats, ar, targets


def test_regime_indices_partition_thresholds():
    targets = torch.tensor([0.5, 6.9, 7.0, 8.5, 9.0, 12.0])
    cfg = AugmentationConfig()
    out = regime_indices(targets, cfg)
    assert out.tolist() == [0, 0, 1, 1, 2, 2]


def test_no_augmentation_when_p_is_zero():
    aug = _augmenter()
    aug._schedule = AugmentationSchedule(normal=0.0, elevated=0.0, flood=0.0)
    feats, ar, tgt = _batch()
    f2, a2, kept, stats = aug.augment(feats, ar, tgt)
    assert torch.equal(f2, feats)
    assert torch.equal(a2, ar)
    assert kept.all()
    assert stats.n_attempted == 0
    assert stats.n_rejected == 0


def test_full_augmentation_changes_value_channel_only():
    aug = _augmenter()
    aug._schedule = AugmentationSchedule(normal=1.0, elevated=1.0, flood=1.0)
    feats, ar, tgt = _batch()
    f2, a2, kept, stats = aug.augment(feats, ar, tgt)
    # Channel 0 mutated for kept samples; other channels untouched.
    assert not torch.equal(f2[..., 0], feats[..., 0])
    assert torch.equal(f2[..., 1:], feats[..., 1:])
    assert stats.n_attempted == feats.shape[0]


def test_scaling_is_event_wide_within_one_sample():
    """Every station in a single sample must get the same multiplicative factor."""
    aug = _augmenter(jitter_sigma_pct=0.0)  # disable jitter to isolate scaling
    aug._schedule = AugmentationSchedule(normal=1.0, elevated=1.0, flood=1.0)
    feats, ar, tgt = _batch(value=2.0)
    f2, a2, kept, _ = aug.augment(feats, ar, tgt)
    for i in range(feats.shape[0]):
        if not kept[i]:
            continue
        ratios = (f2[i, :, 0] / 2.0).unique()
        # All stations share one factor → exactly one unique ratio per sample.
        assert ratios.numel() == 1


def test_value_bounds_gate_rejects_out_of_range():
    """A scale that pushes values past training_max * ratio must be rejected."""
    aug = _augmenter(jitter_sigma_pct=0.0, scale_low=10.0, scale_high=10.0,
                     max_plausible_ratio=1.5)
    aug._schedule = AugmentationSchedule(normal=1.0, elevated=1.0, flood=1.0)
    feats, ar, tgt = _batch(value=1.0)
    f2, a2, kept, stats = aug.augment(feats, ar, tgt)
    assert not kept.any()
    # Rejected samples have their original tensors restored.
    assert torch.equal(f2, feats)
    assert torch.equal(a2, ar)
    assert stats.n_rejected == feats.shape[0]
    assert stats.rejection_reasons.get("value_above_max", 0) > 0


def test_rejection_rate_helper():
    stats = AugmentationStats(n_total=10, n_attempted=5, n_rejected=2)
    assert stats.rejection_rate() == pytest.approx(0.4)


def test_rejection_rate_when_nothing_attempted():
    stats = AugmentationStats(n_total=10, n_attempted=0, n_rejected=0)
    assert stats.rejection_rate() == 0.0


def test_jitter_sigma_scales_with_station_std():
    """Larger station_std must produce a wider jitter distribution."""
    g = torch.Generator(); g.manual_seed(42)
    small_std = SyntheticAugmenter(
        station_std=torch.full((N_STATIONS,), 0.1),
        training_max=torch.full((N_STATIONS,), 100.0),
        config=AugmentationConfig(scale_low=1.0, scale_high=1.0),
        schedule=AugmentationSchedule(normal=1.0, elevated=1.0, flood=1.0),
        rng=g,
    )
    g2 = torch.Generator(); g2.manual_seed(42)
    big_std = SyntheticAugmenter(
        station_std=torch.full((N_STATIONS,), 1.0),
        training_max=torch.full((N_STATIONS,), 100.0),
        config=AugmentationConfig(scale_low=1.0, scale_high=1.0),
        schedule=AugmentationSchedule(normal=1.0, elevated=1.0, flood=1.0),
        rng=g2,
    )
    feats, ar, tgt = _batch(batch_size=256, value=5.0)
    f_small, _, _, _ = small_std.augment(feats.clone(), ar.clone(), tgt)
    f_big, _, _, _ = big_std.augment(feats.clone(), ar.clone(), tgt)
    spread_small = (f_small[..., 0] - 5.0).std().item()
    spread_big = (f_big[..., 0] - 5.0).std().item()
    assert spread_big > spread_small * 5.0  # at least 5x wider


def test_tighten_shrinks_sigma_and_scale_range():
    aug = _augmenter(jitter_sigma_pct=0.01, scale_low=0.8, scale_high=1.2)
    original = aug.config
    new = aug.tighten(factor=0.5)
    assert new.jitter_sigma_pct == pytest.approx(0.005)
    # Scale range shrinks symmetrically around its midpoint (1.0 here).
    assert new.scale_low == pytest.approx(0.9)
    assert new.scale_high == pytest.approx(1.1)
    # The augmenter's live config is now the tightened one.
    assert aug.config is new
    assert aug.config is not original


def test_flood_regime_uses_higher_p_aug_and_asymmetric_scale():
    aug = _augmenter(jitter_sigma_pct=0.0)
    aug._schedule = AugmentationSchedule(normal=0.0, elevated=0.0, flood=1.0)
    feats, ar, _ = _batch(batch_size=64, value=8.0)
    targets_flood = torch.full((64,), 9.5)
    f2, _, kept, _ = aug.augment(feats, ar, targets_flood)
    # Flood regime: every attempted sample scales by [1.0, 1.15] — never below original.
    ratios = (f2[..., 0] / 8.0).reshape(-1)
    assert (ratios >= 1.0 - 1e-6).all()
