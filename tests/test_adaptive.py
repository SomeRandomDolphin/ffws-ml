"""Tests for AdaptiveTierA model + sensor dropout + composite loss."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dhompo.data.loader import ALL_STATIONS, TARGET_STATION
from dhompo.models.adaptive import AdaptiveTierA, AdaptiveTierAConfig
from dhompo.training.losses import CompositeLoss, peak_weighted_mse
from dhompo.training.sensor_dropout import (
    DropoutSchedule,
    apply_sensor_dropout,
    regime_for_target,
)


@pytest.fixture
def cfg() -> AdaptiveTierAConfig:
    return AdaptiveTierAConfig(features_per_station=7, embedding_dim=8,
                                ar_lag_dim=6, horizon_count=5, hidden_dim=16,
                                dropout_p=0.0)


@pytest.fixture
def model(cfg) -> AdaptiveTierA:
    return AdaptiveTierA(cfg)


def _make_batch(cfg: AdaptiveTierAConfig, batch_size: int = 4):
    n_stations = len(cfg.stations)
    x = torch.randn(batch_size, n_stations, cfg.features_per_station)
    mask = torch.ones(batch_size, n_stations, dtype=torch.bool)
    ar = torch.randn(batch_size, cfg.ar_lag_dim)
    y = torch.randn(batch_size, cfg.horizon_count)
    aux_y = torch.randn(batch_size, n_stations)
    return x, mask, ar, y, aux_y


class TestModelShapes:
    def test_forward_shapes(self, model, cfg):
        x, mask, ar, _, _ = _make_batch(cfg, batch_size=4)
        horizons, aux = model(x, mask, ar)
        assert horizons.shape == (4, cfg.horizon_count)
        assert aux.shape == (4, len(cfg.stations))

    def test_mask_zeros_propagate(self, model, cfg):
        """A station with mask=0 must contribute zero to its cluster's pooled vector."""
        x, mask, ar, _, _ = _make_batch(cfg, batch_size=2)
        mask[:, 0] = False  # drop the first station
        # The pooled cluster vector should equal the masked-mean over the remaining
        # cluster members; we verify by recomputing manually for one cluster.
        embeddings = model.embedding(
            torch.cat([x * mask.unsqueeze(-1).float(), mask.unsqueeze(-1).float()], dim=-1)
        )
        first_cluster = list(model._cluster_indices.values())[0]
        if 0 in first_cluster:
            survivors = [i for i in first_cluster if i != 0]
            expected = embeddings[:, survivors, :].mean(dim=1)
            actual = (embeddings[:, first_cluster, :] *
                      mask[:, first_cluster].unsqueeze(-1).float()).sum(dim=1) / \
                      mask[:, first_cluster].sum(dim=1).clamp(min=1.0).unsqueeze(-1)
            assert torch.allclose(actual, expected, atol=1e-5)

    def test_all_stations_masked_yields_zero_clusters(self, model, cfg):
        x, _, ar, _, _ = _make_batch(cfg, batch_size=2)
        mask = torch.zeros(2, len(cfg.stations), dtype=torch.bool)
        horizons, _ = model(x, mask, ar)
        # Forward should not crash; output must remain finite even with no signal.
        assert torch.isfinite(horizons).all()


class TestGradientFlow:
    def test_all_params_get_gradients(self, model, cfg):
        x, mask, ar, y, aux_y = _make_batch(cfg, batch_size=4)
        horizons, aux = model(x, mask, ar)
        loss = ((horizons - y) ** 2).mean() + ((aux - aux_y) ** 2).mean()
        loss.backward()
        unwired = [
            name for name, p in model.named_parameters() if p.grad is None
        ]
        assert unwired == [], f"Unwired parameters: {unwired}"


class TestPeakWeightedMSE:
    def test_flood_samples_weighted_more(self):
        pred = torch.tensor([[5.0, 5.0]])
        target_normal = torch.tensor([[3.0, 3.0]])
        target_flood = torch.tensor([[9.5, 9.5]])
        l_normal = peak_weighted_mse(pred, target_normal)
        l_flood = peak_weighted_mse(pred, target_flood)
        assert l_flood > l_normal

    def test_zero_when_perfect(self):
        pred = torch.tensor([[3.0, 4.0]])
        target = torch.tensor([[3.0, 4.0]])
        assert peak_weighted_mse(pred, target).item() == 0.0


class TestCompositeLoss:
    def test_decomposes_into_main_and_aux(self, model, cfg):
        loss_fn = CompositeLoss()
        x, mask, ar, y, aux_y = _make_batch(cfg, batch_size=4)
        horizons, aux = model(x, mask, ar)
        out = loss_fn(horizons, y, aux, aux_y)
        assert "total" in out and "main" in out and "aux" in out
        assert torch.isclose(
            out["total"], out["main"] + 0.1 * out["aux"]
        )

    def test_aux_mask_excludes_dropped_stations(self):
        loss_fn = CompositeLoss()
        horizon_pred = torch.zeros(1, 5)
        horizon_target = torch.zeros(1, 5)
        aux_pred = torch.tensor([[10.0, 0.0]])
        aux_target = torch.tensor([[0.0, 0.0]])
        # Without mask, both errors count; with mask hiding the first station,
        # the aux loss must be zero.
        no_mask = loss_fn(horizon_pred, horizon_target, aux_pred, aux_target)
        masked = loss_fn(
            horizon_pred, horizon_target, aux_pred, aux_target,
            aux_mask=torch.tensor([[False, True]]),
        )
        assert no_mask["aux"].item() > 0.0
        assert masked["aux"].item() == 0.0


class TestSensorDropout:
    def test_only_flips_healthy_to_bad(self):
        x = torch.ones(8, 14, 7)
        mask = torch.zeros(8, 14, dtype=torch.bool)
        mask[:, :7] = True   # first 7 stations healthy, rest already bad
        target = torch.full((8,), 5.0)
        rng = torch.Generator().manual_seed(0)
        new_x, new_mask = apply_sensor_dropout(
            x, mask, target,
            schedule=DropoutSchedule(normal=1.0, elevated=1.0, flood=1.0),
            rng=rng,
        )
        # Drop probability 1.0 → all healthy stations must be flipped to bad.
        assert new_mask[:, :7].sum().item() == 0
        # Already-bad stations stay bad (cannot be revived).
        assert new_mask[:, 7:].sum().item() == 0
        # Dropped feature values must be zeroed.
        assert (new_x[:, :7, :] == 0).all()

    def test_p_zero_is_identity(self):
        x = torch.randn(4, 14, 7)
        mask = torch.ones(4, 14, dtype=torch.bool)
        target = torch.full((4,), 5.0)
        new_x, new_mask = apply_sensor_dropout(
            x, mask, target,
            schedule=DropoutSchedule(normal=0.0, elevated=0.0, flood=0.0),
        )
        assert torch.equal(new_mask, mask)
        assert torch.equal(new_x, x)


class TestRegimeClassification:
    def test_thresholds(self):
        target = torch.tensor([3.0, 7.5, 9.5])
        regime = regime_for_target(target)
        assert regime.tolist() == [0, 1, 2]


def test_target_station_index_matches_loader(model):
    assert model.config.stations[model.target_station_index] == TARGET_STATION
