"""Adaptive Tier-A model (ARCHITECTURE.md §3).

Architecture
------------
1. Per-station shared dense embedding ``(features_per_station,) → latent_dim``.
   Weights are shared across all 14 stations so the network learns a generic
   "what does this station look like right now" representation.

2. Hydrological cluster aggregation. Stations are grouped into three clusters
   (upstream-west / upstream-east / local). Within each cluster, surviving
   stations contribute via a masked mean of their embeddings. If every station
   in a cluster is masked out, the cluster vector is zero — downstream layers
   learn to handle that.

3. Concatenated cluster vectors + Dhompo autoregressive lag features feed five
   horizon-specific regression heads (h1..h5) and fourteen auxiliary station
   heads (one per station, multi-task supervision).

The model accepts an explicit binary ``mask`` tensor so the ETL-derived quality
flags propagate directly into the forward pass — no implicit assumption that
zero values mean missing data.
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

from dhompo.data.clusters import CLUSTERS
from dhompo.data.loader import ALL_STATIONS, TARGET_STATION


@dataclass(frozen=True)
class AdaptiveTierAConfig:
    features_per_station: int = 7
    embedding_dim: int = 8
    ar_lag_dim: int = 6
    horizon_count: int = 5
    hidden_dim: int = 64
    dropout_p: float = 0.1
    stations: tuple[str, ...] = tuple(ALL_STATIONS)
    cluster_order: tuple[str, ...] = ("upstream_west", "upstream_east", "local")


def _cluster_index_map(
    stations: tuple[str, ...], cluster_order: tuple[str, ...]
) -> dict[str, list[int]]:
    """Map cluster name → ordered list of station indices in that cluster."""
    out: dict[str, list[int]] = {name: [] for name in cluster_order}
    for idx, station in enumerate(stations):
        for name in cluster_order:
            if station in CLUSTERS[name]:
                out[name].append(idx)
                break
    return out


class StationEmbedding(nn.Module):
    """Shared dense embedding applied independently to every station."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_stations, in_features) → (batch, n_stations, out_features)
        return self.net(x)


class ClusterAggregator(nn.Module):
    """Masked-mean aggregation per hydrological cluster.

    Empty clusters (all stations masked) yield a zero vector, which downstream
    layers learn to interpret as "no signal from this cluster".
    """

    def __init__(self, cluster_indices: dict[str, list[int]]) -> None:
        super().__init__()
        self._cluster_indices = cluster_indices
        self._cluster_order = list(cluster_indices.keys())

    def forward(
        self, embeddings: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # embeddings: (batch, n_stations, embed_dim)
        # mask:       (batch, n_stations) with 1 = healthy, 0 = bad/missing
        cluster_vecs = []
        for name in self._cluster_order:
            idx = self._cluster_indices[name]
            if not idx:
                cluster_vecs.append(torch.zeros(
                    embeddings.shape[0], embeddings.shape[-1],
                    device=embeddings.device, dtype=embeddings.dtype,
                ))
                continue
            cluster_emb = embeddings[:, idx, :]                 # (B, |C|, E)
            cluster_mask = mask[:, idx].unsqueeze(-1).float()   # (B, |C|, 1)
            weighted = cluster_emb * cluster_mask                # zero out bad
            denom = cluster_mask.sum(dim=1).clamp(min=1.0)       # avoid /0
            pooled = weighted.sum(dim=1) / denom                 # (B, E)
            cluster_vecs.append(pooled)
        return torch.cat(cluster_vecs, dim=-1)                   # (B, n_clusters * E)


class AdaptiveTierA(nn.Module):
    """Tier-A: 14-station adaptive model with multi-task auxiliary heads.

    Inputs
    ------
    station_features : (batch, n_stations, features_per_station)
        Per-station feature vector. Contents are caller-defined; the existing
        feature engineering (t0, lags, rolling stats) is reshaped to a
        (n_stations, k) matrix per timestep.
    mask : (batch, n_stations)
        Binary mask, 1 = station healthy, 0 = bad/missing. Drives both the
        cluster aggregation and the explicit "availability flag" learned by
        the embedding (mask is appended to station_features inside forward).
    ar_lags : (batch, ar_lag_dim)
        Dhompo autoregressive lag features (e.g. last 6 readings of Dhompo).

    Outputs
    -------
    horizons : (batch, horizon_count) — predictions for h1..h5.
    aux : (batch, n_stations) — auxiliary station-level predictions
        (multi-task supervision).
    """

    def __init__(self, config: AdaptiveTierAConfig | None = None) -> None:
        super().__init__()
        self.config = config or AdaptiveTierAConfig()
        cfg = self.config
        n_stations = len(cfg.stations)

        # +1 because the mask bit is concatenated to each station's features
        self.embedding = StationEmbedding(
            in_features=cfg.features_per_station + 1,
            out_features=cfg.embedding_dim,
        )

        cluster_indices = _cluster_index_map(cfg.stations, cfg.cluster_order)
        self._cluster_indices = cluster_indices
        self.aggregator = ClusterAggregator(cluster_indices)

        merged_dim = (
            cfg.embedding_dim * len(cfg.cluster_order) + cfg.ar_lag_dim
        )
        self.trunk = nn.Sequential(
            nn.Linear(merged_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout_p),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout_p),
        )

        self.horizon_heads = nn.ModuleList(
            [nn.Linear(cfg.hidden_dim, 1) for _ in range(cfg.horizon_count)]
        )
        self.aux_heads = nn.ModuleList(
            [nn.Linear(cfg.embedding_dim, 1) for _ in range(n_stations)]
        )

    @property
    def target_station_index(self) -> int:
        return self.config.stations.index(TARGET_STATION)

    def forward(
        self,
        station_features: torch.Tensor,
        mask: torch.Tensor,
        ar_lags: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if station_features.dim() != 3:
            raise ValueError(
                f"station_features must be 3-D (batch, n_stations, F); "
                f"got shape {tuple(station_features.shape)}."
            )
        mask_f = mask.float().unsqueeze(-1)                       # (B, S, 1)
        x = torch.cat([station_features, mask_f], dim=-1)         # (B, S, F+1)
        x = x * mask_f.expand_as(x[..., :-1]).pad if False else x  # noqa
        # Zero out feature contribution from masked stations (mask bit stays 1).
        feature_part = x[..., :-1] * mask_f
        x = torch.cat([feature_part, mask_f], dim=-1)

        embeddings = self.embedding(x)                            # (B, S, E)
        cluster_vec = self.aggregator(embeddings, mask)           # (B, n_c*E)

        merged = torch.cat([cluster_vec, ar_lags], dim=-1)        # (B, merged)
        trunk = self.trunk(merged)                                # (B, H)

        horizons = torch.cat(
            [head(trunk) for head in self.horizon_heads], dim=-1  # (B, h)
        )

        aux = torch.cat(
            [head(embeddings[:, i, :]) for i, head in enumerate(self.aux_heads)],
            dim=-1,
        )                                                         # (B, S)
        return horizons, aux
