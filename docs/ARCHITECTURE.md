# Dhompo Flood Prediction — Architecture v1

Adaptive multi-horizon (T+1..T+5 h) water-level forecasting at Dhompo with
real-time sensor-quality awareness, automated retraining, and explicit
fallback when telemetry degrades.

This document is the locked v1 plan. Sections are referenced by the
question numbers used during the design interview (Q1..Q12).

---

## 1. Target & Production Metrics (Q1)

| Item | Decision |
|------|----------|
| Predicted variable | Water level (m) at **Dhompo** station |
| Output shape | 5-step sequence at hourly cadence: `h1, h2, h3, h4, h5` (T+1..T+5 h) |
| Input cadence | 30-minute observations |
| Training loss | Peak-weighted MSE (existing `run_peak_weighted_experiment.py`) |
| Production metrics | Peak-weighted RMSE + CSI @ flood threshold 9.0 m, both reported per horizon |

---

## 2. Sensor Topology (Q2)

14 stations contribute to the basin. The diagram in
`reports/figures/diagram-alir.png` is authoritative for travel-time
relationships.

### 2.1 Telemetry availability

| Class | Stations | Real-time? |
|-------|----------|------------|
| Telemetry (green) | Purwodadi, AWLR Kademungan, Klosod, Dhompo | yes |
| AWLR offline (blue) | 11 stations including Bd. Suwoto, Krajan Timur, Bd. Sentono, Bd. Baong, Bd. Bakalan, Bd. Lecari, Bd Guyangan, Sidogiri, Bd. Domas, Bd. Grinting, Jalan Nasional | batch only |

### 2.2 Travel time to Dhompo

| Upstream telemetry | Travel time | Best horizon coverage |
|--------------------|-------------|------------------------|
| Purwodadi          | ~3.5 h      | T+3, T+4, T+5         |
| AWLR Kademungan    | ~2 h        | T+2, T+3              |
| Klosod             | ~1 h        | T+1                   |
| Dhompo (autoreg)   | 0           | floor for all horizons|

### 2.3 Hydrological clusters

- **Upstream-West:** Bd. Suwoto, Krajan Timur, Purwodadi, Bd. Sentono,
  Bd. Baong, Bd. Bakalan, AWLR Kademungan, Bd. Domas, Bd. Grinting.
- **Upstream-East:** Bd. Lecari, Bd Guyangan, Sidogiri, Klosod.
- **Local/target:** Dhompo (+ Jalan Nasional auxiliary downstream).

---

## 3. Adaptive Architecture — Two-Tier Hybrid (Q3)

### 3.1 Model tiers

- **Tier-A** (`@production/h{i}`, i ∈ 1..5): 14-station adaptive model
  with shared per-station embedding, hydrological-cluster aggregation,
  learnable masking, multi-task auxiliary heads.
- **Tier-B** (`@fallback/h{i}`, i ∈ 1..5): Dhompo-only autoregressive
  floor. Activates only when all three telemetry stations are flagged
  bad simultaneously.

Total: 10 model artifacts. Single retrain job covers all.

### 3.2 Input pipeline

1. Raw `(value, quality_flag)` per station → shared dense embedding
   (latent dim **8**, weights shared across stations).
2. Embeddings grouped by hydrological cluster; masked-mean (or attention
   pool) over surviving members per cluster.
3. Cluster embeddings concatenated with Dhompo autoregressive lags →
   horizon-specific regression head.

### 3.3 Serving contract — Option C

Tier-A accepts all 14 inputs at inference. The 11 offline stations are
forward-filled with the most recent available value. Quality flag
escalation:

- `> 30 min` since last update on a telemetry station → effectively
  `MISSING` (`STALE` is reserved for AWLR offline only).
- `> 6 h` since last update on an offline station → `STALE` auto-escalates
  to `MISSING`.

The embedding + mask layer learns to discount stale inputs proportionally.

### 3.4 Training

- **Stochastic sensor dropout:** independent Bernoulli mask, p=0.3 per
  station, applied per batch. The model sees the full permutation space
  during training.
- **Multi-task auxiliary heads:** each of the 14 stations is a secondary
  prediction target. Loss weights: main horizon × 1.0, auxiliary
  station heads × 0.1 each, summed.
- **Cluster-out objective:** training penalises Tier-A for RMSE drift
  when an entire hydrological cluster is masked out.

---

## 4. ETL Quality Detectors (Q4)

Six flag values: `OK`, `STUCK`, `FLATLINE`, `OUT_OF_RANGE`, `MISSING`,
`STALE`. Detectors run in the ETL layer **before** feature engineering;
flags are persisted as parallel columns and replayable from storage.

| Flag | Rule | Threshold |
|------|------|-----------|
| `MISSING` | row absent / NaN | exact |
| `STALE` | reading age > expected cadence | > 30 min telemetry; > 6 h AWLR offline (auto-escalates `STALE` → `MISSING` past 6 h) |
| `FLATLINE` | rolling variance over last 6 readings (3 h) | var < 1e-4 m², excluded during flood regime |
| `STUCK` | self stationary while neighbours move | \|Δself\| < 0.005 m AND median \|Δneighbours\| > 0.02 m, over 6 readings |
| `OUT_OF_RANGE` | physical envelope breach | < 0 m OR > training_max + 0.5 m OR \|Δstep\| > 1.0 m / 30 min |
| Zero-shortcut | value == 0.0 at non-zero-baseline station | immediate `FLATLINE` (per-station `min_plausible` override) |

**Hysteresis:** once `FLATLINE` or `STUCK` is set, require **3 consecutive
OK readings** before clearing back to `OK`.

---

## 5. Switching Policy & Fallback (Q5)

| Sub-decision | Rule |
|--------------|------|
| **Fallback gate** | Tier-B activates only when **all three** telemetry stations (Purwodadi, Kademungan, Klosod) carry bad flags simultaneously. Single-cluster outages stay on Tier-A with degradation flag. |
| **Cadence** | Per-request, stateless. ETL hysteresis is the only stickiness layer. |
| **Shadow execution** | When Tier-A serves, Tier-B runs in shadow on every request (logged silently). When Tier-B serves, Tier-A is **not** shadowed. |
| **Response metadata** | `serving_tier ∈ {A, B}`, `degradation: dict[str,str]`, optional `shadow_predictions`. |

### 5.1 Per-horizon primary station mapping

| Horizon | Primary station | Rationale |
|---------|------------------|-----------|
| h1 | Klosod          | ~1 h travel time |
| h2 | AWLR Kademungan | ~2 h travel time |
| h3 | AWLR Kademungan | ~2 h travel time |
| h4 | Purwodadi       | ~3.5 h travel time |
| h5 | Purwodadi       | ~3.5 h travel time |

When the per-horizon primary station carries a bad flag, the response
populates `degradation[f"h{i}"] = "PRIMARY_STATION_<FLAG>:<station>"`.

---

## 6. Synthetic Data Generation (Q6)

| Aspect | Decision |
|--------|----------|
| Scope | **Basin-coherent** — jitter independent per station; scaling and time-warping applied jointly to all stations within an event window |
| Budget | **On-the-fly** per batch, regime-conditional `p_aug`: flood = 0.7, elevated = 0.5, normal = 0.3 |
| Jitter | Gaussian additive, σ = 0.5% × per-station std (cached at training start) |
| Magnitude scaling | × ∈ U[0.9, 1.15], event-wide, asymmetric for peak emphasis |
| Time-warping | 4-knot DTW, max ±10% local stretch, monotonicity preserved, **disabled** on rising-limb/peak; applied only to recession tail and pre-event baseline |
| Validity gates | Hard rejection if any of: mass-balance violated, monotonicity around peak violated, Purwodadi↔Dhompo cross-correlation peak shifts > 30 min from empirical lag |
| Auto-tightening | If > 30% of synthetic samples rejected per epoch, σ and scaling range tightened by 20%, re-run, logged to MLflow |

---

## 7. Retraining Triggers — CI/CD for ML (Q7)

### 7.1 Triggers

| Trigger | Rule |
|---------|------|
| Schedule | Monthly, 1st of month 02:00 local |
| Event | Any flood detected at Dhompo with peak ≥ 9.0 m |
| Drift | Rolling 7-day h1 RMSE exceeds previous 30-day median × 1.5 |

Volume-based triggers are explicitly **rejected**.

### 7.2 Debounce

- Max one retrain per 24 h for event + drift triggers.
- Schedule trigger always overrides the lock.
- Failed retrains require **manual re-arm** — never auto-re-trigger.

### 7.3 Ingest contract

- `POST /ingest` — accepts batched rows in training schema.
- **Two-phase write:**
  1. Stage in `data/staging/`; ETL computes quality flags + sanity.
  2. Commit to canonical store; emit `ingest.committed` event.
- Idempotent on `(Datetime, source_id)`.
- Schema/range/timestamp violations → quarantined to `data/quarantine/`
  with reason. Never silently dropped.
- The endpoint **emits events only**. A separate orchestrator decides
  whether the cumulative state warrants a retrain.

---

## 8. Registry, Validation Gates, Rollback (Q8)

### 8.1 MLflow alias lifecycle

```
@candidate/h{i}  → newly trained, untested
@staging/h{i}    → passed offline gates, in shadow vs production
@production/h{i} → currently serving Tier-A
@fallback/h{i}   → Tier-B floor (retrained only on monthly schedule)
@archive/h{i}    → last 5 production versions retained
```

### 8.2 Offline gates (`@candidate` → `@staging`, all must pass)

1. Peak-weighted RMSE (h1) ≤ current production × 1.05.
2. Mean RMSE all horizons ≤ current production × 1.05.
3. Per-event mean \|peak_error\| ≤ 0.5 m AND max \|peak_error\| ≤ 1.0 m.
4. CSI @ 9.0 m ≥ current production × 0.95.
5. Mask-robustness: RMSE with random sensor masking (p=0.3) ≤ unmasked RMSE × 1.20.
6. Cluster-out: RMSE with one full cluster masked ≤ unmasked RMSE × 1.40.
7. Tier-B fallback regression: AR-only RMSE ≤ Tier-B current.

### 8.3 Shadow stage (`@staging` → `@production`)

- 14-day floor.
- Auto-extended (cap 60 days) until at least one flood event observed.
- Promote only if: shadow peak-weighted RMSE ≤ production × 1.02 AND
  no flood-event regression.

### 8.4 Auto-rollback

| Cause | Action |
|-------|--------|
| Live h1 RMSE > 30-day median × 2.0 sustained 6 h | Auto-revert + page |
| 503 rate > 5% over 1 h | Auto-revert + page |
| Two consecutive flood events with mean \|peak_error\| > 1.5 m | **Manual review only** |

### 8.5 Promotion authority

- **Auto-promote** for monthly schedule retrains.
- **Approval-gated** for drift- and event-triggered retrains.

---

## 9. Inference Routing (Q9)

### 9.1 Routing logic (per request, stateless)

```
BAD_SET = {FLATLINE, STUCK, OUT_OF_RANGE, MISSING}
# STALE on telemetry stations escalates to MISSING after 30 min.

telemetry_bad = {Purwodadi, AWLR Kademungan, Klosod}
                ∩ {flag ∈ BAD_SET}

if len(telemetry_bad) == 3:
    serving_tier = "B"
    served = Tier-B prediction
    shadow_predictions = None      # do NOT shadow Tier-A here
else:
    serving_tier = "A"
    served = Tier-A prediction
    shadow_predictions = Tier-B prediction (logged)

for h in 1..5:
    primary = {1: "Klosod", 2: "AWLR Kademungan", 3: "AWLR Kademungan",
               4: "Purwodadi", 5: "Purwodadi"}[h]
    if quality_flag[primary] in BAD_SET:
        degradation[f"h{h}"] = f"PRIMARY_STATION_{flag}:{primary}"

return PredictResponse(predictions=served, serving_tier, degradation,
                       shadow_predictions=...)
```

### 9.2 Placement

`TwoTierPredictor` lives in `api/predictor_state.py` and wraps a
`TierAPredictor` + `TierBPredictor`. Route handlers stay thin — request
validation and response serialisation only.

### 9.3 Loading

Eager — both tiers loaded at FastAPI startup.

### 9.4 Latency SLA

p99 < 500 ms per `/predict`. Alarm on p99 > 1 s sustained 5 min.
Component budgets: ETL ≤ 50 ms, FE ≤ 100 ms, inference ≤ 200 ms,
serialisation ≤ 50 ms.

### 9.5 Uncertainty

Point estimates only for v1.0. Conformal prediction wrapper deferred
to v1.1.

---

## 10. Observability (Q10)

### 10.1 Metrics — 14 across 4 families

**Serving health (real-time):**

- `predict.latency_ms` histogram (p50/p95/p99) tagged `serving_tier`.
- `predict.requests_total` counter tagged `serving_tier`, `status_code`.
- `predict.errors_total` counter tagged `error_class`.
- `predict.serving_tier{A|B}` ratio (Tier-B share).

**Data quality (real-time):**

- `etl.quality_flag{station, flag}` counter.
- `etl.telemetry_health` gauge (0–3 healthy telemetry stations).
- `etl.stale_age_seconds{station}` gauge.

**Prediction quality (delayed, joined to ground truth at T+h):**

- `predict.rmse{horizon}` rolling 24h/7d/30d.
- `predict.peak_error{horizon, event_id}` post-flood-event.
- `predict.csi_at_threshold{horizon}` rolling 30d.
- `predict.shadow_delta{horizon}` Tier-A vs Tier-B disagreement.

**Model lifecycle (low-frequency):**

- `model.training_runs_total` tagged `trigger_type`, `outcome`.
- `model.gate_failures_total` tagged `gate_name`.
- `model.shadow_days_remaining` gauge per `@staging` candidate.
- `model.rollbacks_total` tagged `cause`.

### 10.2 Backend

Hybrid:

- **Prometheus + Grafana** for serving + ETL families.
- **MLflow-native** for prediction-quality + lifecycle families.
- Joined in Grafana via the MLflow Postgres datasource.

### 10.3 Alert severity

- **PAGE** (24/7): auto-rollback fired; Tier-B serving sustained > 1 h;
  ingest down > 15 min.
- **TICKET** (next-business-day): drift retrain failed gates; gate
  failure rate > 50% on a single retrain; shadow auto-extended past 30
  days; p99 latency > 1 s sustained 5 min.
- **DASHBOARD-only** (no notification): single-station bad flag;
  single-cluster degraded with Tier-A still serving; schedule retrain
  promotion success.

---

## 11. Compute & Deployment Topology (Q11)

| Aspect | Decision |
|--------|----------|
| Training compute | On-prem always-on box; sklearn primary on CPU. GPU upgrade deferred until pytorch becomes primary or training time exceeds 6 h |
| Serving compute | Separate FastAPI process on the same physical box; isolation via systemd units + cgroup CPU limits (`training.slice` + `serving.slice`) |
| Containers | Not used in v1.0 |
| Hardware | 8 vCPU, 32 GB RAM, 500 GB SSD, no GPU |
| Core allocation | 4 training / 2 serving / 2 ETL+system |

---

## 12. Storage Backend (Q12)

### 12.1 Primary store

**Postgres + TimescaleDB**, single instance, separate databases:

- `dhompo_timeseries` — observations, predictions, ingest events.
- `mlflow` — registry + run metadata.

Grafana connects to the same instance for dashboards.

### 12.2 Schema

```sql
CREATE TABLE observations (
    datetime     TIMESTAMPTZ NOT NULL,
    station      TEXT NOT NULL,
    value        REAL,                    -- nullable for MISSING
    quality_flag TEXT NOT NULL,           -- enum of 6 values
    source_id    TEXT NOT NULL,           -- ingestion provenance
    ingested_at  TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (datetime, station, source_id)
);
SELECT create_hypertable('observations', 'datetime', chunk_time_interval => INTERVAL '1 day');

CREATE TABLE predictions (
    prediction_time TIMESTAMPTZ NOT NULL,
    target_time     TIMESTAMPTZ NOT NULL,
    horizon         SMALLINT    NOT NULL,
    value           REAL        NOT NULL,
    serving_tier    CHAR(1)     NOT NULL,
    shadow_value    REAL,
    model_version   TEXT        NOT NULL,
    request_id      UUID        NOT NULL,
    PRIMARY KEY (request_id, horizon)
);
SELECT create_hypertable('predictions', 'prediction_time', chunk_time_interval => INTERVAL '7 days');

CREATE TABLE ingest_events (
    event_id    UUID PRIMARY KEY,
    received_at TIMESTAMPTZ NOT NULL,
    row_count   INT NOT NULL,
    status      TEXT NOT NULL,
    reason      TEXT
);
```

`observations` is **append-only with provenance**: corrections are
inserted as new rows with later `ingested_at`; consumers read the most
recent. This preserves perfect historical replay.

### 12.3 Migration

One-time bootstrap from `data/data-clean.csv` and
`data/Data generated 2023.xlsx`:

1. Stand up Postgres + TimescaleDB.
2. Compute quality flags retrospectively (most rows → `OK`).
3. Bulk-insert into `observations` with `source_id='bootstrap'`,
   `ingested_at=NOW()`.
4. Repoint `dhompo.data.loader` to a Postgres DAL with the same public
   API. `UPSTREAM_STATIONS` and `TARGET_STATION` remain the canonical
   station-identity source of truth.
5. Frozen CSV files retained as reference snapshots; production code
   never reads them after migration.

### 12.4 Retention

| Store | Retention |
|-------|-----------|
| `observations` | Forever |
| `predictions` | 2 years hot, then archive to Parquet (cold) |
| `ingest_events` | 1 year hot, then archive to Parquet (cold) |
| MLflow artifacts | Last 5 production versions per alias (Q8a) + `@candidate` runs from past 90 days; older candidate runs purged |

---

## 13. Open Items Deferred to Follow-up Sessions

- **Auth on `/predict` and `/ingest`** — API-key + rate-limit spec.
- **Backfill / replay protocol** — re-flagging historical data when ETL
  detector thresholds change.
- **On-call runbook** — page response, escalation chain, manual-rearm
  procedure.
- **Conformal prediction (v1.1)** — uncertainty intervals.
- **GPU upgrade trigger (v1.1)** — concrete criteria for adding
  consumer GPU when pytorch Tier-A becomes primary.

---

## Appendix A — Implementation Phasing

The plan is intentionally large; implementation lands in phases.

| Phase | Scope | Key artefacts |
|-------|-------|---------------|
| **1 — Foundation** | ETL quality detectors, schema additions, `TwoTierPredictor` scaffold, hydrological clusters, routing wired in `/predict` | `src/dhompo/etl/quality.py`, `src/dhompo/data/clusters.py`, updated `api/schemas.py` + `api/predictor_state.py` |
| **2 — Tier-A retraining** | Embedding + cluster aggregation + masking + multi-task heads + stochastic dropout | `training/run_two_tier_experiment.py`, new `dhompo.models.adaptive` module |
| **3 — Synthesis pipeline** | Basin-coherent jitter/scaling/warping with validity gates | `dhompo.data.synthesis` |
| **4 — Storage + Ingest** | Postgres+TimescaleDB schema, DAL, `/ingest` endpoint with two-phase write, bootstrap migration | `dhompo.storage`, `api/routes/ingest.py` |
| **5 — Observability** | Prometheus metrics, MLflow wiring, Grafana dashboards | `dhompo.observability`, dashboards in `ops/grafana/` |
| **6 — Retraining triggers + Registry** | Schedule + event + drift triggers, MLflow alias lifecycle, gates, rollback automation | `dhompo.lifecycle`, `ops/cron/` |
