# Hasil Fase 1 — Baseline Multi-Stasiun +6 Jam

## Konfigurasi

- Target: seluruh 15 stasiun pada `configs/dhompo/network.yaml`.
- Horizon: +1, +2, +3, +4, +5, +6 jam (interval data 30 menit).
- Fitur: nilai t0, lag 1–3, rolling mean/std 3/6/12 jam, perubahan level,
  dan fitur waktu untuk seluruh 15 stasiun (184 fitur).
- Data: 6.546 sampel valid dari dua segmen yang dibangun terpisah.
- Split: 80/20 temporal **di dalam setiap segmen**, dengan purge gap 6 jam
  sebelum test; tidak ada lag atau target yang melintasi gap 26 hari.
- Model: persistence, Ridge, dan multi-output HistGradientBoosting.
- Catatan: semua angka mengukur kesesuaian terhadap data acuan generated,
  bukan akurasi terhadap observasi lapangan.

## Hasil agregat test

| Horizon | Persistence NSE | Ridge NSE | HistGB NSE | HistGB RMSE (m) | HistGB KGE |
|---:|---:|---:|---:|---:|---:|
| +1 jam | 0.6624 | 0.8095 | **0.8345** | 0.3226 | 0.8727 |
| +2 jam | 0.5955 | 0.7640 | **0.7848** | 0.3635 | 0.8402 |
| +3 jam | 0.4908 | 0.7169 | **0.7387** | 0.3939 | 0.8040 |
| +4 jam | 0.3816 | 0.6562 | **0.6731** | 0.4376 | 0.7556 |
| +5 jam | 0.2892 | 0.5861 | **0.6125** | 0.4721 | 0.6963 |
| +6 jam | 0.1755 | 0.5143 | **0.5495** | 0.5071 | 0.6436 |

HistGradientBoosting menjadi baseline terbaik pada semua horizon. Pada +6 jam,
RMSE makro turun sekitar 27,6% dari persistence (0,7000 → 0,5071 m), sedangkan
NSE naik 0,3740. Performa tetap turun terhadap horizon, seperti yang diharapkan.

## Dhompo test

| Horizon | Persistence NSE | HistGB NSE | Persistence RMSE | HistGB RMSE |
|---:|---:|---:|---:|---:|
| +1 jam | 0.9258 | **0.9782** | 0.2924 | **0.1583** |
| +2 jam | 0.8478 | **0.9721** | 0.4185 | **0.1791** |
| +3 jam | 0.7311 | **0.9522** | 0.5562 | **0.2344** |
| +4 jam | 0.5973 | **0.9095** | 0.6802 | **0.3225** |
| +5 jam | 0.4570 | **0.8283** | 0.7895 | **0.4439** |
| +6 jam | 0.3154 | **0.7228** | 0.8860 | **0.5637** |

## Diagnostik stasiun

- `Bd. Lecari` hampir sempurna pada semua horizon (NSE sekitar 0,9997–0,9999),
  sehingga perlu diperiksa kemungkinan pola generated yang terlalu deterministik.
- `Bd. Baong` adalah bottleneck: NSE HistGradientBoosting sekitar 0,10 pada +1
  jam dan -0,024 pada +4 jam. Hal ini konsisten dengan audit Fase 0 yang
  menemukan autokorelasi lag-1 sangat rendah.
- Angka agregat makro harus selalu dibaca bersama hasil per stasiun; satu nilai
  agregat dapat menyembunyikan perbedaan kualitas data yang besar.

## Artefak

- Metrik lengkap: `reports/dhompo/tables/fase1_multistation_metrics.csv`.
- Ringkasan test: `reports/dhompo/tables/fase1_multistation_summary.csv`.
- Model lokal: `models/sklearn/multistation/` (diabaikan Git).
- Konfigurasi: `configs/dhompo/multistation_training.yaml`.

## Reproduksi

```powershell
python training/dhompo/train_multistation.py --dry-run
python training/dhompo/train_multistation.py
python -m pytest -q
```

## Keputusan ke Fase 2

HistGradientBoosting menjadi baseline acuan. Simulator hujan/routing pada Fase 2
harus dievaluasi terhadap persistence dan baseline ini, khususnya pada +6 jam
dan pada `Bd. Baong` sebagai kasus sulit.
