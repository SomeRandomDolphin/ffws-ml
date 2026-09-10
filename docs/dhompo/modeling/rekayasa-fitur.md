# Rekayasa Fitur

## Overview

Setiap timestep direpresentasikan oleh **160 fitur** yang dibangun dari 13 stasiun (12 upstream + Dhompo) ditambah 4 fitur temporal.

## Kategori Fitur

### 1. Nilai Saat Ini & Lag (t0, t-1, t-2, t-3)

Nilai muka air pada waktu saat ini dan 3 timestep sebelumnya (hingga 90 menit ke belakang).

```
Stasiun_t0    → nilai saat ini
Stasiun_lag1  → 30 menit lalu
Stasiun_lag2  → 60 menit lalu
Stasiun_lag3  → 90 menit lalu
```

Total: 13 stasiun × 4 = **52 fitur**

### 2. Rolling Statistics (Mean & Std)

Rata-rata bergerak dan standar deviasi bergerak untuk tiga jendela waktu:

| Window | Step | Menangkap |
|--------|------|-----------|
| 3 jam  | 6    | Fluktuasi jangka pendek |
| 6 jam  | 12   | Tren setengah harian |
| 12 jam | 24   | Tren harian |

- **Rolling mean** — kondisi rata-rata air dalam jendela waktu tersebut
- **Rolling std** — seberapa fluktuatif air dalam jendela waktu tersebut

Total: 13 stasiun × 6 (3 window × 2 statistik) = **78 fitur**

### 3. Rate of Change (diff1, diff2)

Selisih nilai muka air antar timestep, menunjukkan kecepatan perubahan:

- **diff1** = selisih dengan 1 step sebelumnya (30 menit)
- **diff2** = selisih dengan 2 step sebelumnya (60 menit)

Nilai positif menunjukkan air naik, negatif menunjukkan air turun. Nilai besar menandakan perubahan drastis.

Total: 13 stasiun × 2 = **26 fitur**

### 4. Fitur Temporal

| Fitur | Keterangan |
|-------|-----------|
| `hour_sin` | Encoding siklikal jam (sin) |
| `hour_cos` | Encoding siklikal jam (cos) |
| `dayofweek` | Hari dalam minggu (0–6) |
| `is_night` | Flag malam hari (jam 19–06 = 1) |

Encoding siklikal digunakan agar model memahami bahwa jam 23 dan jam 0 itu berdekatan (bukan berjauhan secara numerik).

Total: **4 fitur**

## Ringkasan

| Kategori | Jumlah Fitur |
|----------|-------------|
| Nilai saat ini & lag | 52 |
| Rolling statistics | 78 |
| Rate of change | 26 |
| Temporal | 4 |
| **Total** | **160** |

## Pre-processing

- **StandardScaler** diterapkan hanya untuk model linear (Linear Regression, Ridge, Lasso) karena sensitif terhadap skala fitur
- Model tree-based (Random Forest, Gradient Boosting, XGBoost) menggunakan fitur tanpa scaling

## Kebutuhan Data Minimum

Karena rolling window terbesar adalah 24 step (12 jam), dibutuhkan minimal **24 baris data** (12 jam terakhir) agar semua fitur terisi tanpa NaN.
