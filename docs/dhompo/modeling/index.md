# Arsitektur Model

## Strategi Prediksi: Direct Multi-Horizon Forecasting

Sistem ini menggunakan pendekatan **Direct Forecasting** — setiap horizon prediksi (h1–h5) memiliki model independen. Berbeda dengan pendekatan recursive yang memprediksi satu langkah lalu menggunakan hasil prediksi sebagai input berikutnya, direct forecasting menghindari akumulasi error.

```
Input (t=0)  →  Model h1  →  Prediksi t+1 jam
             →  Model h2  →  Prediksi t+2 jam
             →  Model h3  →  Prediksi t+3 jam
             →  Model h4  →  Prediksi t+4 jam
             →  Model h5  →  Prediksi t+5 jam
```

## Algoritma yang Dievaluasi

| Algoritma | Tipe | Scaling | Karakteristik |
|-----------|------|---------|---------------|
| Linear Regression | Linear | Ya | Baseline, asumsi linearitas |
| Ridge | Linear (L2) | Ya | Regularisasi L2 untuk mengurangi overfitting |
| Lasso | Linear (L1) | Ya | Regularisasi L1, otomatis seleksi fitur |
| Random Forest | Ensemble | Tidak | Bagging, robust terhadap outlier |
| Gradient Boosting | Ensemble | Tidak | Boosting, performa tinggi pada pola non-linear |
| XGBoost | Ensemble | Tidak | Optimasi gradient boosting, regularisasi bawaan |

## Hasil: Model Terbaik per Horizon

| Horizon | Lead-Time | Model Terbaik | NSE | RMSE (m) | MAE (m) |
|---------|-----------|---------------|-----|----------|---------|
| h1 | +1 Jam | XGBoost | 0.9897 | 0.1242 | 0.0746 |
| h2 | +2 Jam | Gradient Boosting | 0.9825 | 0.1618 | 0.0969 |
| h3 | +3 Jam | Gradient Boosting | 0.9563 | 0.2562 | 0.1240 |
| h4 | +4 Jam | Lasso | 0.8894 | 0.4081 | 0.1654 |
| h5 | +5 Jam | Lasso | 0.7713 | 0.5877 | 0.2236 |

!!! note "Pergeseran Arsitektur"
    Pada horizon pendek (h1–h3), model ensemble (tree-based) mendominasi karena mampu menangkap non-linearitas lokal. Pada horizon jauh (h4–h5), model linear teregulasi (Lasso) lebih stabil karena menghindari overfitting pada sinyal yang sudah melemah.

## Detail

- [Rekayasa Fitur](rekayasa-fitur.md) — konstruksi 160 fitur prediktif
- [Evaluasi & Hasil](evaluasi-hasil.md) — metrik, visualisasi, dan diagnostik
