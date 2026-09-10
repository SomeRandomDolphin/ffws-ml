# Evaluasi & Hasil

## Metrik Evaluasi

| Metrik | Formula | Interpretasi |
|--------|---------|-------------|
| **NSE** | $1 - \frac{\sum(y_{obs} - y_{pred})^2}{\sum(y_{obs} - \bar{y}_{obs})^2}$ | 1.0 = sempurna, <0 = lebih buruk dari rata-rata |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_{obs} - y_{pred})^2}$ | Error dalam satuan meter, sensitif terhadap outlier |
| **MAE** | $\frac{1}{n}\sum|y_{obs} - y_{pred}|$ | Error rata-rata absolut dalam meter |
| **PBIAS** | $\frac{\sum(y_{obs} - y_{pred})}{\sum y_{obs}} \times 100$ | Bias sistematik (%), positif = underestimate |
| **R2** | Koefisien determinasi | Proporsi variansi yang dijelaskan model |

## Performa Seluruh Model

### Horizon h1 (+1 Jam)

| Algoritma | NSE | RMSE (m) | MAE (m) |
|-----------|-----|----------|---------|
| XGBoost | **0.9897** | **0.1242** | **0.0746** |
| Gradient Boosting | 0.9878 | 0.1352 | 0.0793 |
| Random Forest | 0.9834 | 0.1576 | 0.0899 |
| Lasso | 0.9699 | 0.2126 | 0.1193 |
| Ridge | 0.9693 | 0.2146 | 0.1200 |
| Linear Regression | 0.9693 | 0.2147 | 0.1200 |

### Degradasi NSE per Horizon

```
h1: ████████████████████████████████████████ 0.990 (Excellent)
h2: ██████████████████████████████████████   0.983 (Excellent)
h3: ████████████████████████████████████     0.956 (Very Good)
h4: ██████████████████████████████           0.889 (Good)
h5: ████████████████████████                 0.771 (Satisfactory)
```

!!! warning "Zona Operasional"
    - **h1–h3 (NSE > 0.95):** Reliable untuk keputusan taktis evakuasi
    - **h4 (NSE ~0.89):** Cukup untuk peringatan dini awal
    - **h5 (NSE ~0.77):** Hanya untuk estimasi kasar, perlu dikombinasikan dengan judgement ahli

## Diagnostik Residual

Analisis residual pada model terbaik menunjukkan:

1. **Distribusi residual mendekati normal** — model tidak memiliki bias sistematik yang signifikan
2. **Tidak ada autokorelasi residual yang kuat** — model telah mengekstrak sebagian besar informasi temporal
3. **Heteroskedastisitas ringan** — error cenderung lebih besar pada nilai muka air tinggi (event banjir), yang merupakan tantangan umum dalam pemodelan hidrologi

## Feature Importance

Analisis feature importance dari model Gradient Boosting menunjukkan pergeseran prediktor seiring bertambahnya horizon:

- **Horizon pendek (h1–h2):** Fitur dari Dhompo sendiri (`Dhompo_t0`, `Dhompo_lag1`) mendominasi — model mengandalkan autokorelasi lokal
- **Horizon jauh (h4–h5):** Fitur dari stasiun hulu (`Bd. Suwoto`, `Krajan Timur`) menjadi lebih penting — model beralih ke sinyal propagasi banjir dari hulu

Fenomena ini konsisten dengan hukum fisika perjalanan gelombang banjir di sungai.

## Output

!!! info "File Output"
    - Metrik lengkap: `reports/dhompo/tables/xls_11_model_comparison_final.xlsx`
    - Prediksi vs ground truth: `reports/dhompo/tables/xls_12_prediksi_vs_ground_truth.xlsx`
