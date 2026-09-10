# Analisis Korelasi & Lag

## Korelasi Pearson Antar Stasiun

Matriks korelasi 14×14 menunjukkan hubungan linear antar semua pasangan stasiun. Korelasi tinggi antar stasiun yang berdekatan secara elevasi mengkonfirmasi konektivitas hidrologis DAS.

!!! info "Output"
    Matriks korelasi: `reports/dhompo/tables/xls_04_korelasi_pearson.xlsx`

## Cross-Correlation Analysis

Analisis cross-correlation dilakukan antara setiap stasiun hulu terhadap Dhompo untuk menentukan **lag optimal** — waktu yang dibutuhkan sinyal air dari hulu untuk sampai ke Dhompo.

Lag dianalisis dari 0 hingga 360 menit (12 jam) dengan step 30 menit.

!!! info "Output"
    Tabel cross-correlation: `reports/dhompo/tables/xls_05_cross_correlation.xlsx`

## Lag Teoritis vs Empiris

Perbandingan antara lag yang dihitung secara teoritis (berdasarkan jarak dan kecepatan aliran) dengan lag yang diperoleh dari cross-correlation empiris.

| Stasiun | Elevasi (m) | Lag Teoritis (step) | Lag Empiris (menit) |
|---------|-------------|---------------------|---------------------|
| Bd. Suwoto | 503 | 9 | Bervariasi |
| Krajan Timur | 335 | 8 | Bervariasi |
| ... | ... | ... | ... |
| Bd. Grinting | 28 | 1 | Bervariasi |

!!! note "Catatan"
    Nilai lag aktual tersedia di `reports/dhompo/tables/xls_06_lag_teoritis_empiris.xlsx`. Lag teoritis digunakan sebagai referensi awal dalam konstruksi fitur untuk pemodelan.
