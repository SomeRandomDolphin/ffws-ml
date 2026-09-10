# Statistik Deskriptif

## Ringkasan Statistik

Data muka air dari 14 stasiun dianalisis menggunakan ukuran tendensi sentral, dispersi, dan bentuk distribusi.

Metrik yang dihitung untuk setiap stasiun:

| Metrik | Keterangan |
|--------|-----------|
| Mean | Rata-rata muka air |
| Std | Standar deviasi |
| Min / Max | Nilai minimum dan maksimum |
| Range | Selisih max - min |
| Skewness | Kemiringan distribusi |
| Kurtosis | Keruncingan distribusi |
| CV | Koefisien variasi (std/mean) |

!!! info "Output"
    Tabel lengkap tersedia di `reports/dhompo/tables/xls_01_statistik_deskriptif.xlsx`

## Deteksi Outlier

Dua metode deteksi outlier digunakan secara paralel:

1. **IQR (Interquartile Range)** — outlier didefinisikan sebagai nilai di luar rentang Q1 - 1.5×IQR hingga Q3 + 1.5×IQR
2. **Z-Score** — outlier didefinisikan sebagai nilai dengan |z| > 3

!!! info "Output"
    Tabel deteksi outlier: `reports/dhompo/tables/xls_02_deteksi_outlier.xlsx`

## Uji Stasioneritas (ADF)

Augmented Dickey-Fuller test diterapkan pada setiap stasiun untuk menguji apakah data bersifat stasioner.

- **H0:** Data memiliki unit root (non-stasioner)
- **H1:** Data stasioner

Hasil: Semua stasiun menolak H0 pada tingkat signifikansi 5% (p-value < 0.05), mengindikasikan data stasioner dan siap digunakan untuk pemodelan tanpa differencing.

!!! info "Output"
    Tabel uji ADF: `reports/dhompo/tables/xls_03_uji_adf.xlsx`
