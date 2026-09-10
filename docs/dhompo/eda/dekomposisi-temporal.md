# Dekomposisi & Pola Temporal

## Pola Diurnal (24 Jam)

Analisis rata-rata muka air per jam menunjukkan siklus diurnal yang konsisten di semua stasiun. Pola ini mengindikasikan pengaruh evapotranspirasi, aktivitas irigasi, dan siklus alami hidrologi.

!!! info "Output"
    Tabel pola diurnal: `reports/dhompo/tables/xls_07_pola_diurnal.xlsx`

## Rolling Statistics

Statistik bergerak (rolling mean dan rolling standard deviation) dihitung untuk beberapa jendela waktu guna mengidentifikasi tren dan volatilitas:

| Window | Step | Fungsi |
|--------|------|--------|
| 3 jam  | 6    | Menangkap fluktuasi jangka pendek |
| 12 jam | 24   | Menangkap tren setengah harian |
| 24 jam | 48   | Menangkap tren harian |
| 48 jam | 96   | Menangkap tren multi-hari |

- **Rolling mean** menunjukkan tren kenaikan muka air secara gradual menjelang puncak musim hujan
- **Rolling std** meningkat tajam saat event banjir, mengindikasikan transisi dari kondisi stabil ke fluktuatif

!!! info "Output"
    Tabel rolling statistics: `reports/dhompo/tables/xls_08_rolling_statistics.xlsx`

## Dekomposisi Musiman (Seasonal Decomposition)

Time series Dhompo didekomposisi menggunakan model **aditif** dengan period=48 (siklus 24 jam pada data 30 menit):

$$X(t) = T(t) + S(t) + R(t)$$

| Komponen | Interpretasi Hidrologis |
|----------|------------------------|
| **Trend** | Evolusi kejenuhan DAS (catchment wetting) — naik seiring akumulasi hujan |
| **Seasonal** | Osilasi periodik 24 jam — dipengaruhi siklus diurnal |
| **Residual** | Event banjir non-periodik dan noise — sinyal yang tidak tertangkap oleh tren dan musiman |

!!! info "Output"
    Tabel dekomposisi: `reports/dhompo/tables/xls_09_dekomposisi_musiman.xlsx`

## ACF & PACF

Analisis Autocorrelation Function (ACF) dan Partial Autocorrelation Function (PACF) digunakan untuk mengidentifikasi struktur autokorelasi dalam data:

- **ACF** menunjukkan korelasi yang menurun perlahan (slow decay), mengkonfirmasi keberadaan tren
- **PACF** menunjukkan cutoff setelah lag awal, mengindikasikan proses autoregresif orde rendah

!!! info "Output"
    Tabel ACF/PACF: `reports/dhompo/tables/xls_10_acf_pacf.xlsx`
