# Exploratory Data Analysis

Analisis eksploratif dilakukan terhadap data muka air dari 14 stasiun pemantauan di DAS Dhompo, dengan interval pencatatan **30 menit** selama periode **Oktober–Desember 2022** (67 hari, ~3.216 observasi per stasiun).

## Tujuan EDA

1. Memahami distribusi dan karakteristik statistik setiap stasiun
2. Mengidentifikasi outlier dan stasioneritas data
3. Menganalisis korelasi spasial antar stasiun
4. Menentukan lag propagasi banjir dari hulu ke hilir
5. Mengidentifikasi pola temporal (diurnal, mingguan)

## Ringkasan Temuan

### Distribusi Data

- Sebagian besar stasiun menunjukkan distribusi **right-skewed** (positif), mengindikasikan kejadian banjir sebagai event langka dengan nilai ekstrem
- Stasiun hilir (Dhompo, Jalan Nasional) memiliki variabilitas paling tinggi karena merupakan akumulasi aliran dari seluruh DAS

### Stasioneritas

- Uji Augmented Dickey-Fuller (ADF) menunjukkan **semua stasiun stasioner** pada tingkat signifikansi 5%
- Data tidak memerlukan differencing sebelum pemodelan

### Korelasi Spasial

- Korelasi Pearson antar stasiun sangat tinggi (r > 0.85 untuk stasiun yang berdekatan)
- Korelasi menurun seiring jarak elevasi, mengkonfirmasi hubungan kausal aliran hulu-hilir

### Pola Temporal

- Terdapat siklus diurnal 24 jam yang konsisten (terdeteksi dari dekomposisi musiman)
- Komponen residu menangkap event banjir non-periodik

## Detail Analisis

- [Statistik Deskriptif](statistik-deskriptif.md)
- [Analisis Korelasi & Lag](korelasi-lag.md)
- [Dekomposisi & Pola Temporal](dekomposisi-temporal.md)
