# Audit Data — Fase 0

Status: selesai. Dokumen ini merangkum sifat data acuan, temuan kualitas, dan
keputusan definisi jaringan 15 stasiun untuk simulator hibrida.

## 1. Sumber data

| Berkas | Isi | Rentang | Baris |
|---|---|---|---|
| `data/data-clean.csv` | 15 stasiun, tanpa curah hujan | 2022-10-01 08:00 → 2022-12-05 15:00 | 3.135 |
| `data/Data generated 2023.xlsx` | 15 stasiun + `Curah hujan` | 2023-01-01 00:00 → 2023-03-14 12:00 | 3.481 |
| `data/data-raw.xlsx` | Ekspor mentah, kolom `Time` + `SP (meters)` per stasiun | sama dengan 2022 | — |

Interval 30 menit. Ada **gap ~26 hari** (2022-12-05 → 2023-01-01). Secara
operasional data diperlakukan sebagai **dua segmen** (`2022_clean`,
`2023_generated`) dan lag/rolling tidak boleh melintasi gap.

## 2. Provenance

- Berkas mentah mengekspor pasangan kolom `Time` + **`SP (meters)`** per
  stasiun. Penamaan "SP" (kemungkinan *Set Point* / profil keluaran model)
  mengindikasikan data adalah **keluaran perangkat lunak/model**, bukan
  observasi lapangan. **Arti `SP` masih perlu dikonfirmasi** ke pembimbing.
- 2022 tidak punya kolom curah hujan, 2023 punya. Indikasi kedua berkas
  tidak dihasilkan oleh satu proses yang identik.
- Konsekuensi: tidak ada *ground truth* observasi. Klaim hasil dibatasi pada
  konsistensi terhadap data acuan dan validasi internal antar-skenario.

## 3. Temuan kualitas

1. **Hujan–level 2023 tidak terhubung.** Korelasi `Curah hujan` vs level
   Dhompo hanya **0.01–0.04** pada lag 0–24 jam. Simulasi hidrologi yang
   sehat semestinya memperlihatkan respons nyata. Level dan hujan tampak
   dibangkitkan terpisah.
2. **Sebagian stasiun nyaris datar / noisy.** Fraksi selisih nol per langkah:
   `Bd. Domas` 49%, `Bd. Sentono` 24%, `Sidogiri` 23%, `Bd. Lecari` 14%.
   Autokorelasi lag-1 `Bd. Baong` hanya ~0.17 (nyaris white noise), sementara
   `Bd. Suwoto` ~0.99 (sangat halus). Campuran ini janggal untuk jaringan
   sungai yang koheren.
3. **2022 vs 2023 berbeda distribusi.** Uji KS menolak kesamaan distribusi
   untuk semua 15 stasiun (p ≈ 0). Bisa karena musim, bisa karena proses
   pembangkitan berbeda.
4. **Curah hujan 2023 jarang dan ekstrem.** 6.9% baris > 0, mean 0.46 mm per
   30 menit, maksimum 35 mm per 30 menit.

## 4. Keputusan yang diambil

1. **15 stasiun aktif.** `Bd. Sentono` (elevasi datum 680.536 m) ditambahkan
   ke `STATION_META` di `src/dhompo/data/loader.py`, sehingga
   `ALL_STATIONS` berisi 15 stasiun. Baseline model Dhompo tetap memakai
   `UPSTREAM_STATIONS` (12 hulu) seperti sebelumnya.
2. **Topologi eksplisit.** Graf 15 node + 14 reach didefinisikan di
   `configs/dhompo/network.yaml`, konsisten dengan `clusters.py` dan
   `dashboard/geo.py`: dua cabang (barat, timur) bertemu di Dhompo,
   `Bd. Sentono` bergabung ke `Purwodadi`, `Dhompo → Jalan Nasional`.
3. **Gold diperbarui.** `make gold` menghasilkan `data/gold/hydro.csv`
   (15 kolom stasiun) dan `data/gold/stations.csv` (15 baris, `Bd. Sentono`
   berperan `aux`).
4. **Data acuan bukan target absolut.** Dipakai untuk kalibrasi parameter
   simulator dan pembanding evaluasi, bukan sebagai klaim akurasi nyata.

## 5. Reproduksi

```powershell
# validasi graf jaringan
python -m pytest tests/test_network.py -q

# regenerasi gold 15 stasiun
make gold

# seluruh tes
python -m pytest -q
```

## 6. Tindak lanjut

- Konfirmasi ke pembimbing: software sumber, arti `SP (meters)`, apakah 2022
  dan 2023 dihasilkan dengan setup yang sama, dan ketersediaan observasi nyata.
- Fase 1: baseline multi-stasiun +6 jam.
- Fase 2: generator hujan + simulator routing pada graf `network.yaml`.
