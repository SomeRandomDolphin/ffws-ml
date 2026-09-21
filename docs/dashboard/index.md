# Dashboard Prediksi Tinggi Air — DAS Dhompo

## Prototipe frontend: Welang Water Monitor

Entry point `python run_dashboard.py` kini membuka **Welang Water Monitor** di
`http://localhost:8050`. Instalasi tetap `python -m pip install -e ".[dashboard]"`.
Restart server bila versi lama masih tampil.

Prototipe memakai 15 nama/koordinat stasiun yang tersedia, tetapi **seluruh muka
air, hujan, ambang, dan proyeksi adalah simulasi**. Tidak ada model ML atau CSV
historis yang dimuat. Skenario tetap dimulai 18 Februari 2026 pukul 12:00 WIB,
dengan riwayat 24 jam dan proyeksi 5 jam. Pulsa di hulu mendahului puncak Dhompo
pada +3 jam. Hujan ditampilkan dalam mm per interval 30 menit.

- Klik marker atau baris stasiun untuk menyinkronkan detail dan grafik.
- Cari nama dan filter status; detail pilihan tetap tersedia ketika daftar kosong.
- Gunakan tombol zoom/Cakupan DAS dan menu Layer peta untuk mengatur peta.
- Putar timeline untuk melihat nilai dan status berubah; putaran berhenti di +5 jam.
- Lipat grafik untuk memperbesar peta; pada ponsel, gunakan tab Daftar/Detail.

State tampilan disimpan dalam sesi browser. Basemap CARTO/OSM dan font web
memerlukan internet; data simulasi, daftar, detail, dan grafik berjalan lokal.
Geometri alur sungai dan batas DAS khusus belum disediakan. Garis antartitik
tidak digunakan untuk mengklaim lokasi sungai.

Implementasi baru: `dashboard/monitor.py`, `dashboard/demo_fixture.py`, dan
`dashboard/assets/monitor.css`. Modul analisis/model lama tetap tersedia untuk
integrasi berikutnya. Pengujian prototipe: `python -m pytest tests/test_monitor.py -q`.

## Dokumentasi dashboard model sebelumnya

Bagian berikut menjelaskan implementasi sebelumnya, bukan sumber data prototipe
frontend yang sekarang dibuka oleh `run_dashboard.py`.

Dashboard interaktif berbasis [Dash](https://dash.plotly.com) untuk memvisualisasikan
prediksi tinggi muka air di **stasiun Dhompo** pada **Sungai Welang** (Kab. Pasuruan,
Jawa Timur). Dashboard ini memakai model ML nyata dari repo (file predictor per horizon).

## Menjalankan

```powershell
python -m pip install -e ".[dashboard]"
python run_dashboard.py
```

Buka `http://localhost:8050`. Klik node pada peta untuk memilih stasiun, geser slider
atau tekan **Putar** untuk melihat simulasi prediksi hingga +5 jam.

## Struktur tampilan

Tampilan pembuka memakai peta geografis sebagai area hero, panel stasiun di
sampingnya, dan timeline tepat di bawah peta. Hubungan dua cabang hulu yang
bertemu di Dhompo ditampilkan sebagai diagram jaringan terpisah agar marker
geografis tidak disalahartikan sebagai jejak sungai. Klik marker atau pilih
stasiun untuk menyinkronkan hydrograph dan ringkasan detail.

## Visual utama

1. **Peta geografis** — marker stasiun di atas basemap Carto Positron tanpa label
   ramai. Warna marker menunjukkan status tinggi air dan marker Dhompo dibuat lebih
   besar. Sumber koordinat di `configs/dhompo/station_geo.csv`.
2. **Diagram aliran** — topologi dua cabang sungai (barat 8 stasiun, timur 4
   stasiun) yang bertemu di Dhompo, lengkap dengan arah, status, dan waktu tempuh.
3. **Aliran menuju Dhompo** — diagram topologi dua cabang dengan nilai, status,
   dan estimasi waktu tempuh. Sumbu waktu negatif dihilangkan agar urutan aliran
   terbaca tanpa interpretasi tambahan.
4. **Hydrograph stasiun terpilih** — observasi dan ambang untuk semua stasiun;
   prediksi model h1–h5 hanya muncul ketika Dhompo dipilih.

Header menampilkan ringkasan naratif status Dhompo, maksimum prediksi pada horizon
5 jam, dan jumlah titik berstatus waspada/bahaya pada frame aktif.

## Sumber data demo

Basis demo adalah `data/data-clean.csv` (2022, interval 30 menit, datum elevasi).
Window dipilih otomatis: saat pertama kali Dhompo melewati persentil `alert`,
dengan 24 baris riwayat sebelumnya sebagai input prediksi. Prediksi dijalankan oleh
`FilePredictor` (model terbaik per horizon: XGBoost h1, Gradient Boosting h2–h3,
Lasso h4–h5).

!!! note "Scaler untuk model linear"
    Model Lasso h4/h5 membutuhkan fitur terskala. `scaler.pkl` tidak tersedia di
    folder model, sehingga dashboard memakai `standard_scaler_global.pkl` yang ada.
    Ini diterapkan khusus dashboard (`dashboard/data.py`) dan tidak mengubah
    perilaku serving API.

## Threshold status

Ambang waspada/bahaya **diturunkan dari data** (persentil per stasiun) agar
menyesuaikan skala elevasi yang berbeda-beda:

| Status | Aturan |
|---|---|
| Bahaya | tinggi air ≥ persentil `danger_quantile` (default 0,99) |
| Waspada | tinggi air ≥ persentil `alert_quantile` (default 0,90) |
| Meningkat | kenaikan 3 jam ≥ `rising_delta_m` (default 0,25 m) |
| Normal | selain di atas |

Override per stasiun dapat ditetapkan di `configs/dhompo/dashboard.yaml`
(`thresholds.station_overrides`).

!!! warning "Threshold banjir 9,0 m"
    Dokumen arsitektur menyebut threshold banjir 9,0 m, tetapi pada datum elevasi
    data-clean 2022 nilai Dhompo berkisar 7–15 m dan 85% waktu berada di atas 9,0 m,
    sehingga angka itu tidak bermakna sebagai ambang bahaya pada datum ini.
    Dashboard memakai persentil data dan membuka opsi override bila threshold resmi
    tersedia.

## Kejujuran data dalam animasi

- Nilai **Dhompo** pada `+h` jam adalah **prediksi model** nyata (h1–h5).
- Nilai stasiun **hulu** pada animasi adalah **propagasi observasi** — observasi yang
  digeser menurut waktu tempuh aliran ke Dhompo (Purwodadi ≈ 3,5 jam, AWLR
  Kademungan ≈ 2 jam, Klosod ≈ 1 jam dari `docs/dhompo/ARCHITECTURE.md`; stasiun
  lain diestimasi linear dari elevasi). Ini **bukan prediksi ML**, dan diberi label
  jelas di panel detail stasiun.
- `Bd. Sentono` punya data dan koordinat tetapi **tidak dipakai model**; ditampilkan
  sebagai node sekunder (abu-abu, garis putus-putus).

## Koordinat stasiun

File `configs/dhompo/station_geo.csv` memuat koordinat dari peneliti dengan kolom
`confidence` dan `source`. Perbaiki koordinat bila tersedia data resmi yang lebih
akurat; dashboard membaca file ini saat runtime.

## Konfigurasi

Semua parameter demo di `configs/dhompo/dashboard.yaml`:

| Kunci | Default | Arti |
|---|---|---|
| `data.path` | `../../data/data-clean.csv` | basis data demo |
| `data.history_rows` | 24 | baris riwayat input prediksi |
| `data.lead_hours` | 0–5 | langkah slider |
| `thresholds.alert_quantile` | 0,90 | ambang waspada |
| `thresholds.danger_quantile` | 0,99 | ambang bahaya |
| `thresholds.rising_delta_m` | 0,25 | ambang tren naik 3 jam |
| `demo.window_end` | `null` | pin window demo ke datetime tertentu |
| `propagation.travel_hours_anchor` | 3 stasiun | anchor waktu tempuh |

## Struktur kode

```
dashboard/            paket aplikasi Dash
  data.py             muat data + prediksi model + threshold
  simulation.py       frame per langkah slider (propagasi)
  status.py           klasifikasi status
  geo.py              koordinat, topologi, layout skematik, grid elevasi
  curves.py           spline Catmull-Rom untuk alur sungai halus
  figures.py          pembangun figur Plotly (terrain, isochrone, ring gauge, band)
  layout.py           layout HTML
  callbacks.py        interaksi Dash
  assets/style.css    gaya visual (tema light elegan, Plus Jakarta Sans)
run_dashboard.py      entry point
tests/test_dashboard_*.py   pengujian
```

## Desain visual

Tema **instrumen ilmiah** memakai latar netral, tipografi Barlow Semi Condensed
untuk judul, Source Sans 3 untuk isi, dan aksen status yang hemat. Basemap
geografis memakai Carto Positron tanpa label ramai; garis antarstasiun diberi
label sebagai hubungan konseptual. Hydrograph memakai arsiran transparan untuk
zona waspada/bahaya dan garis prediksi putus-putus. Panel detail tidak lagi
menampilkan grafik kecil kosong; riwayat berada di hydrograph utama.
