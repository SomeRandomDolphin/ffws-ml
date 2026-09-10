# Skenario DAS Dhompo

Skenario ini memprediksi tinggi muka air **stasiun Dhompo dalam meter**, untuk +1 sampai +5 jam. Observasi berinterval 30 menit. Dataset mencakup 14 stasiun; baseline serving menggunakan 12 stasiun hulu dan Dhompo. Jangan menyamakan seluruh kolom dataset dengan daftar input wajib API.

## 1. Data

| Berkas di `data/` | Peran |
|---|---|
| `data-raw.xlsx` | Data sumber untuk penelusuran awal |
| `data-clean.csv` | Data bersih 2022, waktu pada kolom `Datetime` |
| `Data generated 2023.xlsx` | Data generated 2023 untuk eksperimen lintas periode dan gabungan |

Loader ada di `src/dhompo/data/loader.py`; konfigurasi sumber, target, dan split ada di `configs/dhompo/training.yaml`. Data generated tetap dibedakan dari observasi 2022 dalam interpretasi hasil.

Validasi dari root repository:

```powershell
python training/dhompo/validate_data.py
```

## 2. Urutan notebook

Buka notebook di `research/dhompo/` dan jalankan sel secara berurutan. Resolusi path mendukung kernel yang dimulai dari root repository atau folder notebook.

| Urutan | Notebook | Fokus |
|---|---|---|
| 1 | `01_eda_dhompo.ipynb` | Kualitas data, pola temporal, hubungan antarlokasi |
| 2 | `02_modeling_dhompo.ipynb` | Baseline regresi dan ensemble per horizon |
| 3 | `03_lstm_dhompo.ipynb` | Eksperimen LSTM, GRU, attention, dan TCN; membutuhkan PyTorch |
| 4 | `04_combined_training.ipynb` | Eksperimen data 2022/2023 dan fitur hujan |

Output tersimpan pada notebook adalah hasil eksekusi sebelumnya. Lokasi atau teks lama di output tersebut dipertahankan sebagai catatan historis; kode sel memakai struktur baru.

## 3. Fitur dan pembagian data

`src/dhompo/data/features.py` membangun lag, rolling statistics, perubahan tinggi air, dan fitur waktu. Konfigurasi dapat mengaktifkan travel-time lag, akumulasi hujan, interaksi, serta fitur musiman. Pilihan fitur saat inferensi harus cocok dengan fitur training model.

Pada training gabungan default, fitur dan target dibangun per segmen agar tidak menyeberangi gap antarperiode. Train memakai seluruh 2022 dan 80% awal 2023; test memakai 20% akhir 2023. Mode satu sumber memakai split temporal 80/20 tanpa shuffle.

| Eksperimen dalam notebook gabungan | Tujuan |
|---|---|
| A | Train 2022, uji generalisasi pada 2023 |
| B | Training gabungan 2022 dan bagian awal 2023 |
| C | Menguji tambahan fitur curah hujan pada training gabungan |

A/B/C adalah eksperimen Dhompo, bukan nama lokasi. Opsi `--single-source` pada skrip baseline menjalankan split satu sumber; opsi tersebut tidak identik dengan eksperimen A lintas periode.

## 4. Training dan eksperimen

Jalankan dari root repository. Perintah training membuat hasil baru dan dapat menimpa artefak dengan nama sama.

```powershell
# Baseline gabungan, logging dan registrasi model ke MLflow
python training/dhompo/train_sklearn.py --config configs/shared/sklearn_model.yaml

# Baseline satu sumber
python training/dhompo/train_sklearn.py --single-source 2022_clean

# Eksperimen fitur progresif
python training/dhompo/run_experiments.py --all

# Deep learning
python training/dhompo/train_pytorch.py --config configs/dhompo/lstm_model.yaml
```

Hyperparameter sklearn bersama ada di `configs/shared/sklearn_model.yaml`. Pengaturan LSTM ada di `configs/dhompo/lstm_model.yaml`; koneksi MLflow menggunakan konfigurasi serving dan environment yang dijelaskan pada [panduan deployment](../deployment/index.md).

| Kelompok skrip di `training/dhompo/` | Kegunaan |
|---|---|
| `tune_optuna.py`, `train_stacking.py` | Tuning dan stacking |
| `run_delta_experiment.py`, `run_smoothing_experiment.py` | Variasi target delta dan smoothing |
| `run_peak_weighted_experiment.py`, `flood_event_cv.py`, `diagnose_regime_errors.py` | Evaluasi kejadian banjir dan kesalahan menurut rezim |
| `run_tier_a_adaptive.py` | Training adaptive dengan kualitas sensor |
| `export_pred_vs_truth_combined.py` | Laporan prediksi dan observasi eksperimen A/B/C |
| `generate_experiment_tables.py`, `generate_best_per_experiment.py` | Turunan tabel eksperimen yang sudah tersedia |

Skrip pembuat tabel membutuhkan laporan sumber dari eksperimen sebelumnya. Tidak semua laporan sumber disertakan dalam clone Git.

## 5. Evaluasi dan prediksi

Metrik bersama di `training/evaluate.py`: RMSE, MAE, R², NSE, dan PBIAS. RMSE/MAE Dhompo dinyatakan dalam meter. Perbandingan model harus memakai periode uji, target, dan fitur yang sama.

- Tabel dan figur: `reports/dhompo/tables/` dan `reports/dhompo/figures/`.
- Model notebook dan eksperimen sklearn: `models/sklearn/`; varian delta/smoothing menggunakan subfolder tersendiri.
- Model notebook deep learning: `models/pytorch/`.
- Training yang menggunakan MLflow menyimpan run dan artefak melalui MLflow; lihat `mlruns/` atau tracking server yang dikonfigurasi.
- Adaptive: `artifacts/tier_a_adaptive/`, tetap di lokasi semula.

API dan kontrak request dijelaskan pada [panduan deployment Dhompo](../deployment/index.md). Model dan scaler dari run yang sama perlu tersedia untuk reproduksi prediksi.

## Referensi riset

- [Ringkasan EDA](eda/index.md)
- [Pemodelan dan rekayasa fitur](modeling/index.md)
- [Arsitektur machine learning](arsitektur_machine_learning.md)
- [Arsitektur deep learning](arsitektur_lstm_gru_tcn.md)
- [Rancangan adaptive dan pengembangan lanjutan](ARCHITECTURE.md)

Dokumen temuan dan angka evaluasi merupakan catatan eksperimen sebelumnya. Dokumen rancangan memuat fitur lanjutan yang belum seluruhnya diimplementasikan; lihat [status kode](../code-map.md).
