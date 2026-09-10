# Migrasi Folder dan Perintah

Penataan ini memisahkan kode, konfigurasi, notebook, laporan, dan dokumentasi Dhompo serta Surabaya. Dataset dan artefak model lokal tidak dimasukkan ke commit baru. Nama package `dhompo`, nama modul `urban_*`, parameter model, isi data, dan kontrak API tetap dipertahankan. Perintah Python lama yang menyebut lokasi skrip sebelum pemindahan perlu diperbarui.

## Pemetaan lokasi

| Sebelum | Sesudah |
|---|---|
| `research/01_eda_dhompo.ipynb` hingga `04_combined_training.ipynb` | Nama berkas sama di `research/dhompo/` |
| `research/05_eda_surabaya.ipynb` | `research/surabaya/01_eda_surabaya.ipynb` |
| `training/train_urban_sklearn.py`, `training/report_urban_predictions.py` | Nama berkas sama di `training/surabaya/` |
| Skrip training/eksperimen Dhompo di `training/` | Nama berkas sama di `training/dhompo/` |
| Tiga skrip pembuat laporan di `scripts/` | Nama berkas sama di `training/dhompo/` |
| `configs/training.yaml`, `lstm_model.yaml`, `serving.yaml` | Nama berkas sama di `configs/dhompo/` |
| `configs/urban_water_level.yaml`, `urban_water_level_delta.yaml` | Nama berkas sama di `configs/surabaya/` |
| `configs/sklearn_model.yaml` | `configs/shared/sklearn_model.yaml` |
| Dataset Dhompo di `data/` | Tetap di lokasi lama; data baru diabaikan Git |
| Dataset Surabaya lokal | `data/surabaya/ketinggian_30menit_wide.csv`, diabaikan Git |
| `models/sklearn/`, `models/pytorch/` | Tetap menjadi lokasi model Dhompo |
| Model Surabaya lokal | `models/surabaya/urban/` atau `models/surabaya/urban_delta/`, diabaikan Git |
| Laporan Dhompo/Surabaya di `reports/tables/` | `reports/dhompo/tables/` atau `reports/surabaya/tables/` sesuai skenario |
| `docs/eda/`, `docs/modeling/`, dokumen arsitektur | `docs/dhompo/` dengan nama/subfolder yang sama |
| Dokumen integrasi adaptive di `docs/` | `docs/deployment/` |

`training/evaluate.py` dan `research/eda_helpers.py` tetap menjadi helper bersama. Dokumen Word/RTF lokal, artefak adaptive, riwayat MLflow, dan laporan yang asalnya belum jelas tetap di lokasi semula.

## Contoh perintah baru

```powershell
python training/dhompo/train_sklearn.py --config configs/shared/sklearn_model.yaml --train-config configs/dhompo/training.yaml
python training/dhompo/train_pytorch.py --config configs/dhompo/lstm_model.yaml
python training/dhompo/run_tier_a_adaptive.py --epochs 5
python training/surabaya/train_urban_sklearn.py --config configs/surabaya/urban_water_level.yaml --dry-run
python training/surabaya/report_urban_predictions.py --model-dir models/surabaya/urban
```

Perintah API tetap `uvicorn api.main:app`. Target `make train-sklearn` tetap tersedia dan mengarah ke Dhompo; `make train-dhompo`, `make train-surabaya`, dan `make check-surabaya` memberi pilihan eksplisit.

## Artefak dan reproduksi

Metadata model Surabaya lama dapat menyimpan alamat absolut dari komputer pembuatnya. Jika alamat tersebut tidak ada, predictor dan pembuat laporan mencari nama file yang sama dalam direktori model terpilih. Training baru menyimpan nama file relatif sehingga folder model dapat dipindahkan bersama scaler dan metadata.

Dataset dan model lokal tidak ditambahkan ke commit atau push. Output notebook lama dipertahankan; hanya path pada kode dan referensi teks yang diperbarui. Perubahan struktur tidak menghasilkan angka evaluasi atau model baru.
