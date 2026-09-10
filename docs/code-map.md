# Peta Kode dan Status Implementasi

Nama package Python tetap `dhompo`. Implementasi Surabaya dapat dikenali melalui nama `urban`; folder bahan riset menggunakan nama geografis `surabaya`.

| Tahap | Dhompo | Surabaya |
|---|---|---|
| Konfigurasi | `configs/dhompo/` | `configs/surabaya/` |
| Loader | `src/dhompo/data/loader.py` | `src/dhompo/data/urban_loader.py` |
| Fitur | `src/dhompo/data/features.py` | `src/dhompo/data/urban_features.py` |
| Training | `training/dhompo/` | `training/surabaya/` |
| Prediksi lokal | `src/dhompo/serving/file_predictor.py` | `src/dhompo/serving/urban_file_predictor.py` |
| Notebook | `research/dhompo/` | `research/surabaya/` |
| Hasil | `models/sklearn/`, `models/pytorch/`, `reports/dhompo/` | `models/surabaya/`, `reports/surabaya/` |

## Bagian bersama

- `src/dhompo/config.py`: root repository, pembacaan YAML, dan resolusi path. Nilai path di dalam YAML relatif terhadap lokasi YAML; argumen YAML relatif terhadap root repository.
- `src/dhompo/models/sklearn_models.py` dan `configs/shared/sklearn_model.yaml`: definisi model sklearn dan hyperparameter.
- `training/evaluate.py`: metrik evaluasi bersama.
- `research/eda_helpers.py`: helper penyimpanan figur; nama file tanpa folder diarahkan ke figur Dhompo. Berikan path lengkap untuk skenario lain.
- `tests/`: pengujian data, prediksi, API, adaptive, serta resolusi path. Prefix `test_urban_` menguji Surabaya.

## Status layanan

| Bagian | Status dalam kode |
|---|---|
| Dhompo file/MLflow predictor dan FastAPI | Tersedia; membutuhkan artefak atau registry yang sesuai |
| Dhompo adaptive dan fallback dua tier | Implementasi dan tes tersedia; adaptive membutuhkan checkpoint PyTorch |
| Surabaya training, laporan, predictor Python | Tersedia; model dan scaler hasil training diperlukan |
| Surabaya melalui endpoint FastAPI | Belum diintegrasikan |
| Ingest, penjadwalan retraining, drift trigger, promosi/rollback otomatis | Rancangan dalam dokumen arsitektur; belum merupakan workflow otomatis yang tersedia |

Folder `artifacts/` menyimpan checkpoint adaptive dan ablation lokal. `mlruns/` menyimpan riwayat MLflow; `catboost_info/`, `.pytest_cache/`, dan `__pycache__/` adalah keluaran tool. Folder tersebut tidak perlu dibaca untuk memahami alur utama dan tetap diabaikan Git.
