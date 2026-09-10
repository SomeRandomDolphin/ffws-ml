# Prediksi Muka Air — Dhompo dan Surabaya

Repository riset prediksi tinggi muka air dengan dua skenario: **DAS Dhompo** dan **Surabaya**. Mulai dari panduan skenario untuk mengikuti alur data → EDA → fitur → training → evaluasi → prediksi.

| Aspek | DAS Dhompo | Surabaya |
|---|---|---|
| Target | Stasiun Dhompo | Hang Tuah, target awal yang dapat dikonfigurasi |
| Satuan tinggi air | Meter (m) | Sentimeter (cm) |
| Interval data | 30 menit | 30 menit |
| Horizon | +1 sampai +5 jam | +1 sampai +5 jam |
| Sumber | Data bersih 2022 dan data generated 2023 | CSV sensor perkotaan dalam format wide |
| Variasi riset | Eksperimen A/B/C, fitur tambahan, delta, smoothing, deep learning | `persistence_residual` (default), `delta`, pembanding persistence |
| Penggunaan prediksi | FastAPI dan predictor Python | Predictor Python; belum terhubung ke API |
| Mulai membaca | [Panduan Dhompo](docs/dhompo/index.md) | [Panduan Surabaya](docs/surabaya/index.md) |

A/B/C adalah eksperimen **di dalam Dhompo**. Istilah `urban` pada nama modul dan model berarti skenario **Surabaya**.

## Instalasi

Jalankan dari root repository, dengan Python 3.10 atau lebih baru:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Untuk riset deep learning atau backend adaptive, tambahkan:

```powershell
python -m pip install -r requirements-torch.txt
```

Untuk membangun dokumentasi secara lokal:

```powershell
python -m pip install -e ".[docs]"
python -m mkdocs serve
```

## Mulai dari sini

1. Baca panduan [Dhompo](docs/dhompo/index.md) atau [Surabaya](docs/surabaya/index.md): sumber data, target, dan batasan masing-masing.
2. Buka notebook EDA di `research/<skenario>/`, lalu ikuti urutan notebook dan eksperimen dalam panduan.
3. Pilih konfigurasi di `configs/<skenario>/` dan jalankan skrip di `training/<skenario>/`.
4. Periksa hasil di `reports/<skenario>/`. Model Dhompo berada di `models/sklearn/` atau `models/pytorch/`; model Surabaya lokal berada di `models/surabaya/`. Ketersediaannya dijelaskan dalam [indeks model](models/README.md).

Periksa pipeline Surabaya tanpa melatih atau menimpa model:

```powershell
python training/surabaya/train_urban_sklearn.py --dry-run
```

Validasi data Dhompo:

```powershell
python training/dhompo/validate_data.py
```

## Struktur repository

```text
configs/     dhompo/, surabaya/, shared/ — konfigurasi eksperimen
data/        dataset Dhompo + surabaya/ — dataset lokal (data baru diabaikan Git)
research/    dhompo/, surabaya/         — notebook berurutan
training/    dhompo/, surabaya/         — training dan pembuatan laporan
models/      sklearn/, pytorch/, surabaya/ — artefak model lokal
reports/     dhompo/, surabaya/         — tabel dan figur hasil riset
docs/        dhompo/, surabaya/         — panduan dan catatan riset
src/dhompo/                            — implementasi data, model, serving
api/                                  — FastAPI khusus Dhompo
tests/                                — pengujian kedua skenario
```

Fungsi metrik bersama berada di `training/evaluate.py`; helper figur di `research/eda_helpers.py`; hyperparameter sklearn bersama di `configs/shared/`. Nama package `dhompo` dipertahankan untuk kompatibilitas import. [Peta kode](docs/code-map.md) menjelaskan modul untuk masing-masing skenario.

## Prediksi melalui API Dhompo

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API menyediakan `GET /health`, `GET /model-info`, dan `POST /predict`. Swagger tersedia di `http://localhost:8000/docs`; contoh request ada di [payload.json](payload.json). Artefak model perlu tersedia sebelum prediksi dapat dilayani.

Lihat [panduan API, Docker, dan MLflow](docs/deployment/index.md) untuk konfigurasi backend, kontrak input, dan pemeriksaan readiness.

## Pengujian dan catatan repository

```powershell
python -m pytest -q
```

- [Panduan migrasi folder dan perintah](docs/migration.md) menjelaskan lokasi baru setelah penataan.
- [Indeks data](data/README.md), [model](models/README.md), dan [laporan](reports/README.md) menjelaskan isi dan ketersediaan berkas.
- Model hasil training, laporan, `artifacts/`, `mlruns/`, `catboost_info/`, dan cache adalah hasil lokal. Dokumen Markdown dan kode sumber dilacak Git; beberapa model Dhompo yang sebelumnya dilacak tetap dipertahankan.
- Dokumen arsitektur lama memuat rancangan lanjutan. Status implementasi saat ini dijelaskan di panduan skenario dan [peta kode](docs/code-map.md).
