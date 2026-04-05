# Prediksi Banjir Sungai Dhompo

Sistem prediksi tinggi muka air Sungai Dhompo berbasis deret waktu untuk kebutuhan pemantauan dini banjir. Repository ini mencakup training model multi-horizon (`h1`–`h5`) dan API inferensi dengan FastAPI.



## Struktur Proyek

```text
├── api/                 # FastAPI routes & schema
├── src/dhompo/          # Package utama (data, models, serving)
├── training/            # Script training & evaluasi
├── research/            # Notebook EDA & eksperimen
├── configs/             # Konfigurasi YAML
├── models/              # Artefak model lokal
├── reports/             # Output tabel & figur
├── docker-compose.yml   # Konfigurasi container
└── payload.json         # Contoh request /predict
```


## Instalasi

Repository ini memakai layout `src/`, jadi instalasi yang direkomendasikan untuk development adalah editable install.

1. Buat virtual environment:

```bash
python -m venv .venv
```

2. Aktifkan virtual environment:

```powershell
.venv\Scripts\activate
```

3. Install dependency sesuai kebutuhan:

Untuk development, testing, dan training:

```bash
pip install -e ".[dev]"
```

Untuk runtime minimum API saja:

```bash
pip install -r requirements.txt
```

Catatan:

- `pip install -e ".[dev]"` direkomendasikan agar package `dhompo` bisa di-import langsung dari source code.
- Mode ini juga memudahkan menjalankan `pytest`, script training, dan FastAPI tanpa perlu mengatur `PYTHONPATH` manual.
- Jika hanya memasang `requirements.txt`, beberapa workflow development bisa tetap membutuhkan konfigurasi path tambahan.

## Menjalankan API

### Local

Backend default lokal adalah `file`.

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Endpoint utama:

| Method | Path          | Deskripsi                |
|--------|---------------|--------------------------|
| `GET`  | `/health`     | Health check             |
| `GET`  | `/model-info` | Info backend dan model   |
| `POST` | `/predict`    | Prediksi tinggi muka air |

Dokumentasi Swagger tersedia di `http://localhost:8000/docs`.

### Docker

Repository ini juga menyediakan `docker-compose.yml` untuk menjalankan service secara containerized.

```bash
docker compose up --build -d
```

URL default:

| Service | URL                       |
|---------|---------------------------|
| API     | `http://localhost:8000`   |
| Docs    | `http://localhost:8000/docs` |

## Menjalankan Training

```bash
python training/train_sklearn.py --config configs/sklearn_model.yaml
```

Training configuration utama ada di `configs/training.yaml`, termasuk:

- split temporal train/test
- daftar horizon
- sumber data
- feature flags seperti `travel_time_lags`, `cumulative_rainfall`, `interaction_features`, dan `seasonal_features`

Penting:

- feature engineering saat training harus konsisten dengan feature engineering saat inferensi
- jika artefak model dibuat dengan konfigurasi fitur yang berbeda, inferensi dapat gagal karena mismatch nama fitur

## Menjalankan Test

Setelah editable install:

```bash
pytest -q
```

Alternatif via Makefile:

```bash
make test
```

## Contoh Request Prediksi

Contoh payload tersedia di `payload.json`.

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  --data-binary "@payload.json"
```

```powershell
Invoke-WebRequest `
  -Method Post `
  -Uri http://localhost:8000/predict `
  -ContentType "application/json" `
  -InFile payload.json | Select-Object -ExpandProperty Content
```

Validasi request:

- minimal `24` baris history
- interval tepat `30` menit
- semua stasiun wajib ada di setiap baris

## Troubleshooting

| Masalah | Penyebab umum | Solusi |
|---------|----------------|--------|
| `ModuleNotFoundError: dhompo` saat test | package belum terpasang editable | jalankan `pip install -e ".[dev]"` |
| `/predict` gagal karena feature names mismatch | artefak model tidak cocok dengan konfigurasi fitur inferensi | samakan konfigurasi feature engineering atau latih ulang model |
| scaler/model file tidak ditemukan | artefak lokal belum tersedia | pastikan isi `models/sklearn/` lengkap |
