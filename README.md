# Prediksi Banjir Sungai Dhompo

> Prediksi tinggi muka air multi-horizon (`h1`–`h5`) untuk Sungai Dhompo.
> Stack: **FastAPI** · **scikit-learn/XGBoost** · **MLflow Registry** · **Docker Compose**

---

## Daftar Isi

- [Struktur Proyek](#struktur-proyek)
- [Prasyarat](#prasyarat)
- [Instalasi](#instalasi)
- [Menjalankan Service](#menjalankan-service)
- [Backend Serving](#backend-serving)
- [Training & Registry MLflow](#training--registry-mlflow)
- [Uji Prediksi](#uji-prediksi)
- [Troubleshooting](#troubleshooting)

---

## Struktur Proyek

```text
├── api/                 # FastAPI routes & schema
├── src/dhompo/          # Package utama (data, models, serving)
├── training/            # Script training & evaluasi
├── research/            # Notebook EDA & modeling
├── configs/             # Konfigurasi YAML
├── models/              # Scaler & artefak lokal
├── reports/             # Output tabel & figur
├── docker-compose.yml   # MLflow + API
└── payload.json         # Contoh request /predict
```

## Prasyarat

- Python `>=3.12`
- Docker Desktop + Docker Compose

## Instalasi

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

## Menjalankan Service

### Local

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker compose up --build -d
```

### Endpoint

| Method | Path          | Deskripsi               |
|--------|---------------|-------------------------|
| `GET`  | `/health`     | Health check            |
| `GET`  | `/model-info` | Info model yang aktif   |
| `POST` | `/predict`    | Prediksi tinggi muka air|

### URL Dashboard

| Service | URL                          |
|---------|------------------------------|
| API     | http://localhost:8000/docs    |
| MLflow  | http://localhost:5000         |

## Serving Backend

API mendukung dua backend untuk memuat model saat inferensi.

### `PREDICTOR_BACKEND=file` (default)

- Load model langsung dari filesystem lokal `models/sklearn/*`
- Tidak membutuhkan MLflow registry saat inferensi

### `PREDICTOR_BACKEND=mlflow`

- Load model dari MLflow Model Registry (contoh: `models:/dhompo_h1@production`)
- Membutuhkan service `mlflow`, folder `mlruns/`, dan `models/sklearn/scaler.pkl`
- Konfigurasi via environment variable:

```bash
PREDICTOR_BACKEND=mlflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MODEL_ALIAS=production
```

## Training & Registry MLflow

### Menjalankan Training

```bash
python training/train_sklearn.py --config configs/sklearn_model.yaml
```

Script ini akan:
1. Load data dan bangun feature matrix
2. Train model untuk horizon `h1`–`h5`
3. Log params & metrics ke MLflow
4. Register model sebagai `dhompo_h1` s.d. `dhompo_h5`


### Set Alias `production`

1. Buka http://localhost:5000
2. Masuk ke tab _Model Training_
3. Masuk ke _Model Registry_
3. Pilih model _dhompo_h1_ s.d. _dhompo_h5_
4. Pada version yang dipilih, tambahkan alias `production`

### Verifikasi

```bash
# PowerShell
Invoke-RestMethod http://localhost:8000/model-info
```

Respons yang diharapkan:

```json
{
  "backend": "mlflow",
  "models": {
    "h1": "models:/dhompo_h1@production",
    "h2": "models:/dhompo_h2@production",
    "h3": "models:/dhompo_h3@production",
    "h4": "models:/dhompo_h4@production",
    "h5": "models:/dhompo_h5@production"
  }
}
```

## Uji Prediksi

Contoh request tersedia di `payload.json`.

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

Contoh respons:

```json
{
  "predictions": {
    "h1": 8.972,
    "h2": 9.4666,
    "h3": 10.0866,
    "h4": 12.4381,
    "h5": 13.2154
  },
  "backend": "mlflow",
  "models": { "h1": "models:/dhompo_h1@production", "..." : "..." },
  "timestamp": "2022-11-21T11:30:00",
  "prediction_time": "2026-03-10T04:08:51.245857Z"
}
```

### Validasi Request

- Minimal **24 baris** history
- Interval tepat **30 menit**
- Semua stasiun wajib ada di setiap baris

## Troubleshooting

| Masalah | Penyebab | Solusi |
|---------|----------|--------|
| `backend=file` padahal seharusnya `mlflow` | Env belum diterapkan | Recreate container `api` setelah ubah env, pastikan `PREDICTOR_BACKEND=mlflow` |
| `No versions found` / alias tidak ketemu | Alias belum diset | Set alias `production` pada model di MLflow UI |
| `Invalid Host header` dari MLflow | Konfigurasi host salah | Pastikan `--allowed-hosts` benar di `docker-compose.yml` |
| `No such file or directory` saat load model | Artifacts belum ada | Pastikan `mlruns/` berisi artifacts yang direferensikan registry |
| `Read-only file system` saat load model | Mount salah | Mount `mlruns` ke service `api` tanpa flag `read-only` |
| `Scaler file tidak ditemukan` | File belum ada | Pastikan `models/sklearn/scaler.pkl` tersedia |
