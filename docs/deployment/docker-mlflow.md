# Docker & MLflow

## Docker Compose

Sistem terdiri dari dua service:

```yaml
services:
  mlflow:    # MLflow Tracking Server (:5000)
  api:       # FastAPI Prediction Service (:8000)
```

### Menjalankan

```bash
docker compose up --build -d
```

### Dashboard

| Service | URL |
|---------|-----|
| API Docs (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

## MLflow Model Registry

### Konsep

Setiap model yang di-train didaftarkan ke MLflow Registry dengan nama:

- `dhompo_h1`
- `dhompo_h2`
- `dhompo_h3`
- `dhompo_h4`
- `dhompo_h5`

### Alias-Based Deployment

Repo ini menggunakan pola **alias** (bukan stage yang sudah deprecated). API akan memuat model dengan alias `production`:

```
models:/dhompo_h1@production
models:/dhompo_h2@production
...
```

### Set Alias

1. Buka http://localhost:5000
2. Masuk ke tab **Models**
3. Pilih model `dhompo_h1` s.d. `dhompo_h5`
4. Pada version yang dipilih, tambahkan alias `production`

### Training & Registrasi

```bash
python training/dhompo/train_sklearn.py --config configs/shared/sklearn_model.yaml
```

Script ini akan:

1. Load data dan bangun feature matrix
2. Train 6 algoritma × 5 horizon = 30 model
3. Log parameter dan metrik ke MLflow
4. Register model terbaik ke MLflow Registry

### Verifikasi

```bash
# PowerShell
Invoke-RestMethod http://localhost:8000/model-info

# curl
curl http://localhost:8000/model-info
```

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `backend=file` padahal seharusnya `mlflow` | Recreate container `api`, pastikan `PREDICTOR_BACKEND=mlflow` |
| `No versions found` | Set alias `production` pada model di MLflow UI |
| `Invalid Host header` dari MLflow | Pastikan `--allowed-hosts` benar di `docker-compose.yml` |
| `No such file or directory` | Pastikan `mlruns/` berisi artifacts |
| `Read-only file system` | Mount `mlruns` tanpa flag `read-only` |
| `Scaler file tidak ditemukan` | Pastikan `models/sklearn/scaler.pkl` ada |
