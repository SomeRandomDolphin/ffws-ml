# API dan Deployment Dhompo

API yang tersedia melayani **Dhompo**, dengan tinggi muka air dalam **meter**. Alur Surabaya menggunakan predictor Python seperti dijelaskan dalam [panduan Surabaya](../surabaya/index.md).

## Menjalankan layanan

Dari root repository setelah instalasi:

```powershell
$env:PREDICTOR_BACKEND="file"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

| Backend | Kebutuhan |
|---|---|
| `file` (default lokal) | Model lokal di `models/sklearn/` |
| `mlflow` (default Docker Compose) | Registry MLflow dan alias model; scaler lokal untuk horizon linear |
| `tier_a_adaptive` | PyTorch, `artifacts/tier_a_adaptive/best.pt`, dan `normalizer.pkl` |

Model file Dhompo yang dipakai: `model_xgboost_h1.pkl`, `model_gradient_boosting_h2.pkl`, `model_gradient_boosting_h3.pkl`, `model_lasso_l1_h4.pkl`, dan `model_lasso_l1_h5.pkl`. Jika diperlukan model linear, `scaler.pkl` harus berasal dari training yang sama. Ketersediaan scaler dan kecocokan fitur perlu diperiksa untuk artefak yang akan digunakan.

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/model-info
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -InFile payload.json
```

Swagger: `http://localhost:8000/docs`.

## Kontrak prediksi

- `POST /predict` menerima `history` dengan minimal 24 observasi, interval tepat 30 menit, timestamp berurutan pada menit :00 atau :30.
- Setiap observasi berisi `timestamp` dan `readings`; semua nama stasiun yang diwajibkan harus tersedia dan nilainya numerik dalam meter.
- Gunakan `/model-info` untuk nama stasiun serta kebutuhan history, dan `payload.json` di root sebagai contoh lengkap.
- Response berisi `predictions.h1`–`h5`, backend/model, waktu observasi dan prediksi, serta metadata `serving_tier`, `degradation`, `shadow_predictions`, dan `quality_flags`.
- Tier-B adalah fallback berbasis history Dhompo ketika ketiga telemetri utama bermasalah. Schema dan validasi input tetap berlaku ketika fallback digunakan.

| Status | Makna |
|---|---|
| 200 | Berhasil; untuk health berarti model siap |
| 422 | Input prediksi tidak valid |
| 503 | Model belum siap atau gagal melayani prediksi |

## Docker dan MLflow

```powershell
docker compose up --build -d
```

Compose menjalankan API pada port 8000 dan MLflow pada port 5000, memakai volume `configs/`, `models/`, dan `mlruns/`. Registry yang diperlukan adalah `dhompo_h1` sampai `dhompo_h5` dengan alias `production` secara default. Alias dapat diganti melalui `MODEL_ALIAS`.

Menyalakan container belum membuat registry berisi model. Training dan penyiapan alias diperlukan sebelum backend MLflow siap.

## Referensi

- [Docker dan MLflow](docker-mlflow.md)
- [Backend Tier-A Adaptive](backend_integration.md)
- [Detail integrasi adaptive](TIER_A_ADAPTIVE_BACKEND_INTEGRATION.md)
