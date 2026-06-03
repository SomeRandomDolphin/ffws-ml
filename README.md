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

Repository ini memakai layout `src/`, jadi instalasi yang direkomendasikan untuk development adalah editable install. Python yang didukung adalah `>=3.10`.

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
python -m pip install -e ".[dev]"
```

Untuk runtime minimum API saja:

```bash
python -m pip install -r requirements.txt
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

## Panduan Integrasi

Bagian ini merangkum hal yang perlu disiapkan saat API ini dihubungkan ke
frontend, dashboard, ETL, atau service lain.

### 1. Pilih backend model

API membaca backend dari environment variable `PREDICTOR_BACKEND`.

| Backend | Kapan dipakai | Kebutuhan |
|---------|---------------|-----------|
| `file` | Integrasi lokal / deployment sederhana | file model di `models/sklearn/` |
| `mlflow` | Deployment dengan MLflow Model Registry | MLflow server aktif dan model punya alias |
| `tier_a_adaptive` | Eksperimen Tier-A PyTorch adaptive | dependency torch dan artefak adaptive |

Contoh local file backend:

```powershell
$env:PREDICTOR_BACKEND="file"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Contoh MLflow backend:

```powershell
$env:PREDICTOR_BACKEND="mlflow"
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
$env:MODEL_ALIAS="production"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Docker Compose sudah memakai backend `mlflow` dan menghubungkan API ke service
`mlflow` melalui network Docker.

### 2. Siapkan artefak model

Untuk backend `file`, direktori `models/sklearn/` harus berisi:

```text
models/sklearn/
|-- xgboost_h1.pkl
|-- gradient_boosting_h2.pkl
|-- gradient_boosting_h3.pkl
|-- lasso_alpha001_h4.pkl
|-- lasso_alpha001_h5.pkl
`-- scaler.pkl
```

Untuk backend `mlflow`, registry harus memiliki model:

```text
dhompo_h1@production
dhompo_h2@production
dhompo_h3@production
dhompo_h4@production
dhompo_h5@production
```

Alias dapat diganti dengan `MODEL_ALIAS`. Backend MLflow tetap membutuhkan
`models/sklearn/scaler.pkl` untuk horizon `h4` dan `h5`.

Untuk backend `tier_a_adaptive`, install dependency tambahan dan pastikan
artefak tersedia:

```bash
python -m pip install -r requirements-torch.txt
python training/run_tier_a_adaptive.py
```

Output yang dibaca saat serving:

```text
artifacts/tier_a_adaptive/best.pt
artifacts/tier_a_adaptive/normalizer.pkl
```

Langkah backend yang lebih lengkap tersedia di
[`docs/TIER_A_ADAPTIVE_BACKEND_INTEGRATION.md`](docs/TIER_A_ADAPTIVE_BACKEND_INTEGRATION.md).

### 3. Validasi readiness sebelum dipakai

Service integrator sebaiknya mengecek readiness sebelum mengirim prediksi:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-info
```

`/health` akan mengembalikan HTTP `200` jika model siap dan `503` jika model
gagal dimuat. `/model-info` mengembalikan backend aktif, daftar model,
minimal history, dan daftar stasiun yang wajib dikirim.

### 4. Kirim request prediksi

Endpoint utama integrasi:

```text
POST /predict
Content-Type: application/json
```

Kontrak input:

- body berisi field `history`
- minimal `24` baris history atau `12` jam terakhir
- timestamp berurutan naik dengan jarak tepat `30` menit
- timestamp berada di boundary `:00` atau `:30`
- setiap baris wajib berisi semua stasiun yang dibutuhkan
- nilai tinggi muka air dikirim sebagai angka dalam meter

Nama stasiun wajib sama persis:

```text
Bd. Suwoto
Krajan Timur
Purwodadi
Bd. Lecari
Bd. Bakalan
Bd. Baong
AWLR Kademungan
Bd Guyangan
Sidogiri
Bd. Domas
Klosod
Bd. Grinting
Dhompo
```

Contoh request lengkap tersedia di `payload.json`.

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  --data-binary "@payload.json"
```

Catatan integrasi data:

- jangan mengubah nama stasiun di frontend/client; lakukan mapping ke nama
  kanonis sebelum request
- jika source data memakai timezone, kirim timestamp secara konsisten,
  idealnya dengan offset ISO 8601
- kontrak API saat ini mewajibkan angka untuk semua stasiun; jika integrasi
  perlu mengirim sensor kosong/null, schema request perlu diperluas terlebih
  dahulu

### 5. Baca response prediksi

Response sukses memiliki bentuk utama berikut:

```json
{
  "predictions": {
    "h1": 1.23,
    "h2": 1.31,
    "h3": 1.42,
    "h4": 1.51,
    "h5": 1.64
  },
  "backend": "file",
  "models": {
    "h1": "xgboost_h1.pkl",
    "h2": "gradient_boosting_h2.pkl",
    "h3": "gradient_boosting_h3.pkl",
    "h4": "lasso_alpha001_h4.pkl",
    "h5": "lasso_alpha001_h5.pkl"
  },
  "timestamp": "2022-11-21T11:30:00",
  "prediction_time": "2026-06-04T00:00:00Z",
  "serving_tier": "A",
  "degradation": {},
  "shadow_predictions": {
    "h1": 1.2,
    "h2": 1.2,
    "h3": 1.2,
    "h4": 1.2,
    "h5": 1.2
  },
  "quality_flags": {
    "Dhompo": "OK"
  }
}
```

Interpretasi response:

- `predictions.h1` sampai `predictions.h5` adalah prediksi tinggi muka air
  Dhompo untuk +1 sampai +5 jam
- `timestamp` adalah timestamp observasi terakhir dari request
- `prediction_time` adalah waktu server saat prediksi dibuat
- `serving_tier="A"` berarti model utama dipakai
- `serving_tier="B"` berarti fallback Dhompo-only aktif karena telemetri utama
  bermasalah
- `degradation` berisi alasan degradasi per horizon, misalnya
  `PRIMARY_STATION_FLATLINE:Purwodadi`
- `shadow_predictions` berisi prediksi fallback saat Tier-A melayani request;
  nilainya `null` saat Tier-B sedang melayani request
- `quality_flags` dapat dipakai frontend untuk menampilkan status kualitas data

### 6. Error handling untuk client

| Status | Arti | Tindakan client |
|--------|------|-----------------|
| `200` | Prediksi berhasil | tampilkan prediksi dan metadata kualitas |
| `422` | Request tidak valid | cek jumlah history, interval timestamp, dan nama stasiun |
| `503` | Model/API belum siap | tampilkan status tidak tersedia dan retry setelah health check OK |

### 7. Checklist integrasi

- tentukan backend serving: `file`, `mlflow`, atau `tier_a_adaptive`
- pastikan artefak model tersedia sesuai backend
- jalankan API dan cek `/health`
- gunakan `/model-info` sebagai sumber konfigurasi client
- mapping nama stasiun dari sumber data ke nama kanonis di README ini
- kirim minimal 24 baris history berinterval 30 menit ke `/predict`
- tampilkan `predictions`, `serving_tier`, `degradation`, dan `quality_flags`
- simpan response metadata untuk audit dan troubleshooting
- API saat ini belum punya auth dan CORS masih terbuka; jika API dibuka ke
  jaringan publik, pasang auth/rate-limit di reverse proxy atau API gateway

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
python -m pytest -q
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
| `ModuleNotFoundError: dhompo` saat test | package belum terpasang editable | jalankan `python -m pip install -e ".[dev]"` |
| `ModuleNotFoundError: dhompo` tetap muncul setelah install | pytest tidak membaca layout `src/` atau command dijalankan dari direktori lain | jalankan dari root repo dengan `python -m pytest ...`; konfigurasi pytest sudah menambahkan `src/` ke path |
| `pytest: command not found` | dependency test belum terinstall atau shell tidak memakai venv | jalankan `python -m pip install -e ".[dev]"`, lalu `python -m pytest -q` |
| `Package 'dhompo' requires a different Python` | versi Python venv tidak sesuai constraint package | gunakan Python `>=3.10`, lalu reinstall package |
| `/predict` gagal karena feature names mismatch | artefak model tidak cocok dengan konfigurasi fitur inferensi | samakan konfigurasi feature engineering atau latih ulang model |
| scaler/model file tidak ditemukan | artefak lokal belum tersedia | pastikan isi `models/sklearn/` lengkap |
| `/health` mengembalikan `503` | backend/model gagal dimuat saat startup | cek `PREDICTOR_BACKEND`, artefak model, dan `MLFLOW_TRACKING_URI` |
| client mendapat `422` | payload tidak memenuhi kontrak API | cek minimal 24 baris, interval 30 menit, dan nama stasiun persis |
