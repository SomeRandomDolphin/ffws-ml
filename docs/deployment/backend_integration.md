# Integrasi Backend Tier-A Adaptive

Dokumen ini menjelaskan langkah backend untuk memakai `tier_a_adaptive`
sebagai model utama API prediksi Dhompo. Backend ini memakai model PyTorch
`AdaptiveTierA`, bukan file model sklearn.

## Ringkasan Alur

```text
client/backend integrator
  -> POST /predict
  -> FastAPI validation
  -> TwoTierPredictor
  -> quality flag detection
  -> Tier-A AdaptivePredictor jika telemetry cukup sehat
  -> Tier-B persistence fallback jika semua telemetry utama bermasalah
  -> response predictions + serving_tier + degradation + quality_flags
```

Backend `tier_a_adaptive` tetap memakai router dua-tier yang sama. Perbedaannya
ada di model Tier-A yang dimuat: `TierAAdaptivePredictor` membaca checkpoint
PyTorch dari `artifacts/tier_a_adaptive/`.

## 1. Prasyarat

Jalankan dari root repository.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pip install -r requirements-torch.txt
```

Dependency torch dipisah di `requirements-torch.txt`, karena backend default
sklearn tidak membutuhkannya.

## 2. Train Artefak Adaptive

Jalankan training untuk membuat checkpoint dan normalizer:

```powershell
python training/dhompo/run_tier_a_adaptive.py --epochs 200 --batch-size 256
```

Untuk smoke test cepat:

```powershell
python training/dhompo/run_tier_a_adaptive.py --epochs 5 --batch-size 64
```

Output yang wajib tersedia:

```text
artifacts/tier_a_adaptive/
|-- best.pt
`-- normalizer.pkl
```

`best.pt` dipilih dari epoch dengan `test_main` terbaik pada temporal split
training script. `normalizer.pkl` harus berasal dari run training yang sama
dengan checkpoint.

Jika ingin logging eksperimen ke MLflow:

```powershell
python training/dhompo/run_tier_a_adaptive.py --epochs 200 --batch-size 256 --mlflow
```

Catatan: opsi `--mlflow` saat ini untuk logging metrics training. Serving
`tier_a_adaptive` tetap membaca artefak lokal dari `artifacts/tier_a_adaptive/`.

## 3. Konfigurasi Backend API

Set environment variable sebelum menjalankan FastAPI:

```powershell
$env:PREDICTOR_BACKEND="tier_a_adaptive"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Untuk Linux/systemd, environment minimalnya:

```bash
export PREDICTOR_BACKEND=tier_a_adaptive
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Saat startup, API akan:

1. membaca `PREDICTOR_BACKEND`
2. membuat `TierAAdaptivePredictor`
3. memuat `artifacts/tier_a_adaptive/best.pt`
4. memuat `artifacts/tier_a_adaptive/normalizer.pkl`
5. membungkusnya dengan `TwoTierPredictor`

Jika salah satu artefak tidak tersedia, `/health` akan mengembalikan `503`.

## 4. Validasi Readiness

Setelah API hidup:

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/model-info
```

Expected state:

```json
{
  "ready": true,
  "backend": "tier_a_adaptive"
}
```

`/model-info` harus menampilkan mapping model:

```json
{
  "models": {
    "h1": "tier_a_adaptive",
    "h2": "tier_a_adaptive",
    "h3": "tier_a_adaptive",
    "h4": "tier_a_adaptive",
    "h5": "tier_a_adaptive"
  }
}
```

## 5. Kontrak Request Dari Backend Integrator

Endpoint:

```text
POST /predict
Content-Type: application/json
```

Kontrak payload:

- field utama: `history`
- minimal 24 baris history
- interval tepat 30 menit
- timestamp urut naik
- timestamp di boundary `:00` atau `:30`
- nilai tinggi muka air dalam meter
- semua stasiun wajib dikirim sesuai nama kanonis API

Stasiun wajib:

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

Adaptive model juga mengenal `Jalan Nasional`. API tidak mewajibkannya, tetapi
boleh dikirim jika datanya tersedia. Jika tidak dikirim, slot tersebut akan
dianggap missing dan di-mask oleh model.

Contoh smoke request:

```powershell
Invoke-WebRequest `
  -Method Post `
  -Uri http://localhost:8000/predict `
  -ContentType "application/json" `
  -InFile payload.json | Select-Object -ExpandProperty Content
```

## 6. Interpretasi Response

Response sukses:

```json
{
  "predictions": {
    "h1": 1.23,
    "h2": 1.31,
    "h3": 1.42,
    "h4": 1.51,
    "h5": 1.64
  },
  "backend": "tier_a_adaptive",
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

Field yang harus disimpan backend integrator:

- `predictions`: hasil utama untuk h1 sampai h5
- `backend`: backend model aktif
- `serving_tier`: `A` untuk adaptive, `B` untuk fallback
- `degradation`: alasan degradasi per horizon jika sensor primer bermasalah
- `shadow_predictions`: output fallback saat Tier-A melayani request
- `quality_flags`: status kualitas sensor pada timestep terakhir
- `prediction_time`: waktu server membuat prediksi
- `timestamp`: timestamp observasi terakhir dari request

Jika `serving_tier` bernilai `B`, hasil yang dilayani bukan adaptive model,
tetapi fallback persistence dari nilai Dhompo terakhir.

## 7. Validasi Kualitas Sebelum Production

`best.pt` dari training adalah kandidat model, bukan otomatis production-ready.
Sebelum dipakai untuk operasional, lakukan gate berikut.

### 7.1 Test Integrasi

```powershell
pytest tests/test_tier_a_adaptive_predictor.py -q
pytest tests/test_two_tier.py -q
pytest tests/test_api.py -q
```

Test ini memastikan:

- checkpoint adaptive bisa dimuat
- response memiliki h1 sampai h5
- routing dua-tier tetap berjalan
- fallback Tier-B aktif saat telemetry utama rusak
- API tetap memenuhi kontrak response

### 7.2 Evaluasi Offline

Bandingkan adaptive dengan backend sebelumnya (`file` atau `mlflow`) pada data
holdout temporal.

Metric minimal:

- RMSE per horizon h1 sampai h5
- MAE per horizon h1 sampai h5
- peak-weighted RMSE
- error saat puncak banjir
- CSI pada threshold banjir, misalnya 9.0 m

Kriteria praktis:

- h1 tidak boleh lebih buruk dari production saat ini
- rata-rata semua horizon tidak boleh lebih buruk signifikan
- error di event banjir harus dicek manual, bukan hanya rata-rata global
- performa saat sensor masked harus tetap stabil

### 7.3 Uji Robustness Sensor

Minimal skenario uji:

- Klosod bermasalah: cek degradasi h1
- AWLR Kademungan bermasalah: cek degradasi h2 dan h3
- Purwodadi bermasalah: cek degradasi h4 dan h5
- semua telemetry utama bermasalah: `serving_tier` harus `B`
- satu stasiun offline atau missing: Tier-A tetap melayani jika telemetry utama
  tidak semuanya rusak

### 7.4 Shadow Run

Untuk integrasi production, jalankan adaptive sebagai shadow sebelum dijadikan
model utama.

Skema yang disarankan:

1. production tetap memakai backend lama
2. kirim request yang sama ke instance adaptive shadow
3. simpan response adaptive dan response production
4. saat ground truth tersedia, hitung error kedua model
5. promote adaptive hanya jika performanya stabil

## 8. Deployment Docker

`docker/Dockerfile.api` saat ini hanya menginstall `requirements.txt`, sehingga
image default belum cukup untuk `tier_a_adaptive` karena belum ada torch.

Jika ingin memakai Docker untuk adaptive backend, image API harus:

1. menginstall `requirements.txt`
2. menginstall `requirements-torch.txt`
3. menyalin source `src/`, `api/`, dan `configs/`
4. mendapatkan mount/read access ke `artifacts/tier_a_adaptive/`
5. menjalankan container dengan `PREDICTOR_BACKEND=tier_a_adaptive`

Environment minimum:

```yaml
environment:
  - PREDICTOR_BACKEND=tier_a_adaptive
volumes:
  - ./configs:/app/configs:ro
  - ./artifacts:/app/artifacts:ro
```

## 9. Rollback

Rollback dilakukan dengan mengganti backend.

Kembali ke sklearn file backend:

```powershell
$env:PREDICTOR_BACKEND="file"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Kembali ke MLflow backend:

```powershell
$env:PREDICTOR_BACKEND="mlflow"
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
$env:MODEL_ALIAS="production"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Untuk deployment production, rollback sebaiknya cukup mengubah environment dan
restart service, tanpa menghapus artefak adaptive.

## 10. Troubleshooting

| Masalah | Penyebab umum | Solusi |
|---------|---------------|--------|
| `/health` `503` | `best.pt` atau `normalizer.pkl` tidak ada | jalankan training adaptive dan cek folder `artifacts/tier_a_adaptive/` |
| `/health` `503` dengan error torch | dependency PyTorch belum terinstall | jalankan `pip install -r requirements-torch.txt` |
| `/predict` `422` | payload tidak memenuhi schema | cek minimal 24 baris, interval 30 menit, dan nama stasiun |
| `serving_tier` selalu `B` | telemetry utama dianggap bad | cek nilai Purwodadi, AWLR Kademungan, dan Klosod pada history terakhir |
| prediksi tidak masuk akal | checkpoint belum tervalidasi | ulangi evaluasi offline, cek data training, dan bandingkan dengan backend lama |
| Docker adaptive gagal start | image tidak punya torch atau artefak tidak dimount | update Dockerfile/image dan mount `./artifacts` |

## File Terkait

- `api/predictor_state.py`: memilih backend dan membuat predictor
- `api/routes/predict.py`: endpoint `/predict`
- `src/dhompo/serving/tier_a_adaptive.py`: wrapper serving adaptive
- `src/dhompo/models/adaptive.py`: arsitektur model PyTorch
- `training/dhompo/run_tier_a_adaptive.py`: training checkpoint adaptive
- `src/dhompo/serving/two_tier.py`: routing Tier-A dan Tier-B fallback
