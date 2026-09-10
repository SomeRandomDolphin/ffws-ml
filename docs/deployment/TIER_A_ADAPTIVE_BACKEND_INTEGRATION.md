# Integrasi Backend Tier-A Adaptive

Dokumen ini menjelaskan setup backend untuk memakai `tier_a_adaptive` sebagai
model utama API prediksi. Backend ini memakai checkpoint PyTorch lokal, bukan
file model sklearn.

## 1. Prasyarat

Gunakan Python `>=3.10`, lalu install dependency dari root repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pip install -r requirements-torch.txt
```

Di Windows PowerShell, aktivasi venv:

```powershell
.venv\Scripts\activate
```

Jika hanya butuh runtime API tanpa test:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-torch.txt
```

## 2. Train Artefak Adaptive

Jalankan training untuk menghasilkan checkpoint dan normalizer:

```bash
python training/dhompo/run_tier_a_adaptive.py --epochs 200 --batch-size 256
```

Smoke test cepat:

```bash
python training/dhompo/run_tier_a_adaptive.py --epochs 5 --batch-size 64
```

Output wajib:

```text
artifacts/tier_a_adaptive/
|-- best.pt
`-- normalizer.pkl
```

`best.pt` dan `normalizer.pkl` harus berasal dari run training yang sama.

## 3. Jalankan API Dengan Adaptive Backend

```bash
export PREDICTOR_BACKEND=tier_a_adaptive
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

PowerShell:

```powershell
$env:PREDICTOR_BACKEND="tier_a_adaptive"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Saat startup, API akan memuat:

- `artifacts/tier_a_adaptive/best.pt`
- `artifacts/tier_a_adaptive/normalizer.pkl`

Lalu model dibungkus oleh `TwoTierPredictor`, sehingga fallback Tier-B tetap
aktif jika semua telemetry utama bermasalah.

## 4. Validasi Readiness

```bash
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

## 5. Test Integrasi

Install dev dependency dulu:

```bash
python -m pip install -e ".[dev]"
```

Lalu jalankan test dengan module invocation supaya pasti memakai Python dari
venv aktif:

```bash
python -m pytest tests/test_tier_a_adaptive_predictor.py -q
python -m pytest tests/test_two_tier.py -q
python -m pytest tests/test_api.py -q
```

Jika command `pytest` tidak ditemukan, jangan install `pytest` via `apt`.
Masalahnya ada di Python environment; gunakan `python -m pip install -e ".[dev]"`.

## 6. Kontrak Request

Endpoint:

```text
POST /predict
Content-Type: application/json
```

Payload wajib berisi `history` dengan minimal 24 baris, interval 30 menit,
timestamp urut naik, dan semua stasiun wajib:

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

Adaptive model juga mengenal `Jalan Nasional`. API saat ini tidak mewajibkannya;
jika tidak dikirim, slot tersebut akan dianggap missing/masked.

## 7. Response Yang Perlu Disimpan

Simpan field berikut untuk audit dan monitoring:

- `predictions`: prediksi h1 sampai h5
- `backend`: harus `tier_a_adaptive` saat adaptive aktif
- `serving_tier`: `A` untuk adaptive, `B` untuk fallback
- `degradation`: alasan degradasi per horizon
- `shadow_predictions`: prediksi fallback saat Tier-A melayani
- `quality_flags`: status kualitas sensor pada timestep terakhir
- `timestamp`: timestamp observasi terakhir
- `prediction_time`: waktu server membuat prediksi

## 8. Gate Kualitas Sebelum Production

`best.pt` adalah kandidat model, bukan otomatis production-ready. Sebelum
dipakai operasional, bandingkan dengan backend production lama memakai:

- RMSE per horizon h1 sampai h5
- MAE per horizon h1 sampai h5
- peak-weighted RMSE
- error saat puncak banjir
- CSI pada threshold banjir, misalnya 9.0 m
- robustness saat Klosod, AWLR Kademungan, atau Purwodadi bermasalah

Untuk production, jalankan adaptive sebagai shadow terlebih dahulu: request yang
sama dikirim ke backend lama dan instance adaptive, lalu hasilnya dibandingkan
ketika ground truth tersedia.

## 9. Troubleshooting

| Masalah | Penyebab umum | Solusi |
|---------|---------------|--------|
| `pytest: command not found` | dev dependency belum terinstall | jalankan `python -m pip install -e ".[dev]"` |
| `ModuleNotFoundError: dhompo` | pytest tidak menemukan package layout `src/` atau dijalankan dari direktori lain | jalankan dari root repo dengan `python -m pytest ...` |
| `Package 'dhompo' requires a different Python` | Python venv di bawah versi minimum package | gunakan Python `>=3.10`, lalu reinstall package |
| `/health` `503` | checkpoint atau normalizer tidak ada | cek `artifacts/tier_a_adaptive/` dan jalankan training |
| `/health` `503` dengan error torch | PyTorch belum terinstall | jalankan `python -m pip install -r requirements-torch.txt` |
| `/predict` `422` | payload tidak valid | cek minimal 24 baris, interval 30 menit, dan nama stasiun |
| `serving_tier` selalu `B` | telemetry utama dianggap bad | cek Purwodadi, AWLR Kademungan, dan Klosod pada history terakhir |
