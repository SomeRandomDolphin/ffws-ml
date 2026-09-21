# Dataset Sintetis DAS Welang

Direktori keluaran lokal `scripts/build_synthetic.py`. CSV hasil generasi
diabaikan Git; hanya dokumen ini yang menjelaskan provenance dan regenerasi.

Data dibangun dari generator hujan Markov-Gamma dan simulator transfer routing
15 stasiun. Parameter dikalibrasi terhadap data generated 2022/2023, bukan
observasi lapangan. Dataset hanya sesuai untuk eksperimen surrogate, augmentasi,
dan skenario, bukan klaim prakiraan operasional.

```powershell
# kalibrasi parameter saja
python scripts/build_synthetic.py --calibrate-only

# smoke test 30 hari
python scripts/build_synthetic.py --periods 1440 --output data/synthetic/smoke.csv

# dataset default 10 tahun
python scripts/build_synthetic.py
```
