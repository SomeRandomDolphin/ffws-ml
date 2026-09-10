# Riset Prediksi Muka Air Dhompo dan Surabaya

Proyek ini memiliki dua skenario prediksi dengan data dan target berbeda. Keduanya memakai observasi 30 menit untuk memprediksi +1 hingga +5 jam.

| Skenario | Target dan satuan | Panduan |
|---|---|---|
| DAS Dhompo | Tinggi muka air stasiun Dhompo, meter | [Mulai dari Dhompo](dhompo/index.md) |
| Surabaya | Tinggi air Hang Tuah, sentimeter | [Mulai dari Surabaya](surabaya/index.md) |

Ikuti alur **data → EDA → fitur → training → evaluasi → prediksi** dalam panduan masing-masing. Eksperimen A/B/C adalah variasi Dhompo, sedangkan `urban` adalah nama teknis yang digunakan kode Surabaya.

## Navigasi

- [Peta kode dan status implementasi](code-map.md)
- [Migrasi folder dan perintah](migration.md)
- [API Dhompo dan deployment](deployment/index.md)

Instalasi untuk riset, dari root repository:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Tambahkan `python -m pip install -r requirements-torch.txt` untuk notebook deep learning dan backend adaptive. Untuk dokumentasi gunakan `python -m pip install -e ".[docs]"`, kemudian `python -m mkdocs serve`.
