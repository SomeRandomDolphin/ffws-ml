# Pedoman kerja proyek

Repository ini mencakup riset prediksi muka air Dhompo/Welang dan Surabaya serta ITS Water Dashboard. Frontend di `frontend/` menggunakan Next.js, React, MapLibre, dan Recharts. Pedoman visual berikut berlaku untuk frontend; pekerjaan backend dan riset mengikuti kebutuhan tugas masing-masing.

## Sebelum mengubah frontend

- Baca [DESIGN.md](DESIGN.md) untuk arah visual, token, komponen, dan penyajian data.
- Ikuti [frontend/AGENTS.md](frontend/AGENTS.md), termasuk kewajiban membaca dokumentasi Next.js lokal yang relevan sebelum menulis kode. Pertahankan blok yang dikelola Next.js.
- Periksa status Git dan kode terkini. Pertahankan perubahan pengguna yang belum di-commit dan hindari perubahan di luar tugas.
- DESIGN.md mendokumentasikan target desain. Jangan menganggap seluruh frontend sudah menerapkannya atau sudah lolos pengujian.

<!-- antislop:start -->
## antislop

Pengguna telah memilih DURING dengan pemeriksaan akhir sebagai alur default pekerjaan UI. Terapkan aturan selama perencanaan dan implementasi, lalu jalankan Delivery Gate untuk hasil yang dikerjakan. Jangan menanyakan pilihan DURING/AFTER berulang kali; ubah mode bila pengguna meminta.

Muat core dan modul yang relevan dengan tugas melalui file lokal berikut:

- Core: [antislop](.codex/skills/antislop/SKILL.md).
- UI dan visual: [antislop-ui](.codex/skills/antislop-ui/SKILL.md).
- Teks: [antislop-copywriting](.codex/skills/antislop-copywriting/SKILL.md).
- Aksesibilitas dan keadaan kontrol: [antislop-human](.codex/skills/antislop-human/SKILL.md).
- Tata letak responsif: [antislop-layoutmobile](.codex/skills/antislop-layoutmobile/SKILL.md).
- Komentar kode: [antislop-code](.codex/skills/antislop-code/SKILL.md).

Audit AFTER terpisah dilakukan saat diminta: laporkan temuan bernomor dan prioritas sesuai skill, kemudian perbaiki temuan yang dipilih pengguna. Pemeriksaan akhir DURING merupakan bagian dari pekerjaan yang sudah diminta.

Persetujuan pengguna atas arah desain dan mode kerja tetap berlaku pada pekerjaan berikutnya. Tema terang pada DESIGN.md adalah pilihan pengguna untuk tahap ini; jangan menambahkan tema gelap hanya untuk memenuhi default skill.
<!-- antislop:end -->

## Implementasi dan integritas data

- Gunakan token dan komponen bersama untuk fungsi yang sama. Periksa komponen yang ada sebelum membuat komponen baru.
- Pusatkan token CSS pada fondasi global. Untuk MapLibre atau grafik yang memerlukan nilai JavaScript, gunakan pemetaan bersama yang konsisten dengan token; hindari salinan warna terpisah per halaman.
- Saat merapikan CSS, perbaiki aturan sumber dan override terkait. Hindari menambah override berlapis untuk menutupi konflik lama.
- Pertahankan struktur navigasi dan fungsi kontrol kecuali tugas meminta perubahan. Jangan membuat logo, aset, atau menu baru tanpa arahan yang sesuai.
- Penataan visual tidak mengubah perhitungan, ambang status, satuan, sumber data, API, atau logika prediksi.
- Bedakan data simulasi, live, kedaluwarsa, dan koneksi terputus. Pertahankan asumsi satuan yang belum terverifikasi; jangan menyajikannya sebagai kepastian.
- Gunakan label bersama warna untuk status. Ikuti pemisahan status muka air, kualitas air, dan kesegaran data pada DESIGN.md.

## Verifikasi dan pelaporan

Untuk perubahan UI, jalankan `npm run build` dari `frontend/`, lalu periksa halaman dan interaksi terdampak di browser. Uji lebar 375, 768, dan 1440px, fokus keyboard, penutupan panel dengan Escape, reduced motion, serta keadaan loading/kosong/error yang relevan.

Pada penataan lintas dashboard, periksa pencarian stasiun, pilihan wilayah, lapisan peta, legenda, popup, grafik, navigasi detail, dan kembali ke peta. Pastikan sumber/satuan data tetap benar. Catat tindakan dan hasil aktual, error console, serta hambatan seperti sumber live yang tidak tersedia.

Laporkan perubahan, bukti pemeriksaan, dan batasannya. Jangan menyatakan PASS untuk pemeriksaan yang belum dilakukan. Untuk perubahan Markdown saja, verifikasi isi, konsistensi, dan tautan lokal; build dan pengujian browser tidak diperlukan. Gate UI yang tidak relevan diberi keterangan tidak berlaku, bukan PASS hasil pengujian aplikasi.
