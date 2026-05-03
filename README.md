# Rehabilitasi Project — Deteksi Anomali Gerakan Rehabilitasi Tangan

> **Judul Skripsi:** Deteksi Anomali Gerakan Rehabilitasi Tangan Menggunakan Autoencoder Berbasis Data Pose Estimation (MediaPipe)

---

## Struktur Proyek

```
Rehabilitasi_Project/
│
├── data/
│   ├── raw_videos/          ← Letakkan 10–20 video rekaman gerakan NORMAL di sini (.mp4)
│   └── processed_data/      ← Output CSV dari Fase 1 akan tersimpan di sini
│
├── models/
│   └── otak_Rehabilitasi.h5  ← Unduh dari Google Colab setelah Fase 2 selesai
│
├── fase1_data_factory.py    ← Jalankan PERTAMA (ekstraksi data)
├── fase2_colab_training.ipynb ← Upload & jalankan di Google Colab
├── fase3_Rehabilitasi_app.py ← Jalankan TERAKHIR (aplikasi real-time)
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan

### Langkah 0 — Install Dependencies (sekali saja)
```bash
pip install -r requirements.txt
```

### Fase 1 — Ekstraksi Data (Lokal)
1. Letakkan file video rekaman gerakan rehabilitasi **normal** di folder `data/raw_videos/`.
2. Jalankan skrip berikut:
   ```bash
   python fase1_data_factory.py
   ```
3. Output: `data/processed_data/dataset_normal.csv`

### Fase 2 — Pelatihan Model (Google Colab)
1. Buka [Google Colab](https://colab.research.google.com/) dan unggah file `fase2_colab_training.ipynb`.
2. Unggah juga file `dataset_normal.csv` ke Colab.
3. Jalankan semua cell. Catat nilai **THRESHOLD** dari output terakhir.
4. Unduh file `otak_Rehabilitasi.h5` dan letakkan di folder `models/`.

### Fase 3 — Aplikasi Real-Time (Lokal)
1. Buka file `fase3_Rehabilitasi_app.py`.
2. Ganti nilai variabel `THRESHOLD` dengan angka dari output Colab.
3. Jalankan:
   ```bash
   python fase3_Rehabilitasi_app.py
   ```
4. Arahkan tangan ke kamera webcam.
5. Tekan **Q** untuk keluar.

---

## Konsep Sistem

| Komponen | Detail |
|---|---|
| **Data** | 21 landmark tangan × 3 sumbu (x, y, z) = 63 fitur |
| **Normalisasi** | Ego-centric: pergelangan tangan (landmark 0) sebagai titik pusat (0,0,0) |
| **Sekuensialisasi** | Sliding window 30 frame |
| **Model** | LSTM-Autoencoder (hanya dilatih dengan data normal) |
| **Deteksi Anomali** | MSE rekonstruksi error vs. Threshold |
| **Threshold** | μ(MSE) + 3σ(MSE) dari data latih |

---

## Logika Deteksi

```
MSE < Threshold  →  ✅ NORMAL
MSE ≥ Threshold  →  🚨 ANOMALI TERDETEKSI
```

---

## Catatan Penting
- **Jangan ubah** dimensi `reshape` dan normalisasi ego-centric di kedua skrip.
- Nilai `WINDOW_SIZE` di Fase 1, Fase 2, dan Fase 3 **harus sama** (default: 30).
- Threshold yang digunakan di `fase3_Rehabilitasi_app.py` **harus menggunakan** nilai dari output Colab Fase 2.

