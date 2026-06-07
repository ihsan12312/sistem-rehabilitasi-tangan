"""
==========================================================================
FASE 1: PABRIK DATA (DATA FACTORY)
Proyek Skripsi: Deteksi Anomali Gerakan Rehabilitasi Tangan
Menggunakan Autoencoder Berbasis Data Pose Estimation (MediaPipe)
==========================================================================
TUJUAN   : Mengekstrak landmark tangan dari video rekaman gerakan normal,
            menormalisasi koordinat, lalu menyimpannya ke file CSV.
EKSEKUSI : Jalankan di VS Code / komputer lokal.
OUTPUT   : data/processed_data/dataset_normal.csv
--------------------------------------------------------------------------
PENINGKATAN v2 (Advanced Data Quality):
- [BARU] EMA Smoothing Filter: meredam micro-jitter MediaPipe antar frame
- [BARU] Scale Invariance: membagi koordinat dengan jarak Wrist→MCP Jari
        Tengah (Titik 9), agar model kebal terhadap variasi jarak kamera
- [BARU] model_complexity=1: akurasi landmark lebih baik saat ekstraksi
==========================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import warnings
import logging

# ---- MEMBUNGKAM PERINGATAN INTERNAL (Log Noise) ----
os.environ['GLOG_minloglevel'] = '2'
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# -----------------------------------------------------------------------
# KONFIGURASI (Sesuaikan jika perlu)
# -----------------------------------------------------------------------
WINDOW_SIZE     = 30          # Jumlah frame per sekuens (sliding window)
VIDEO_FOLDER    = "data/raw_videos/"
OUTPUT_CSV      = "data/processed_data/dataset_normal.csv"
MIN_DETECT_CONF = 0.7         # Minimum confidence untuk deteksi tangan
# Frame Skipping: ambil setiap frame ke-N agar FPS ekstraksi cocok dengan FPS real-time
# Video asli 30 FPS, aplikasi real-time ~7.5 FPS → ambil setiap frame ke-4
FRAME_SKIP      = 4
# -----------------------------------------------------------------------


def extract_and_normalize(video_path: str, window_size: int = 30) -> np.ndarray:
    """
    Membaca satu file video, mengekstrak landmark tangan per-frame
    dengan MediaPipe, menormalisasi secara ego-centric (pergelangan
    tangan sebagai titik pusat 0,0,0) + Scale Invariance, lalu
    menghasilkan sekuens menggunakan sliding window.

    Normalisasi v2 (Advanced):
      1. EMA Smoothing     : meredam micro-jitter MediaPipe antar frame
      2. Translational Inv.: kurangi semua titik dengan Wrist (titik 0)
      3. Scale Invariance  : bagi dengan jarak Wrist → MCP Jari Tengah (titik 9)

    Sinkronisasi Temporal: Hanya mengambil setiap frame ke-FRAME_SKIP
    agar kecepatan gerakan yang dipelajari AI cocok dengan kecepatan
    kamera real-time (~7.5 FPS).

    Parameter:
        video_path  (str) : Path ke file video MP4.
        window_size (int) : Panjang sekuens (jumlah frame per window).

    Return:
        np.ndarray shape (N, window_size, 63)
            N  = jumlah sekuens yang terbentuk
            63 = 21 titik landmark * 3 sumbu (x, y, z)
        Mengembalikan array kosong jika video tidak valid.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,           # Ditingkatkan ke 1 untuk akurasi landmark lebih baik
        min_detection_confidence=MIN_DETECT_CONF
    )

    # [PERBAIKAN] Validasi kamera/file video dapat dibuka sebelum proses
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Gagal membuka video: {video_path}. Dilewati.")
        hands.close()
        return np.array([])

    # [PENINGKATAN] Tampilkan info FPS video asli untuk transparansi data
    original_fps  = cap.get(cv2.CAP_PROP_FPS)
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    effective_fps = original_fps / FRAME_SKIP if original_fps > 0 else "?"
    print(f"  Memproses: {os.path.basename(video_path)}")
    print(f"    FPS asli: {original_fps:.1f} | Total frame: {total_frames} | "
        f"FPS efektif (setelah skip): {effective_fps:.1f}")

    frame_data  = []
    frame_count = 0

    # ---- [BARU] Variabel untuk EMA Smoothing ----
    # alpha = 1.0 → tanpa smoothing, 0.1 → sangat lambat/halus
    # alpha = 0.7 memberikan keseimbangan antara responsif dan mulus
    prev_coords = None
    alpha = 0.7

    while True:
        # [EFISIENSI] grab() hanya mengambil frame tanpa men-decode (murah).
        grabbed = cap.grab()
        if not grabbed:
            break

        frame_count += 1
        # Hanya proses setiap frame ke-FRAME_SKIP (Sinkronisasi Temporal).
        # Frame yang di-skip cukup di-grab tanpa decode → tidak membuang waktu
        # men-decode 3 dari 4 frame. Frame yang dipakai persis sama (4,8,12,...)
        # sehingga output CSV identik dengan versi sebelumnya.
        if frame_count % FRAME_SKIP != 0:
            continue

        # [EFISIENSI] retrieve() men-decode HANYA frame yang benar-benar dipakai.
        ret, image = cap.retrieve()
        if not ret:
            break

        # Konversi BGR (OpenCV) -> RGB (MediaPipe)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results   = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # 1. Ekstraksi 21 koordinat (x, y, z) → shape: (21, 3)
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32
            )

            # 2. [BARU] EMA Smoothing — hilangkan micro-jitter kamera
            if prev_coords is None:
                smoothed_coords = coords
            else:
                smoothed_coords = alpha * coords + (1 - alpha) * prev_coords
            prev_coords = smoothed_coords

            # 3. NORMALISASI EGO-CENTRIC (Translational Invariance)
            # Geser titik nol ke pergelangan tangan agar posisi layar tidak berpengaruh
            wrist = smoothed_coords[0]
            translated_coords = smoothed_coords - wrist  # shape: (21, 3)

            # 4. [BARU] NORMALISASI SKALA (Scale Invariance)
            # Bagi dengan jarak Wrist (0,0,0) → MCP Jari Tengah (titik 9)
            # Rumus: d = sqrt(x² + y² + z²) = np.linalg.norm(titik_9)
            middle_finger_mcp = translated_coords[9]
            scale = np.linalg.norm(middle_finger_mcp)  # Jarak dari 0,0,0

            if scale > 1e-6:   # Hindari pembagian dengan nol
                normalized_coords = translated_coords / scale
            else:
                normalized_coords = translated_coords

            # Ratakan menjadi 1D: 21 * 3 = 63 fitur
            frame_data.append(normalized_coords.flatten())  # shape: (63,)

    cap.release()
    hands.close()

    # Validasi: pastikan cukup frame untuk membentuk minimal 1 sekuens
    if len(frame_data) < window_size:
        print(f"  [PERINGATAN] Hanya {len(frame_data)} frame tangan terdeteksi "
            f"(butuh minimal {window_size}). Video dilewati.")
        return np.array([])

    # ---- SLIDING WINDOW ----
    # Memecah seluruh frame menjadi potongan sekuens yang overlap
    data_array = np.array(frame_data)       # shape: (total_frame_valid, 63)
    sequences  = []
    for i in range(len(data_array) - window_size + 1):
        sequences.append(data_array[i : i + window_size])

    print(f"    => {len(sequences)} sekuens dihasilkan.")
    return np.array(sequences)              # shape: (N, window_size, 63)


# ========================================================================
# EKSEKUSI UTAMA
# ========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  FASE 1 - EKSTRAKSI & NORMALISASI DATA DIMULAI")
    print("=" * 60)
    print(f"  Folder  : {VIDEO_FOLDER}")
    print(f"  Window  : {WINDOW_SIZE} frame | Frame Skip: setiap frame ke-{FRAME_SKIP}")
    print(f"  Output  : {OUTPUT_CSV}")
    print("=" * 60)

    # Pastikan folder output ada
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # [PERBAIKAN] Tambahkan sorted() agar video diproses secara konsisten (bukan acak)
    if not os.path.isdir(VIDEO_FOLDER):
        print(f"[ERROR] Folder tidak ditemukan: {VIDEO_FOLDER}")
        exit(1)

    video_files = sorted([f for f in os.listdir(VIDEO_FOLDER) if f.endswith(".mp4")])

    if not video_files:
        print(f"[ERROR] Tidak ada file .mp4 ditemukan di folder: {VIDEO_FOLDER}")
        print("  Letakkan video rekaman gerakan normal di folder tersebut lalu jalankan ulang.")
        exit(1)

    print(f"\nTotal video ditemukan: {len(video_files)}")
    print("-" * 60)

    all_sequences = []
    skipped_count = 0

    for idx, video_file in enumerate(video_files, start=1):
        print(f"\n[{idx}/{len(video_files)}]", end=" ")
        seq = extract_and_normalize(
            os.path.join(VIDEO_FOLDER, video_file),
            window_size=WINDOW_SIZE
        )
        if seq.size > 0:
            all_sequences.append(seq)
        else:
            skipped_count += 1

    print("\n" + "-" * 60)

    if not all_sequences:
        print("\n[ERROR] Tidak ada sekuens valid yang berhasil diekstrak.")
        print("  Pastikan video tidak gelap/blur dan tangan terlihat jelas di kamera.")
        exit(1)

    # Gabungkan semua sekuens dari semua video
    # shape akhir: (Total_Sekuens, WINDOW_SIZE, 63)
    final_dataset = np.vstack(all_sequences)
    print(f"\nRingkasan Ekstraksi:")
    print(f"  Video berhasil    : {len(all_sequences)}")
    print(f"  Video dilewati    : {skipped_count}")
    print(f"  Total sekuens     : {final_dataset.shape[0]}")
    print(f"  Panjang sekuens   : {final_dataset.shape[1]} frame")
    print(f"  Fitur per frame   : {final_dataset.shape[2]} (21 landmark x 3 sumbu)")

    # CSV tidak mendukung array 3D → reshape ke 2D sebelum disimpan
    # shape: (Total_Sekuens, WINDOW_SIZE * 63) = (N, 1890)
    final_dataset_2d = final_dataset.reshape(final_dataset.shape[0], -1)
    pd.DataFrame(final_dataset_2d).to_csv(OUTPUT_CSV, index=False)

    print(f"\n[SUKSES] Dataset berhasil disimpan ke: {OUTPUT_CSV}")
    print(f"         Ukuran matriks 2D : {final_dataset_2d.shape}")
    print("\nLangkah selanjutnya:")
    print("  Jalankan: py fase2_training_lokal.py")
    print("=" * 60)
