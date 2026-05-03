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
==========================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os


# -----------------------------------------------------------------------
# KONFIGURASI (Sesuaikan jika perlu)
# -----------------------------------------------------------------------
WINDOW_SIZE    = 30          # Jumlah frame per sekuens (sliding window)
VIDEO_FOLDER   = "data/raw_videos/"
OUTPUT_CSV     = "data/processed_data/dataset_normal.csv"
MIN_DETECT_CONF = 0.7        # Minimum confidence untuk deteksi tangan
# -----------------------------------------------------------------------


def extract_and_normalize(video_path, window_size=30):
    """
    Membaca satu file video, mengekstrak landmark tangan per-frame
    dengan MediaPipe, menormalisasi secara ego-centric (pergelangan
    tangan sebagai titik pusat 0,0,0), lalu menghasilkan sekuens
    menggunakan sliding window.

    Parameter:
        video_path  (str)  : Path ke file video MP4.
        window_size (int)  : Panjang sekuens (jumlah frame per window).

    Return:
        np.ndarray shape (N, window_size, 63)
            N = jumlah sekuens yang terbentuk
            63 = 21 titik landmark * 3 sumbu (x, y, z)
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECT_CONF
    )

    cap = cv2.VideoCapture(video_path)
    frame_data = []

    print(f"  Memproses: {video_path}")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Konversi BGR (OpenCV) -> RGB (MediaPipe)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Ekstraksi 21 koordinat (x, y, z)
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            )  # shape: (21, 3)

            # ---- NORMALISASI EGO-CENTRIC ----
            # Kurangi semua titik dengan koordinat pergelangan tangan (index 0)
            # Tujuan: gerakan yang sama di posisi layar berbeda tetap identik
            normalized_coords = coords - coords[0]  # shape: (21, 3)

            # Ratakan menjadi 1D: 21 * 3 = 63 fitur
            frame_data.append(normalized_coords.flatten())  # shape: (63,)

    cap.release()
    hands.close()

    if len(frame_data) < window_size:
        print(f"  [PERINGATAN] Video terlalu pendek ({len(frame_data)} frame), dilewati.")
        return np.array([])

    # ---- SLIDING WINDOW ----
    # Memecah seluruh frame menjadi potongan sekuens overlap
    data_array = np.array(frame_data)  # shape: (total_frame, 63)
    sequences = []
    for i in range(len(data_array) - window_size):
        sequences.append(data_array[i: i + window_size])  # shape: (window_size, 63)

    print(f"  => {len(sequences)} sekuens dihasilkan dari video ini.")
    return np.array(sequences)  # shape: (N, window_size, 63)


# ========================================================================
# EKSEKUSI UTAMA
# ========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  FASE 1 - EKSTRAKSI & NORMALISASI DATA DIMULAI")
    print("=" * 60)

    # Pastikan folder output ada
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    all_sequences = []
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(".mp4")]

    if not video_files:
        print(f"[ERROR] Tidak ada file .mp4 ditemukan di folder: {VIDEO_FOLDER}")
        print("  Letakkan video rekaman gerakan normal di folder tersebut lalu jalankan ulang.")
        exit()

    print(f"\nTotal video ditemukan: {len(video_files)}")
    print("-" * 60)

    for video_file in video_files:
        seq = extract_and_normalize(
            os.path.join(VIDEO_FOLDER, video_file),
            window_size=WINDOW_SIZE
        )
        if seq.size > 0:
            all_sequences.append(seq)

    if not all_sequences:
        print("\n[ERROR] Tidak ada sekuens valid yang berhasil diekstrak. Periksa video Anda.")
        exit()

    # Gabungkan semua sekuens dari semua video
    # shape akhir: (Total_Sekuens, WINDOW_SIZE, 63)
    final_dataset = np.vstack(all_sequences)
    print("-" * 60)
    print(f"\nTotal sekuens gabungan  : {final_dataset.shape[0]}")
    print(f"Panjang tiap sekuens    : {final_dataset.shape[1]} frame")
    print(f"Jumlah fitur per frame  : {final_dataset.shape[2]} (21 landmark x 3 sumbu)")

    # CSV tidak mendukung array 3D -> reshape ke 2D sebelum disimpan
    # shape: (Total_Sekuens, WINDOW_SIZE * 63) = (N, 1890)
    final_dataset_2d = final_dataset.reshape(final_dataset.shape[0], -1)
    pd.DataFrame(final_dataset_2d).to_csv(OUTPUT_CSV, index=False)

    print(f"\n[SUKSES] Dataset berhasil disimpan ke: {OUTPUT_CSV}")
    print(f"         Ukuran matriks 2D : {final_dataset_2d.shape}")
    print("\nLangkah selanjutnya:")
    print("  Unggah file CSV ini ke Google Colab dan jalankan fase2_colab_training.ipynb")
    print("=" * 60)

