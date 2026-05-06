"""
==========================================================================
ADAPTER DATASET PUBLIK -> FORMAT Rehabilitasi
Proyek Skripsi: Deteksi Anomali Gerakan Rehabilitasi Tangan
==========================================================================
TUJUAN : Mengkonversi dataset DynTherapy (33 pose landmarks MediaPipe)
        menjadi format dataset_normal.csv yang dibutuhkan Fase 2.
        
Dataset DynTherapy berisi gerakan fisioterapi nyata dari jurnal akademik:
"DynTherapy: Physical Therapy Exercises Dataset" (Mendeley, 2024)
DOI: 10.17632/hghdm99rwg.1

CARA PAKAI:
1. Pastikan file fitness_poses_dyntherapy.csv ada di data/raw_datasets/
2. Jalankan: python adapter_dataset_publik.py
3. Output: data/processed_data/dataset_normal.csv (siap untuk Fase 2)
==========================================================================
"""

import pandas as pd
import numpy as np
import os

# -----------------------------------------------------------------------
# KONFIGURASI
# -----------------------------------------------------------------------
INPUT_CSV   = "data/raw_datasets/fitness_poses_dyntherapy.csv"
OUTPUT_CSV  = "data/processed_data/dataset_normal.csv"
WINDOW_SIZE = 30       # Harus sama dengan Fase 1 dan Fase 2
N_FEATURES  = 63       # 21 titik * 3 sumbu (x,y,z) — JANGAN UBAH

# -----------------------------------------------------------------------
# PEMETAAN LANDMARK MEDIAPIPE POSE -> HAND PROXY
# -----------------------------------------------------------------------
# DynTherapy pakai MediaPipe POSE (33 titik, full body).
# Kita ambil 11 titik yang relevan untuk ekstremitas atas (tangan & lengan)
# lalu duplikasi untuk mengisi 21 titik format MediaPipe HANDS.
#
# Index pose landmarks yang relevan:
# 11=bahu kiri, 12=bahu kanan
# 13=siku kiri, 14=siku kanan
# 15=pergelangan kiri, 16=pergelangan kanan
# 17=jempol kiri, 18=jempol kanan
# 19=jari telunjuk kiri, 20=jari telunjuk kanan
# 21=jari kelingking kiri, 22=jari kelingking kanan
UPPER_LIMB_INDICES = [12, 14, 16, 18, 20, 22]  # Gunakan sisi kanan

# Mapping ke 21 "hand proxy" landmarks (duplikasi titik untuk mengisi 21 slot)
# Urutan: pergelangan(0), jempol(1-4), telunjuk(5-8), tengah(9-12),
#         manis(13-16), kelingking(17-20)
# Kita petakan dari 6 titik pose ke 21 titik proxy:
PROXY_MAP = {
    0: 2,   # pergelangan -> pergelangan kanan (index 16 di pose = indeks ke-2 di upper_limb)
    1: 3,   # ibu jari MCP -> jempol kanan (index 18 di pose = indeks ke-3)
    2: 3,
    3: 3,
    4: 3,
    5: 4,   # jari telunjuk -> telunjuk kanan (index 20 = indeks ke-4)
    6: 4,
    7: 4,
    8: 4,
    9: 5,   # jari tengah -> kelingking (index 22 = indeks ke-5, closest available)
    10: 5,
    11: 5,
    12: 5,
    13: 5,
    14: 5,
    15: 5,
    16: 5,
    17: 5,
    18: 5,
    19: 5,
    20: 5,
}

# -----------------------------------------------------------------------

def parse_row_to_pose_landmarks(row_values):
    """
    Mengurai satu baris data DynTherapy menjadi array 33 pose landmarks.
    Format baris: [filename, label, x0,y0,z0, x1,y1,z1, ..., x32,y32,z32]
    
    Return: np.array shape (33, 3) atau None jika gagal
    """
    coords_raw = row_values[2:]  # Skip kolom filename dan label
    if len(coords_raw) < 33 * 3:
        return None
    coords = np.array(coords_raw[:33*3], dtype=np.float32).reshape(33, 3)
    return coords


def pose_to_hand_proxy(pose_coords):
    """
    Mengkonversi 33 pose landmarks ke 21 hand-proxy landmarks.
    Mengambil titik-titik ekstremitas atas kanan dan memetakannya ke format
    21 landmark MediaPipe Hands.
    
    Parameter:
        pose_coords: np.array shape (33, 3)
    
    Return:
        np.array shape (21, 3)
    """
    # Ambil titik-titik lengan atas kanan (6 titik)
    upper_limb = pose_coords[UPPER_LIMB_INDICES]  # shape: (6, 3)
    
    # Petakan ke 21 titik proxy
    hand_proxy = np.zeros((21, 3), dtype=np.float32)
    for hand_idx, upper_idx in PROXY_MAP.items():
        hand_proxy[hand_idx] = upper_limb[upper_idx]
    
    return hand_proxy


def normalize_ego_centric(hand_coords):
    """
    Normalisasi ego-centric: jadikan titik 0 (pergelangan tangan) sebagai pusat.
    IDENTIK dengan normalisasi di fase1_data_factory.py.
    
    Parameter:
        hand_coords: np.array shape (21, 3)
    
    Return:
        np.array shape (21, 3) yang sudah dinormalisasi
    """
    return hand_coords - hand_coords[0]


def build_sequences(frame_list, window_size):
    """
    Membuat sekuens sliding window dari list frame.
    
    Parameter:
        frame_list  : list of np.array (63,)
        window_size : int
    
    Return:
        np.array shape (N, window_size, 63)
    """
    data = np.array(frame_list)  # shape: (total_frame, 63)
    if len(data) < window_size:
        return np.array([])
    
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i: i + window_size])
    
    return np.array(sequences)


# ========================================================================
# EKSEKUSI UTAMA
# ========================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  ADAPTER DATASET PUBLIK -> FORMAT Rehabilitasi")
    print("=" * 65)
    
    # Cek file input ada
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] File tidak ditemukan: {INPUT_CSV}")
        print("  Pastikan fitness_poses_dyntherapy.csv ada di data/raw_datasets/")
        exit()

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Load dataset
    print(f"\nMemuat dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, header=None)
    print(f"Total baris ditemukan : {len(df)}")
    print(f"Total kolom           : {len(df.columns)}")
    
    # Tampilkan distribusi label
    labels = df.iloc[:, 1]
    print(f"\nDistribusi kelas gerakan:")
    for label, count in labels.value_counts().items():
        print(f"  {label:35s}: {count} baris")

    # Pilih gerakan yang PALING relevan untuk rehabilitasi tangan/bahu
    GERAKAN_NORMAL = [
        'shoulder_press_up',
        'shoulder_press_down',
        'shoulder_flexion_up',
        'lateral_raises_up',
        'lateral_raises_down',
        'knee_raises_up',
        'knee_raises_down',
    ]
    
    print(f"\nMenggunakan {len(GERAKAN_NORMAL)} kelas gerakan sebagai 'NORMAL':")
    for g in GERAKAN_NORMAL:
        print(f"  - {g}")
    
    df_filtered = df[df.iloc[:, 1].isin(GERAKAN_NORMAL)].reset_index(drop=True)
    print(f"\nTotal baris setelah filter: {len(df_filtered)}")

    # Proses setiap baris menjadi frame landmark
    print(f"\nMemproses {len(df_filtered)} baris menjadi sekuens...")
    all_sequences = []
    skipped = 0
    
    # Proses per kelompok gerakan agar sekuens tidak campur label berbeda
    for label in GERAKAN_NORMAL:
        df_label = df_filtered[df_filtered.iloc[:, 1] == label].reset_index(drop=True)
        if len(df_label) == 0:
            continue
        
        frame_list = []
        for idx, row in df_label.iterrows():
            pose_coords = parse_row_to_pose_landmarks(row.values)
            if pose_coords is None:
                skipped += 1
                continue
            
            # Konversi pose 33 titik -> tangan proxy 21 titik
            hand_proxy = pose_to_hand_proxy(pose_coords)
            
            # Normalisasi ego-centric (identik dengan fase1)
            normalized = normalize_ego_centric(hand_proxy)
            
            # Ratakan: (21, 3) -> (63,)
            frame_list.append(normalized.flatten())
        
        # Buat sekuens sliding window untuk label ini
        seqs = build_sequences(frame_list, WINDOW_SIZE)
        if seqs.size > 0:
            all_sequences.append(seqs)
            print(f"  [{label}]: {len(frame_list)} frame -> {len(seqs)} sekuens")
        else:
            print(f"  [{label}]: terlalu sedikit frame, dilewati")

    if not all_sequences:
        print("\n[ERROR] Tidak ada sekuens valid yang terbentuk.")
        exit()

    # Gabungkan semua sekuens
    final_dataset = np.vstack(all_sequences)  # shape: (N, WINDOW_SIZE, 63)
    
    print("-" * 65)
    print(f"\nTotal sekuens akhir     : {final_dataset.shape[0]}")
    print(f"Timestep per sekuens    : {final_dataset.shape[1]}")
    print(f"Fitur per timestep      : {final_dataset.shape[2]}")
    print(f"Baris dilewati          : {skipped}")
    
    # Simpan ke CSV (reshape 3D -> 2D)
    final_2d = final_dataset.reshape(final_dataset.shape[0], -1)
    pd.DataFrame(final_2d).to_csv(OUTPUT_CSV, index=False)
    
    size_kb = os.path.getsize(OUTPUT_CSV) / 1024
    print(f"\n[SUKSES] Dataset disimpan ke: {OUTPUT_CSV}")
    print(f"         Ukuran file         : {size_kb:.1f} KB")
    print(f"         Shape matriks 2D    : {final_2d.shape}")
    print("\nLangkah selanjutnya:")
    print("  Upload dataset_normal.csv ke Google Colab")
    print("  Jalankan fase2_colab_training.py")
    print("=" * 65)

