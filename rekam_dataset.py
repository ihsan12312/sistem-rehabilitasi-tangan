"""
==========================================================================
ALAT PEREKAM DATASET NORMAL
Proyek Skripsi: Deteksi Anomali Gerakan Rehabilitasi Tangan
==========================================================================
CARA PAKAI:
  1. Pastikan seluruh tangan masuk dan menghadap ke kamera.
  2. Tekan 'R' untuk MULAI merekam (lingkaran merah akan muncul).
  3. Tekan 'S' untuk BERHENTI merekam (video otomatis tersimpan).
  4. Tekan 'Q' untuk KELUAR dari program.

CATATAN: Rekam minimal 5 detik per video agar dataset valid untuk Fase 1.
==========================================================================
"""

import cv2
import os
import time

# -----------------------------------------------------------------------
# KONFIGURASI
# -----------------------------------------------------------------------
OUTPUT_DIR   = "data/raw_videos"
FPS          = 30
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
MIN_DURATION_SEC = 5    # Minimum durasi rekaman agar video valid di Fase 1
# -----------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("  ALAT PEREKAM DATASET NORMAL - Rehabilitasi")
print("=" * 55)
print("  Instruksi:")
print("  1. Pastikan tangan terlihat jelas di kamera.")
print("  2. Tekan 'R' untuk MULAI merekam.")
print("  3. Tekan 'S' untuk BERHENTI (video tersimpan).")
print("  4. Tekan 'Q' untuk KELUAR.")
print(f"  PENTING: Rekam minimal {MIN_DURATION_SEC} detik per video!")
print("=" * 55)

# [PERBAIKAN] Cari nomor urut terakhir dengan try/except agar
# tidak crash jika ada file dengan nama tidak standar di folder
existing_files = [f for f in os.listdir(OUTPUT_DIR)
                  if f.startswith("video_") and f.endswith(".mp4")]
numbers = []
for f in existing_files:
    try:
        num = int(f.split("_")[1].split(".")[0])
        numbers.append(num)
    except (IndexError, ValueError):
        pass  # Lewati file dengan nama tidak standar

next_num = max(numbers) + 1 if numbers else 1
print(f"\nVideo berikutnya akan disimpan sebagai: video_{next_num:02d}.mp4")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Webcam tidak dapat dibuka! Pastikan kamera tersambung.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
out          = None
is_recording = False
record_start = 0.0
frame_count_rec = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Webcam gagal dibaca.")
        break

    frame        = cv2.flip(frame, 1)   # Mirror (efek cermin)
    display      = frame.copy()
    h, w         = display.shape[:2]

    if is_recording:
        out.write(frame)
        frame_count_rec += 1

        # Hitung durasi rekaman
        elapsed     = time.time() - record_start
        dur_str     = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        min_reached = elapsed >= MIN_DURATION_SEC

        # Indikator rekam: lingkaran merah
        cv2.circle(display, (30, 40), 12, (0, 0, 255), -1)

        # Teks status rekam
        status_text = f"MEREKAM video_{next_num:02d}.mp4  |  {dur_str}"
        cv2.putText(display, status_text,
                    (52, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # [PENINGKATAN] Tampilkan peringatan jika durasi belum cukup
        if not min_reached:
            sisa = MIN_DURATION_SEC - elapsed
            warn = f"Minimal {MIN_DURATION_SEC}s! Sisa: {sisa:.1f}s (terus rekam...)"
            cv2.putText(display, warn,
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        else:
            cv2.putText(display, "Durasi OK! Tekan 'S' untuk stop.",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 2)
    else:
        cv2.putText(display, f"Siap  |  Video berikutnya: video_{next_num:02d}.mp4",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 2)
        cv2.putText(display, "Tekan 'R' untuk Mulai Rekam",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("Perekam Dataset Rehabilitasi  |  [R] Rekam  [S] Stop  [Q] Keluar",
               display)

    key = cv2.waitKey(1) & 0xFF

    # Mulai rekam
    if key == ord('r') and not is_recording:
        filename        = os.path.join(OUTPUT_DIR, f"video_{next_num:02d}.mp4")
        out             = cv2.VideoWriter(filename, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
        is_recording    = True
        record_start    = time.time()
        frame_count_rec = 0
        print(f"\nMulai merekam -> {filename}")

    # Stop rekam
    elif key == ord('s') and is_recording:
        elapsed = time.time() - record_start
        if elapsed < MIN_DURATION_SEC:
            # [PERBAIKAN] Cegah video terlalu pendek — peringatan di terminal
            print(f"  [PERINGATAN] Durasi hanya {elapsed:.1f}s (minimal {MIN_DURATION_SEC}s).")
            print(f"  Video video_{next_num:02d}.mp4 disimpan tapi mungkin terlalu pendek untuk Fase 1.")
        else:
            print(f"  Durasi: {elapsed:.1f}s ({frame_count_rec} frame)")

        is_recording = False
        out.release()
        print(f"Tersimpan! -> video_{next_num:02d}.mp4")
        next_num += 1
        print(f"Video berikutnya: video_{next_num:02d}.mp4")

    # Keluar
    elif key in (ord('q'), 27):
        if is_recording:
            out.release()
            elapsed = time.time() - record_start
            print(f"Tersimpan (saat keluar)! -> video_{next_num:02d}.mp4 ({elapsed:.1f}s)")
        break

cap.release()
cv2.destroyAllWindows()
print("\nSelesai. Kamera ditutup.")
print(f"Total file di folder: {len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.mp4')])}")
