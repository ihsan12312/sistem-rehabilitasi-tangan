import cv2
import os
import time

# Konfigurasi
OUTPUT_DIR = "data/raw_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

print("="*50)
print("ALAT PEREKAM DATASET NORMAL - Rehabilitasi")
print("="*50)
print("Instruksi:")
print("1. Pastikan seluruh tangan masuk dan menghadap ke kamera.")
print("2. Tekan 'R' untuk MULAI merekam (tulisan merah akan muncul).")
print("3. Tekan 'S' untuk BERHENTI merekam (video otomatis tersimpan).")
print("4. Tekan 'Q' untuk KELUAR dari program.")
print("="*50)

# Cari nomor urut terakhir
existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("video_") and f.endswith(".mp4")]
numbers = [int(f.split("_")[1].split(".")[0]) for f in existing_files]
next_num = max(numbers) + 1 if numbers else 1

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec .mp4
out = None
is_recording = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Webcam gagal dibaca")
        break
        
    frame = cv2.flip(frame, 1) # Mirror
    display_frame = frame.copy()
    
    if is_recording:
        # Tulis ke file
        out.write(frame)
        # Indikator Rekam
        cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
        cv2.putText(display_frame, f"MEREKAM video_{next_num}.mp4 (Tekan S untuk stop)", 
                    (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(display_frame, "Tekan 'R' untuk Mulai Rekam", 
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
    cv2.imshow("Perekam Dataset Rehabilitasi", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('r') and not is_recording:
        filename = os.path.join(OUTPUT_DIR, f"video_{next_num:02d}.mp4")
        out = cv2.VideoWriter(filename, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
        is_recording = True
        print(f"Mulai merekam -> {filename}")
        
    elif key == ord('s') and is_recording:
        is_recording = False
        out.release()
        print(f"Tersimpan! -> video_{next_num:02d}.mp4")
        next_num += 1
        
    elif key == ord('q'):
        if is_recording:
            out.release()
            print(f"Tersimpan! -> video_{next_num:02d}.mp4")
        break

cap.release()
cv2.destroyAllWindows()
print("Selesai. Kamera ditutup.")

