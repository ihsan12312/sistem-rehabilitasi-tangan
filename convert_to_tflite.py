"""
==========================================================================
SCRIPT KONVERSI MODEL: Keras (.keras) → TFLite (.tflite)
Proyek Skripsi: Deteksi Anomali Gerakan Rehabilitasi Tangan
==========================================================================
Jalankan SEKALI saja sebelum menjalankan fase3_aplikasi_realtime.py:
    python convert_to_tflite.py

Hasil: models/otak_Rehabilitasi.tflite
Manfaat: Inferensi di CPU 3-5x lebih cepat dibanding model .keras biasa
==========================================================================
"""

import os
import sys
import time
from pathlib import Path

# Suppress TF noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("=" * 60)
print("  KONVERSI MODEL: Keras → TFLite")
print("=" * 60)

# --- Path ---
KERAS_PATH  = Path("models/otak_Rehabilitasi.keras")
TFLITE_PATH = Path("models/otak_Rehabilitasi.tflite")

# Validasi file input ada
if not KERAS_PATH.exists():
    print(f"\n[ERROR] File tidak ditemukan: {KERAS_PATH}")
    print("Pastikan kamu menjalankan script ini dari folder Project_Rehabilitasi/")
    sys.exit(1)

print(f"\n[1/3] Memuat model Keras dari: {KERAS_PATH}")
print("      (ini mungkin butuh beberapa detik...)")

try:
    import tensorflow as tf
    print(f"      TensorFlow versi: {tf.__version__}")
except ImportError:
    print("[ERROR] TensorFlow tidak terinstall. Jalankan: pip install tensorflow")
    sys.exit(1)

t0 = time.time()
try:
    model = tf.keras.models.load_model(str(KERAS_PATH), compile=False)
    print(f"      Model berhasil dimuat! ({time.time()-t0:.1f} detik)")
    print(f"      Input shape  : {model.input_shape}")
    print(f"      Output shape : {model.output_shape}")
except Exception as e:
    print(f"[ERROR] Gagal memuat model: {e}")
    sys.exit(1)

print(f"\n[2/3] Mengonversi ke TFLite...")
t1 = time.time()
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Aktifkan optimasi standar (quantization ringan, tidak kehilangan akurasi)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Target untuk CPU standar — tidak butuh hardware khusus
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Operasi TFLite standar
        tf.lite.OpsSet.SELECT_TF_OPS,    # Fallback ke TF ops jika perlu (LSTM)
    ]

    tflite_model = converter.convert()
    print(f"      Konversi selesai! ({time.time()-t1:.1f} detik)")
except Exception as e:
    print(f"[ERROR] Gagal konversi: {e}")
    print("\nCoba konversi tanpa optimasi...")
    try:
        converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter2.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        tflite_model = converter2.convert()
        print("      Konversi tanpa optimasi berhasil!")
    except Exception as e2:
        print(f"[ERROR] Konversi gagal total: {e2}")
        sys.exit(1)

print(f"\n[3/3] Menyimpan file TFLite...")
try:
    TFLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    size_keras  = KERAS_PATH.stat().st_size  / (1024 * 1024)
    size_tflite = TFLITE_PATH.stat().st_size / (1024 * 1024)

    print(f"      Disimpan ke  : {TFLITE_PATH}")
    print(f"      Ukuran .keras  : {size_keras:.2f} MB")
    print(f"      Ukuran .tflite : {size_tflite:.2f} MB")
    print(f"      Kompresi       : {(1 - size_tflite/size_keras)*100:.1f}% lebih kecil")
except Exception as e:
    print(f"[ERROR] Gagal menyimpan file: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("  KONVERSI BERHASIL!")
print(f"  Total waktu: {time.time()-t0:.1f} detik")
print("=" * 60)
print("\nLangkah selanjutnya:")
print("  Jalankan aplikasi real-time seperti biasa:")
print("  python fase3_aplikasi_realtime.py")
print("\n  Sistem akan otomatis menggunakan model TFLite yang lebih cepat.")
print("=" * 60)
