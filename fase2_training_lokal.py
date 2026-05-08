"""
==========================================================================
FASE 2: PELATIHAN MODEL LSTM-AUTOENCODER — ADVANCED
Proyek Skripsi: Deteksi Anomali Gerakan Rehabilitasi Tangan
Versi: 2.0 Advanced
==========================================================================
PENINGKATAN:
- Logging terstruktur dengan timestamp
- Plot training history otomatis disimpan ke PNG
- Validasi threshold via distribusi statistik (histogram disimpan)
- Laporan evaluasi model lengkap (MSE stats per split)
- Arsitektur model dengan BatchNormalization
- Konfigurasi via dataclass
==========================================================================
"""

import logging
import os
import sys
import time

# Silence TensorFlow oneDNN warnings before importing anything else
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Training")


@dataclass
class TrainingConfig:
    csv_file:    str   = "data/processed_data/dataset_normal.csv"
    model_dir:   str   = "models"
    window_size: int   = 30
    n_features:  int   = 63
    batch_size:  int   = 32
    epochs:      int   = 150
    test_size:   float = 0.15
    val_split:   float = 0.10
    random_seed: int   = 42
    lr:          float = 0.001
    # Threshold: μ + k*σ
    # k=4 memberikan toleransi yang cukup untuk variasi gerakan manusia
    # dan menekan False Positive hingga nyaris 0%.
    threshold_k: float = 4.0


def build_model(cfg: TrainingConfig):
    """Bangun arsitektur LSTM-Autoencoder dengan BatchNormalization."""
    import tensorflow as tf
    LSTM = tf.keras.layers.LSTM
    BatchNormalization = tf.keras.layers.BatchNormalization
    Dense = tf.keras.layers.Dense
    Dropout = tf.keras.layers.Dropout
    Input = tf.keras.layers.Input
    RepeatVector = tf.keras.layers.RepeatVector
    TimeDistributed = tf.keras.layers.TimeDistributed
    Sequential = tf.keras.models.Sequential

    model = Sequential(name="LSTM_Autoencoder_Rehabilitasi_v2")

    # ---- INPUT (cara baru Keras — menghilangkan UserWarning) ----
    model.add(Input(shape=(cfg.window_size, cfg.n_features)))

    # ---- ENCODER ----
    model.add(LSTM(128, activation='tanh', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(64, activation='tanh', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='tanh', return_sequences=False))
    model.add(BatchNormalization())

    # ---- BOTTLENECK ----
    model.add(RepeatVector(cfg.window_size))

    # ---- DECODER ----
    model.add(LSTM(32, activation='tanh', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(64, activation='tanh', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='tanh', return_sequences=True))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(cfg.n_features)))

    Adam = tf.keras.optimizers.Adam
    model.compile(
        optimizer=Adam(learning_rate=cfg.lr),
        loss='mse',
        metrics=['mae']
    )
    return model


def save_training_plot(history, model_dir: str):
    """Simpan grafik loss/mae ke PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Training History — LSTM-Autoencoder Rehabilitasi", fontsize=14)

        axes[0].plot(history.history['loss'], label='Train Loss', color='#2196F3')
        axes[0].plot(history.history['val_loss'], label='Val Loss', color='#FF5722', linestyle='--')
        axes[0].set_title('Loss (MSE)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history.history['mae'], label='Train MAE', color='#4CAF50')
        axes[1].plot(history.history['val_mae'], label='Val MAE', color='#FF9800', linestyle='--')
        axes[1].set_title('MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = Path(model_dir) / "training_history.png"
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()
        log.info(f"Plot training disimpan: {out_path}")
    except ImportError:
        log.warning("matplotlib tidak tersedia, skip plot training.")


def save_threshold_histogram(mse_train: np.ndarray, threshold: float, model_dir: str):
    """Simpan histogram distribusi MSE training + garis threshold."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(mse_train, bins=50, color='#2196F3', alpha=0.7, edgecolor='white', label='MSE Training')
        ax.axvline(threshold, color='#FF5722', linewidth=2, linestyle='--',
                label=f'Threshold = {threshold:.6f}')
        ax.set_title('Distribusi MSE Training & Threshold Anomali', fontsize=13)
        ax.set_xlabel('MSE')
        ax.set_ylabel('Frekuensi')
        ax.legend()
        ax.grid(True, alpha=0.3)

        out_path = Path(model_dir) / "threshold_distribution.png"
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()
        log.info(f"Histogram threshold disimpan: {out_path}")
    except ImportError:
        log.warning("matplotlib tidak tersedia, skip histogram.")


def evaluate_model(model, X_test: np.ndarray, threshold: float) -> dict:
    """Evaluasi model pada data test dan return stats."""
    X_pred = model.predict(X_test, verbose=0)
    mse_per_sample = np.mean(np.power(X_test - X_pred, 2), axis=(1, 2))
    anomaly_pct    = (mse_per_sample >= threshold).mean() * 100

    return {
        "test_mse_mean":  float(np.mean(mse_per_sample)),
        "test_mse_std":   float(np.std(mse_per_sample)),
        "test_mse_max":   float(np.max(mse_per_sample)),
        "anomaly_pct":    float(anomaly_pct),
    }


def main():
    cfg = TrainingConfig()

    log.info("=" * 60)
    log.info("  FASE 2 ADVANCED: PELATIHAN MODEL AI LOKAL")
    log.info("=" * 60)
    log.info(f"  CSV      : {cfg.csv_file}")
    log.info(f"  Window   : {cfg.window_size}  |  Features: {cfg.n_features}")
    log.info(f"  Epochs   : {cfg.epochs}  |  Batch: {cfg.batch_size}")
    log.info(f"  Threshold k: mu + {cfg.threshold_k}*sigma")
    log.info("=" * 60)

    # ---- 1. Load Data ----
    if not Path(cfg.csv_file).exists():
        log.error(f"Dataset tidak ditemukan: {cfg.csv_file}")
        sys.exit(1)

    log.info("1. Memuat dataset...")
    df = pd.read_csv(cfg.csv_file)
    data_3d = df.values.reshape(-1, cfg.window_size, cfg.n_features).astype(np.float32)
    log.info(f"   Shape dataset: {data_3d.shape}  ({data_3d.shape[0]} sekuens)")

    # ---- 2. Normalisasi ----
    log.info("2. Normalisasi MinMaxScaler...")
    data_2d        = data_3d.reshape(-1, cfg.n_features)
    scaler         = MinMaxScaler(feature_range=(0, 1))
    data_2d_scaled = scaler.fit_transform(data_2d)
    data_scaled    = data_2d_scaled.reshape(-1, cfg.window_size, cfg.n_features)

    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    scaler_path = Path(cfg.model_dir) / "scaler_Rehabilitasi.pkl"
    joblib.dump(scaler, scaler_path)
    log.info(f"   Scaler disimpan: {scaler_path}")

    # ---- 3. Split ----
    log.info("3. Split data Train/Test...")
    X_train, X_test = train_test_split(
        data_scaled, test_size=cfg.test_size,
        random_state=cfg.random_seed, shuffle=True
    )
    log.info(f"   Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ---- 4. Build Model ----
    import tensorflow as tf
    EarlyStopping = tf.keras.callbacks.EarlyStopping
    ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
    ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau

    log.info("4. Membangun arsitektur LSTM-Autoencoder Advanced...")
    model = build_model(cfg)
    model.summary(print_fn=lambda x: log.info(f"   {x}"))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20,
                    restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(Path(cfg.model_dir) / "otak_Rehabilitasi_best.keras"),
                        monitor='val_loss', save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                        patience=8, min_lr=1e-7, verbose=1),
    ]

    # ---- 5. Training ----
    log.info("\n=== MULAI TRAINING ===")
    t_start = time.time()
    history = model.fit(
        X_train, X_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_split=cfg.val_split,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )
    elapsed = time.time() - t_start
    log.info(f"\nTraining selesai dalam {elapsed/60:.1f} menit.")

    # ---- 6. Simpan plot ----
    save_training_plot(history, cfg.model_dir)

    # ---- 7. Hitung Threshold ----
    log.info("\n5. Menghitung threshold statistik dari data training...")
    X_pred_train  = model.predict(X_train, verbose=0)
    mse_train     = np.mean(np.power(X_train - X_pred_train, 2), axis=(1, 2))
    mu_mse        = float(np.mean(mse_train))
    sigma_mse     = float(np.std(mse_train))
    THRESHOLD     = mu_mse + (cfg.threshold_k * sigma_mse)

    log.info(f"   mu (mean MSE) = {mu_mse:.8f}")
    log.info(f"   sigma (std  MSE) = {sigma_mse:.8f}")
    log.info(f"   Threshold (mu + {cfg.threshold_k}*sigma) = {THRESHOLD:.8f}")

    save_threshold_histogram(mse_train, THRESHOLD, cfg.model_dir)

    # ---- 8. Evaluasi Test Set ----
    log.info("\n6. Evaluasi pada data test...")
    eval_stats = evaluate_model(model, X_test, THRESHOLD)
    log.info(f"   Test MSE mean : {eval_stats['test_mse_mean']:.8f}")
    log.info(f"   Test MSE std  : {eval_stats['test_mse_std']:.8f}")
    log.info(f"   Test MSE max  : {eval_stats['test_mse_max']:.8f}")
    log.info(f"   % data test di atas threshold: {eval_stats['anomaly_pct']:.2f}%")
    log.info("   (Diharapkan mendekati 0% karena semua data test = NORMAL)")

    # ---- 9. Simpan Model & Threshold ----
    model_path = Path(cfg.model_dir) / "otak_Rehabilitasi.keras"
    model.save(str(model_path))
    log.info(f"\nModel disimpan: {model_path}")

    thr_path = Path(cfg.model_dir) / "threshold_value.txt"
    with open(thr_path, 'w') as f:
        f.write(f"THRESHOLD = {THRESHOLD:.10f}\n")
        f.write(f"MU = {mu_mse:.10f}\n")
        f.write(f"SIGMA = {sigma_mse:.10f}\n")
        f.write(f"K = {cfg.threshold_k}\n")
        f.write(f"TRAIN_SAMPLES = {len(X_train)}\n")
        f.write(f"TEST_SAMPLES = {len(X_test)}\n")
        f.write(f"TEST_MSE_MEAN = {eval_stats['test_mse_mean']:.10f}\n")
        f.write(f"GENERATED_AT = {datetime.now().isoformat()}\n")
    log.info(f"Threshold disimpan: {thr_path}")

    log.info("\n" + "=" * 60)
    log.info("  TRAINING SELESAI & SUKSES!")
    log.info("=" * 60)
    log.info(f"  Threshold Final : {THRESHOLD:.8f}")
    log.info(f"  Model           : {model_path}")
    log.info(f"  Scaler          : {scaler_path}")
    log.info(f"  Plot            : models/training_history.png")
    log.info(f"  Histogram       : models/threshold_distribution.png")
    log.info("  Sekarang jalankan fase3_aplikasi_realtime.py!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
