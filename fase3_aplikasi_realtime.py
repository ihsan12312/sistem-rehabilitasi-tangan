"""
==========================================================================
 FASE 3: APLIKASI REAL-TIME "Rehabilitasi" - ADVANCED DEPLOYMENT
 Proyek Skripsi: Deteksi Anomali Gerakan Rehabilitasi Tangan
 Versi: 3.0 Advanced | Menggunakan LSTM-Autoencoder + MediaPipe
==========================================================================
 FITUR ADVANCED:
   - Arsitektur OOP penuh (class RehabilitasiSystem)
   - Smoothing prediksi (anti false-alarm)
   - Minimum anomaly streak detection
   - Real-time MSE graph di layar
   - Panel statistik sesi (skor, FPS, durasi, anomali count)
   - Auto-save session log ke CSV
   - Graceful shutdown (Ctrl+C aman)
   - Snapshot foto (tekan 'S')
   - Reset sesi (tekan 'R')
   - Validasi model otomatis saat startup
==========================================================================
 KONTROL:
   Q = Keluar   |   S = Simpan Snapshot   |   R = Reset Sesi
==========================================================================
"""

# --- Environment fix HARUS paling pertama ---
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import csv
import logging
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import joblib
import mediapipe as mp
import numpy as np

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Rehabilitasi")


# ============================================================
# KONFIGURASI SISTEM (Dataclass — mudah diubah)
# ============================================================
@dataclass
class SystemConfig:
    model_path:       str   = "models/otak_Rehabilitasi.keras"
    scaler_path:      str   = "models/scaler_Rehabilitasi.pkl"
    threshold_path:   str   = "models/threshold_value.txt"
    session_log_dir:  str   = "data/session_logs"
    snapshot_dir:     str   = "data/snapshots"

    window_size:      int   = 30          # HARUS sama dengan saat training
    n_features:       int   = 63          # 21 landmark × 3 sumbu
    min_detect_conf:  float = 0.7
    min_track_conf:   float = 0.5

    # Anti false-alarm: anomali dilaporkan setelah N frame berturut-turut
    anomaly_streak_needed: int = 5

    # Smoothing: rata-rata dari N prediksi MSE terakhir
    mse_smooth_window: int = 8

    # Tampilan MSE graph
    mse_graph_history: int = 90          # Jumlah titik di graph


# ============================================================
# CLASS UTAMA
# ============================================================
class RehabilitasiSystem:
    """
    Sistem deteksi anomali rehabilitasi tangan berbasis LSTM-Autoencoder.
    Menggabungkan MediaPipe untuk pose estimation dan real-time visualization.
    """

    def __init__(self, config: SystemConfig):
        self.cfg = config
        self._running = False

        # AI Components
        self.model    = None
        self.scaler   = None
        self.threshold: float = 0.015

        # MediaPipe
        self.mp_hands    = mp.solutions.hands
        self.mp_drawing  = mp.solutions.drawing_utils
        self.mp_styles   = mp.solutions.drawing_styles
        self.hands       = None

        # Buffer sekuens landmark
        self.seq_buffer = deque(maxlen=config.window_size)

        # Smoothing MSE
        self.mse_history_smooth = deque(maxlen=config.mse_smooth_window)

        # MSE untuk graph
        self.mse_graph_data = deque(maxlen=config.mse_graph_history)

        # Status deteksi
        self.current_mse:    float = 0.0
        self.current_status: str   = "MENDETEKSI TANGAN..."
        self.current_color         = (0, 255, 255)
        self.anomaly_streak: int   = 0
        self.is_anomaly:     bool  = False

        # Statistik sesi
        self.session_start     = time.time()
        self.total_frames      = 0
        self.detected_frames   = 0
        self.normal_frames     = 0
        self.anomaly_events    = 0
        self._prev_anomaly     = False

        # FPS
        self._fps_times = deque(maxlen=30)

        # Session log
        self._session_log: list = []

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

    # --------------------------------------------------------
    # LIFECYCLE
    # --------------------------------------------------------
    def _signal_handler(self, sig, frame):
        log.info("Ctrl+C diterima. Menutup sistem...")
        self._running = False

    def load_all(self) -> bool:
        """Muat semua komponen AI. Return False jika ada yang gagal."""
        if not self._load_model():
            return False
        if not self._load_scaler():
            return False
        if not self._load_threshold():
            return False
        self._validate_model_shape()
        return True

    def _load_model(self) -> bool:
        from tensorflow.keras.models import load_model
        log.info(f"Memuat Model AI dari: {self.cfg.model_path}")
        try:
            self.model = load_model(self.cfg.model_path, compile=False)
            log.info("Model berhasil dimuat!")
            return True
        except Exception as e:
            log.error(f"Gagal memuat model: {e}")
            log.error("Pastikan file .keras ada di folder 'models/'")
            return False

    def _load_scaler(self) -> bool:
        log.info(f"Memuat Scaler dari: {self.cfg.scaler_path}")
        try:
            self.scaler = joblib.load(self.cfg.scaler_path)
            log.info("Scaler berhasil dimuat!")
            return True
        except Exception as e:
            log.error(f"Gagal memuat scaler: {e}")
            return False

    def _load_threshold(self) -> bool:
        log.info(f"Memuat Threshold dari: {self.cfg.threshold_path}")
        try:
            if not Path(self.cfg.threshold_path).exists():
                raise FileNotFoundError("File threshold tidak ditemukan.")
            with open(self.cfg.threshold_path, 'r') as f:
                for line in f:
                    if line.startswith("THRESHOLD"):
                        self.threshold = float(line.split("=")[1].strip())
                        log.info(f"Threshold dimuat: {self.threshold:.8f}")
                        return True
            log.warning("Format threshold tidak sesuai, memakai default 0.015")
            self.threshold = 0.015
            return True
        except Exception as e:
            log.error(f"Gagal membaca threshold: {e}")
            return False

    def _validate_model_shape(self):
        """Validasi input shape model agar sesuai konfigurasi."""
        try:
            expected = (None, self.cfg.window_size, self.cfg.n_features)
            actual   = tuple(self.model.input_shape)
            if actual == expected:
                log.info(f"Validasi shape model OK: {actual}")
            else:
                log.warning(f"Shape model {actual} != konfigurasi {expected}. Periksa WINDOW_SIZE/N_FEATURES!")
        except Exception:
            pass

    # --------------------------------------------------------
    # CORE INFERENCE
    # --------------------------------------------------------
    def _extract_landmarks(self, hand_landmarks) -> np.ndarray:
        """Ekstrak dan normalisasi landmark tangan (ego-centric)."""
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32
        )  # (21, 3)
        coords -= coords[0]                      # Normalisasi ego-centric
        return coords.flatten().reshape(1, -1)   # (1, 63)

    def _predict_mse(self) -> float:
        """Jalankan inferensi dan hitung MSE (dismoothing)."""
        input_data = np.array(self.seq_buffer).reshape(
            1, self.cfg.window_size, self.cfg.n_features
        )
        recon = self.model.predict(input_data, verbose=0)
        raw_mse = float(np.mean(np.power(input_data - recon, 2)))

        self.mse_history_smooth.append(raw_mse)
        smoothed = float(np.mean(self.mse_history_smooth))
        return smoothed

    def _update_anomaly_status(self, mse: float):
        """Update status anomali dengan streak detection."""
        if mse >= self.threshold:
            self.anomaly_streak += 1
        else:
            self.anomaly_streak = 0

        prev = self.is_anomaly
        self.is_anomaly = (self.anomaly_streak >= self.cfg.anomaly_streak_needed)

        # Hitung event anomali baru
        if self.is_anomaly and not prev:
            self.anomaly_events += 1

        if self.is_anomaly:
            self.current_status = "!! ANOMALI TERDETEKSI !!"
            self.current_color  = (0, 0, 255)
            self._prev_anomaly  = True
        else:
            self.current_status = "NORMAL"
            self.current_color  = (0, 255, 0)
            self.normal_frames += 1
            self._prev_anomaly  = False

    # --------------------------------------------------------
    # SESSION LOGGING
    # --------------------------------------------------------
    def _log_frame(self, mse: float, status: str):
        self._session_log.append({
            "timestamp": datetime.now().isoformat(timespec='milliseconds'),
            "mse":       round(mse, 8),
            "status":    status,
            "threshold": round(self.threshold, 8),
        })

    def _save_session_log(self):
        if not self._session_log:
            return
        Path(self.cfg.session_log_dir).mkdir(parents=True, exist_ok=True)
        fname = datetime.now().strftime("sesi_%Y%m%d_%H%M%S.csv")
        fpath = Path(self.cfg.session_log_dir) / fname
        keys  = ["timestamp", "mse", "status", "threshold"]
        with open(fpath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._session_log)
        log.info(f"Log sesi disimpan: {fpath}")

    def _reset_session(self):
        self.session_start   = time.time()
        self.total_frames    = 0
        self.detected_frames = 0
        self.normal_frames   = 0
        self.anomaly_events  = 0
        self.mse_graph_data.clear()
        self.mse_history_smooth.clear()
        self.seq_buffer.clear()
        self.anomaly_streak  = 0
        self.is_anomaly      = False
        self._session_log    = []
        log.info("Sesi di-reset.")

    def _save_snapshot(self, frame: np.ndarray):
        Path(self.cfg.snapshot_dir).mkdir(parents=True, exist_ok=True)
        fname = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S.jpg")
        fpath = str(Path(self.cfg.snapshot_dir) / fname)
        cv2.imwrite(fpath, frame)
        log.info(f"Snapshot disimpan: {fpath}")

    # --------------------------------------------------------
    # DRAWING / UI
    # --------------------------------------------------------
    def _draw_mse_graph(self, frame: np.ndarray):
        """Render mini MSE graph di sudut kiri atas."""
        if len(self.mse_graph_data) < 2:
            return

        gx, gy, gw, gh = 10, 50, 220, 80

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (gx, gy), (gx + gw, gy + gh), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (80, 80, 80), 1)

        # Judul
        cv2.putText(frame, "MSE Graph", (gx + 4, gy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        data   = list(self.mse_graph_data)
        max_v  = max(max(data), self.threshold * 1.5) + 1e-9
        n      = len(data)
        pts    = []
        for i, v in enumerate(data):
            px = int(gx + 2 + i * (gw - 4) / max(n - 1, 1))
            py = int(gy + gh - 2 - (v / max_v) * (gh - 4))
            pts.append((px, py))

        # Garis threshold
        ty = int(gy + gh - 2 - (self.threshold / max_v) * (gh - 4))
        cv2.line(frame, (gx + 2, ty), (gx + gw - 2, ty), (0, 80, 255), 1)

        # Garis MSE
        for i in range(1, len(pts)):
            col = (0, 0, 220) if data[i] >= self.threshold else (0, 220, 80)
            cv2.line(frame, pts[i - 1], pts[i], col, 1, cv2.LINE_AA)

    def _draw_stats_panel(self, frame: np.ndarray, fps: float):
        """Panel statistik sesi di sisi kanan."""
        h, w = frame.shape[:2]
        pw, ph = 200, 160
        px = w - pw - 10
        py = 50

        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (80, 80, 80), 1)

        elapsed = time.time() - self.session_start
        m, s    = divmod(int(elapsed), 60)

        score = 0.0
        if self.detected_frames > 0:
            score = (self.normal_frames / self.detected_frames) * 100

        # Warna skor
        sc_color = (0, 220, 80) if score >= 80 else (0, 180, 255) if score >= 50 else (0, 0, 220)

        lines = [
            ("== STATISTIK SESI ==", (200, 200, 200)),
            (f"Durasi   : {m:02d}:{s:02d}",       (180, 180, 180)),
            (f"FPS      : {fps:.1f}",              (180, 180, 180)),
            (f"Skor     : {score:.1f}%",           sc_color),
            (f"Anomali  : {self.anomaly_events}x", (0, 100, 255) if self.anomaly_events else (180, 180, 180)),
            (f"MSE      : {self.current_mse:.5f}", (180, 180, 180)),
            (f"Thr      : {self.threshold:.5f}",   (180, 180, 180)),
        ]

        for i, (txt, col) in enumerate(lines):
            cv2.putText(frame, txt, (px + 8, py + 18 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    def _draw_header(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 42), (20, 20, 20), -1)
        cv2.putText(frame,
                    "Rehabilitasi  |  Sistem Deteksi Anomali Gerakan Tangan  |  v3.0 Advanced",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    def _draw_status_bar(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        # Progress buffer
        progress = int(w * len(self.seq_buffer) / self.cfg.window_size)
        cv2.rectangle(frame, (0, h - 107), (w, h - 102), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, h - 107), (progress, h - 102), (0, 200, 255), -1)

        # Overlay bawah
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 100), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Status utama
        cv2.putText(frame, self.current_status,
                    (20, h - 60), cv2.FONT_HERSHEY_DUPLEX, 1.2,
                    self.current_color, 2, cv2.LINE_AA)

        # Info bawah
        buf_pct  = int(len(self.seq_buffer) / self.cfg.window_size * 100)
        info_str = (f"MSE: {self.current_mse:.6f}  |  "
                    f"Thr: {self.threshold:.6f}  |  "
                    f"Buffer: {buf_pct}%  |  "
                    f"[Q] Keluar  [S] Snapshot  [R] Reset")
        cv2.putText(frame, info_str,
                    (20, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (180, 180, 180), 1, cv2.LINE_AA)

    def _draw_hand_skeleton(self, frame, hand_landmarks):
        """Warna skeleton berbeda saat anomali vs normal."""
        lm_style = self.mp_styles.get_default_hand_landmarks_style()
        cn_style = self.mp_styles.get_default_hand_connections_style()

        if self.is_anomaly:
            # Override warna koneksi menjadi merah saat anomali
            from mediapipe.python.solutions.drawing_styles import DrawingSpec
            cn_style = DrawingSpec(color=(0, 0, 255), thickness=2)

        self.mp_drawing.draw_landmarks(
            frame, hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            lm_style, cn_style
        )

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------
    def run(self):
        """Entry point utama. Jalankan loop webcam."""
        log.info("=" * 60)
        log.info("  Rehabilitasi - SISTEM DETEKSI ANOMALI REAL-TIME v3.0")
        log.info("=" * 60)
        log.info(f"  Threshold : {self.threshold:.8f}")
        log.info(f"  Window    : {self.cfg.window_size} frame")
        log.info(f"  Streak    : {self.cfg.anomaly_streak_needed} frame berturut")
        log.info(f"  Smooth    : {self.cfg.mse_smooth_window} sample")
        log.info("=" * 60)
        log.info("Kontrol: [Q] Keluar  [S] Snapshot  [R] Reset Sesi")
        log.info("-" * 60)

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.cfg.min_detect_conf,
            min_tracking_confidence=self.cfg.min_track_conf,
        )

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log.error("Webcam tidak dapat diakses!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self._running      = True
        self.session_start = time.time()

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    log.warning("Gagal membaca frame, mencoba lagi...")
                    time.sleep(0.05)
                    continue

                self.total_frames += 1
                now = time.time()
                self._fps_times.append(now)
                fps = len(self._fps_times) / max(now - self._fps_times[0], 1e-9) if len(self._fps_times) > 1 else 0.0

                frame = cv2.flip(frame, 1)

                # ---- Proses MediaPipe ----
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = self.hands.process(rgb)
                rgb.flags.writeable = True

                if results.multi_hand_landmarks:
                    self.detected_frames += 1
                    hand_lm = results.multi_hand_landmarks[0]

                    # Gambar skeleton
                    self._draw_hand_skeleton(frame, hand_lm)

                    # Ekstrak & scale landmark
                    flat   = self._extract_landmarks(hand_lm)
                    scaled = self.scaler.transform(flat).flatten()
                    self.seq_buffer.append(scaled)

                    # Inferensi jika buffer penuh
                    if len(self.seq_buffer) == self.cfg.window_size:
                        self.current_mse = self._predict_mse()
                        self.mse_graph_data.append(self.current_mse)
                        self._update_anomaly_status(self.current_mse)
                        self._log_frame(self.current_mse, self.current_status)
                    else:
                        self.current_status = f"Mengisi buffer... {len(self.seq_buffer)}/{self.cfg.window_size}"
                        self.current_color  = (0, 200, 255)
                else:
                    self.current_status = "Arahkan tangan ke kamera..."
                    self.current_color  = (0, 255, 255)
                    self.anomaly_streak = 0

                # ---- Render UI ----
                self._draw_header(frame)
                self._draw_mse_graph(frame)
                self._draw_stats_panel(frame, fps)
                self._draw_status_bar(frame)

                cv2.imshow("Rehabilitasi - Deteksi Anomali v3.0", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    log.info("Keluar dari aplikasi...")
                    break
                elif key == ord('s'):
                    self._save_snapshot(frame)
                elif key == ord('r'):
                    self._reset_session()

        finally:
            self._running = False
            cap.release()
            if self.hands:
                self.hands.close()
            cv2.destroyAllWindows()
            self._print_session_summary()
            self._save_session_log()

    # --------------------------------------------------------
    # RINGKASAN SESI
    # --------------------------------------------------------
    def _print_session_summary(self):
        elapsed = time.time() - self.session_start
        m, s    = divmod(int(elapsed), 60)
        score   = (self.normal_frames / max(self.detected_frames, 1)) * 100

        log.info("\n" + "=" * 60)
        log.info("  RINGKASAN SESI REHABILITASI")
        log.info("=" * 60)
        log.info(f"  Durasi sesi         : {m:02d} menit {s:02d} detik")
        log.info(f"  Total frame diproses: {self.total_frames}")
        log.info(f"  Frame tangan terdeteksi: {self.detected_frames}")
        log.info(f"  Skor Rehabilitasi   : {score:.2f}% NORMAL")
        log.info(f"  Total kejadian anomali: {self.anomaly_events}x")
        log.info(f"  MSE terakhir        : {self.current_mse:.8f}")
        log.info(f"  Threshold           : {self.threshold:.8f}")
        log.info("=" * 60)
        log.info("  Log sesi disimpan ke data/session_logs/")
        log.info("=" * 60)


# ============================================================
# PROGRAM UTAMA
# ============================================================
if __name__ == "__main__":
    config = SystemConfig()
    system = RehabilitasiSystem(config)

    if not system.load_all():
        log.error("Sistem tidak dapat dijalankan. Periksa folder 'models/'.")
        sys.exit(1)

    system.run()
