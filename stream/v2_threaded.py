"""
stream/v2_threaded.py - Captura e inferência em threads separadas.
Adaptado com auto-detecção: prioriza CSI, fallback para IP Webcam.
"""
import argparse
import queue
import threading
import time
from pathlib import Path
import sys
import subprocess
import cv2
import numpy as np
from ultralytics import YOLO
import torch

_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

sys.path.insert(0, str(Path(__file__).parent.parent))

def discover_camera(fallback_url):
    try:
        result = subprocess.run(["rpicam-hello", "--list-cameras"], capture_output=True, text=True, timeout=2)
        if "Available cameras" in result.stdout and "0 cameras" not in result.stdout:
            return "0", False
    except Exception:
        pass
    return fallback_url, True

class CameraCapture:
    def __init__(self, source, is_ip_cam, width, height, fps=30):
        self.is_ip_cam = is_ip_cam
        self._buf = queue.Queue(maxsize=1)
        self._running = threading.Event()
        self._running.set()
        
        if is_ip_cam:
            self._cap = cv2.VideoCapture(source)
            self._thread = threading.Thread(target=self._capture_loop_ip, daemon=True)
        else:
            cmd = [
                "rpicam-vid", "-t", "0", "-n", "--codec", "mjpeg",
                "--camera", source,
                "--width", str(width), "--height", str(height),
                "--framerate", str(fps),
                "-o", "-",
            ]
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self._thread = threading.Thread(target=self._capture_loop_rpi, daemon=True)
            
        self.frames_captured = 0
        self.frames_dropped = 0

    def start(self):
        self._thread.start()
        return self

    def _capture_loop_ip(self):
        while self._running.is_set():
            ret, frame = self._cap.read()
            if not ret: continue
            self._enqueue_frame(frame)

    def _capture_loop_rpi(self):
        raw = b""
        while self._running.is_set():
            chunk = self._proc.stdout.read(4096)
            if not chunk: break
            raw += chunk
            end = raw.rfind(b"\xff\xd9")
            if end == -1: continue
            start = raw.rfind(b"\xff\xd8", 0, end)
            if start == -1: continue
            jpg = raw[start:end+2]
            raw = raw[end+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue
            self._enqueue_frame(frame)

    def _enqueue_frame(self, frame):
        if self._buf.full():
            try:
                self._buf.get_nowait()
                self.frames_dropped += 1
            except queue.Empty:
                pass
        self._buf.put(frame)
        self.frames_captured += 1

    def read(self, timeout=1.0):
        try:
            return self._buf.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._running.clear()
        if self.is_ip_cam:
            self._cap.release()
        else:
            self._proc.terminate()
            try: self._proc.wait(2.0)
            except subprocess.TimeoutExpired: self._proc.kill()

class YOLOInference:
    def __init__(self, model_path: str, conf: float = 0.4):
        self.model = YOLO(model_path)
        self.conf  = conf
        self.count = 0
        self.total_ms = 0.0

    def run(self, frame):
        t0 = time.perf_counter()
        results = self.model(frame, conf=self.conf, verbose=False)
        elapsed = (time.perf_counter() - t0) * 1000
        self.count    += 1
        self.total_ms += elapsed
        annotated = results[0].plot()
        n_det = len(results[0].boxes)
        return annotated, n_det, elapsed

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--width",   type=int,   default=640)
    p.add_argument("--height",  type=int,   default=480)
    p.add_argument("--fps",     type=int,   default=30)
    p.add_argument("--model",   type=str,   default="models/yolov8n.pt")
    p.add_argument("--conf",    type=float, default=0.4)
    p.add_argument("--frames",  type=int,   default=50)
    p.add_argument("--fallback-url", type=str, required=True)
    return p.parse_args()

def main():
    args = parse_args()
    source, is_ip_cam = discover_camera(args.fallback_url)
    
    camera = CameraCapture(source, is_ip_cam, args.width, args.height, args.fps)
    yolo   = YOLOInference(args.model, args.conf)
    
    camera.start()
    time.sleep(0.5)
    
    print(f"[INFO] Processando {args.frames} frames com threading...")
    print(f"{'Frame':>6} | {'Inferência':>10} | {'FPS inst.':>9} | {'Detecções':>9}")
    print("-" * 48)
    
    t_start = time.perf_counter()
    frame_count = 0
    
    while frame_count < args.frames:
        frame = camera.read(timeout=2.0)
        if frame is None:
            print("[AVISO] Timeout na leitura do frame.")
            continue
            
        annotated, n_det, infer_ms = yolo.run(frame)
        frame_count += 1
        elapsed_total = (time.perf_counter() - t_start)
        fps_avg = frame_count / elapsed_total if elapsed_total > 0 else 0
        
        if frame_count % 10 == 0:
            print(f"{frame_count:>6} | {infer_ms:>9.1f}ms | {fps_avg:>8.1f} | {n_det:>9}")
            
    camera.stop()
    total_time = time.perf_counter() - t_start
    
    print("\n" + "=" * 58)
    print("RELATORIO - Threading com buffer de 1 frame")
    print("=" * 58)
    print(f"  Frames processados   : {frame_count}")
    print(f"  Tempo total          : {total_time:.1f} s")
    print(f"  FPS medio sustentado : {frame_count/total_time:.1f} FPS")
    print(f"  Inferencia media     : {yolo.avg_ms:.1f} ms")
    print(f"  Frames capturados    : {camera.frames_captured}")
    print(f"  Frames descartados   : {camera.frames_dropped} ({100*camera.frames_dropped/max(camera.frames_captured,1):.0f}%)")
    print("=" * 58)

if __name__ == "__main__":
    main()
