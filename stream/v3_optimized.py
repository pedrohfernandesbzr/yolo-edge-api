"""
stream/v3_optimized.py — Pipeline otimizado para tempo real no Raspberry Pi 5.
Combina threading, frame skip, resolução adaptativa, OSD e fallback para IP Webcam.
"""
import argparse
import queue
import threading
import time
from pathlib import Path
from collections import deque
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
    print("[INFO] Buscando câmeras conectadas...")
    try:
        result = subprocess.run(["rpicam-hello", "--list-cameras"], capture_output=True, text=True, timeout=2)
        if "Available cameras" in result.stdout and "0 cameras" not in result.stdout:
            print("[INFO] -> Câmera física detectada!")
            return "0", False
    except Exception:
        pass
    print(f"[INFO] -> Usando IP Webcam: {fallback_url}")
    return fallback_url, True

class OptimizedCamera:
    def __init__(self, source, is_ip_cam, width, height, fps=30):
        self.is_ip_cam = is_ip_cam
        self._buf = queue.Queue(maxsize=1)
        self._running = threading.Event()
        self._running.set()
        
        if is_ip_cam:
            self._cap = cv2.VideoCapture(source)
            self._thread = threading.Thread(target=self._loop_ip, daemon=True)
        else:
            cmd = [
                "rpicam-vid", "-t", "0", "-n", "--codec", "mjpeg",
                "--camera", source,
                "--width", str(width), "--height", str(height),
                "--framerate", str(fps), "-o", "-",
            ]
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self._thread = threading.Thread(target=self._loop_rpi, daemon=True)
            self._raw = b""

        self.frames_in = 0
        self.frames_out = 0

    def start(self):
        self._thread.start()
        return self

    def _enqueue(self, frame):
        self.frames_in += 1
        if self._buf.full():
            try: self._buf.get_nowait()
            except queue.Empty: pass
        self._buf.put(frame)

    def _loop_ip(self):
        while self._running.is_set():
            ret, frame = self._cap.read()
            if not ret: continue
            self._enqueue(frame)

    def _loop_rpi(self):
        while self._running.is_set():
            chunk = self._proc.stdout.read(4096)
            if not chunk: break
            self._raw += chunk
            end = self._raw.rfind(b"\xff\xd9")
            if end == -1: continue
            start = self._raw.rfind(b"\xff\xd8", 0, end)
            if start == -1: continue
            jpg = self._raw[start:end + 2]
            self._raw = self._raw[end + 2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue
            self._enqueue(frame)

    def read(self, timeout=1.0):
        try:
            frame = self._buf.get(timeout=timeout)
            self.frames_out += 1
            return frame
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
        self._thread.join(timeout=2.0)

class RealtimeDetector:
    def __init__(self, model_path: str, conf: float, infer_every: int, infer_size: int):
        self.model = YOLO(model_path)
        self.conf = conf
        self.infer_every = infer_every
        self.infer_size = infer_size
        self._frame_idx = 0
        self._last_boxes = []
        self._last_infer_ms = 0.0
        self._fps_window = deque(maxlen=30)
        self._t_last = time.perf_counter()

    def process(self, frame: np.ndarray) -> np.ndarray:
        self._frame_idx += 1
        now = time.perf_counter()
        self._fps_window.append(now - self._t_last)
        self._t_last = now

        if self._frame_idx % self.infer_every == 0:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (self.infer_size, self.infer_size))
            t0 = time.perf_counter()
            results = self.model(small, conf=self.conf, verbose=False)
            self._last_infer_ms = (time.perf_counter() - t0) * 1000
            
            sx, sy = w / self.infer_size, h / self.infer_size
            self._last_boxes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    label = self.model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    self._last_boxes.append((label, conf, int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)))

        output = frame.copy()
        for (label, conf, x1, y1, x2, y2) in self._last_boxes:
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            caption = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(output, (x1, y1-th-8), (x1+tw+4, y1), (0,255,0), -1)
            cv2.putText(output, caption, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)

        fps_display = (len(self._fps_window) / sum(self._fps_window)) if self._fps_window else 0
        is_infer_frame = (self._frame_idx % self.infer_every == 0)
        
        for i, line in enumerate([f"FPS: {fps_display:.1f}", f"Infer: {self._last_infer_ms:.0f}ms", f"Det: {len(self._last_boxes)}", f"Frame: {self._frame_idx}"]):
            cv2.putText(output, line, (10, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if is_infer_frame else (200, 200, 200), 2)
        return output

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--model", type=str, default="models/yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--infer-every", type=int, default=3)
    p.add_argument("--infer-size", type=int, default=320)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--fallback-url", type=str, required=True)
    return p.parse_args()

def main():
    args = parse_args()
    source, is_ip_cam = discover_camera(args.fallback_url)
    
    camera = OptimizedCamera(source, is_ip_cam, args.width, args.height, args.fps).start()
    detector = RealtimeDetector(args.model, args.conf, args.infer_every, args.infer_size)
    
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))
        print(f"[INFO] Gravando saida em: {args.output}")

    time.sleep(0.5)
    print("[INFO] Processando em background. Aguarde uns 10 segundos e pressione Ctrl+C...")

    try:
        while True:
            frame = camera.read(timeout=2.0)
            if frame is None: continue
            annotated = detector.process(frame)
            if writer: writer.write(annotated)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        if writer: writer.release()
        print(f"[INFO] Finalizado. Frames processados: {detector._frame_idx}")

if __name__ == "__main__":
    main()
