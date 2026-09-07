"""
stream/capture_frames.py
Captura frames do stream de câmera e salva em dataset/raw/.
Adaptado com fallback para IP Webcam.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path("dataset/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

class RpicamCapture:
    def __init__(self, device: int, width: int, height: int, fps: int = 15):
        cmd = [
            "rpicam-vid", "-t", "0", "-n", "--codec", "mjpeg",
            "--camera", str(device),
            "--width", str(width), "--height", str(height),
            "--framerate", str(fps), "-o", "-",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        self._buf = b""

    def read(self):
        while True:
            start = self._buf.find(b"\xff\xd8")
            end = self._buf.find(b"\xff\xd9", start + 2) if start != -1 else -1
            if start != -1 and end != -1:
                jpg = self._buf[start:end + 2]
                self._buf = self._buf[end + 2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                return frame is not None, frame
            chunk = self._proc.stdout.read(4096)
            if not chunk: return False, None
            self._buf += chunk

    def isOpened(self):
        return self._proc.poll() is None

    def release(self):
        self._proc.terminate()
        try: self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired: self._proc.kill()

def is_sharp_enough(frame: np.ndarray, threshold: float = 80.0) -> bool:
    """Descarta frames borrados usando a variância do Laplaciano."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score >= threshold

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total",    type=int, default=150, help="Total de frames a capturar")
    p.add_argument("--interval", type=float, default=1.0, help="Intervalo entre capturas em segundos")
    p.add_argument("--width",    type=int, default=640)
    p.add_argument("--height",   type=int, default=480)
    p.add_argument("--fallback-url", type=str, required=True)
    return p.parse_args()

def main():
    args = parse_args()
    source, is_ip_cam = discover_camera(args.fallback_url)
    
    if is_ip_cam:
        cap = cv2.VideoCapture(source)
    else:
        cap = RpicamCapture(int(source), args.width, args.height)

    saved = 0
    skipped = 0
    last_saved = 0.0

    print(f"[INFO] Iniciando captura: {args.total} frames | intervalo: {args.interval}s")
    print(f"[INFO] Salvando em: {OUTPUT_DIR.resolve()}")
    print("[INFO] Pressione Ctrl+C para encerrar antecipadamente.")

    try:
        while saved < args.total:
            if is_ip_cam:
                ret, frame = cap.read()
            else:
                ret, frame = cap.read()

            if not ret:
                time.sleep(0.1)
                continue

            now = time.time()
            if now - last_saved < args.interval:
                continue

            if not is_sharp_enough(frame):
                skipped += 1
                continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = OUTPUT_DIR / f"frame_{ts}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            
            saved += 1
            last_saved = time.time()
            print(f"\r  [{saved:>3}/{args.total}] Salvo: {path.name} (descartados: {skipped})", end="")

    except KeyboardInterrupt:
        print("\n[INFO] Captura interrompida pelo usuário")
    finally:
        cap.release()
        print(f"\n[OK] {saved} frames salvos em {OUTPUT_DIR}")
        print(f"[OK] {skipped} frames borrados descartados automaticamente")

if __name__ == "__main__":
    main()
