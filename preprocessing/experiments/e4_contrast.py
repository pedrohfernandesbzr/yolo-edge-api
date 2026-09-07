"""
Experimento E4: equalização de histograma vs CLAHE.
Usa o dataset epi-v1-dark (imagens escurecidas por gamma) para
simular condições de iluminação adversa.
"""
import sys, cv2, numpy as np
sys.path.insert(0, '.')
from preprocessing.utils.evaluate import evaluate_pipeline
import preprocessing.utils.evaluate as ev_module

DATASET_DARK = "dataset/exports/epi-v1-dark/data.yaml"

def equalize_hist_hsv(frame: np.ndarray) -> np.ndarray:
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq   = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    rgb    = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)
    return rgb

def equalize_hist_lab(frame: np.ndarray) -> np.ndarray:
    lab       = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b   = cv2.split(lab)
    l_eq      = cv2.equalizeHist(l)
    lab_eq    = cv2.merge([l_eq, a, b])
    rgb       = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return rgb

def clahe_hsv(frame: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    clahe  = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_cl   = clahe.apply(v)
    hsv_cl = cv2.merge([h, s, v_cl])
    return cv2.cvtColor(hsv_cl, cv2.COLOR_HSV2RGB)

def clahe_lab(frame: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    clahe  = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab    = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_cl   = clahe.apply(l)
    lab_cl = cv2.merge([l_cl, a, b])
    return cv2.cvtColor(lab_cl, cv2.COLOR_LAB2RGB)

def rgb_only(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    print("=" * 65)
    print(" E4 — Equalização de Contraste em Imagens Subexpostas")
    print("=" * 65)

    original_ds = ev_module.DATASET_YAML
    ev_module.DATASET_YAML = DATASET_DARK

    results = []
    results.append(evaluate_pipeline(rgb_only,         "E4-A: RGB apenas (ilum. ruim)"))
    results.append(evaluate_pipeline(equalize_hist_hsv,"E4-B: equalizeHist (canal V-HSV)"))
    results.append(evaluate_pipeline(equalize_hist_lab,"E4-C: equalizeHist (canal L-LAB)"))
    results.append(evaluate_pipeline(clahe_hsv,        "E4-D: CLAHE clip=2 tile=8 (HSV)"))
    results.append(evaluate_pipeline(clahe_lab,        "E4-E: CLAHE clip=2 tile=8 (LAB)"))
    results.append(evaluate_pipeline(
        lambda f: clahe_lab(f, clip=4.0, tile=8),
        "E4-F: CLAHE clip=4 tile=8 (LAB)"
    ))

    ev_module.DATASET_YAML = original_ds

    print("\n--- Resumo E4 (dataset escurecido) ---")
    b = results[0]['map50']
    for r in results[1:]:
        print(f"  {r['label']:38s}  mAP={r['map50']:.4f}  delta={r['map50']-b:+.4f}")
