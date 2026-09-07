#!/usr/bin/env python3
import yaml
from pathlib import Path

def check_split(base_path, split_name):
    img_dir = base_path / split_name / 'images'
    lbl_dir = base_path / split_name / 'labels'
    imgs = len(list(img_dir.glob('*.jpg'))) if img_dir.exists() else 0
    lbls = len(list(lbl_dir.glob('*.txt'))) if lbl_dir.exists() else 0
    return imgs, lbls

def main():
    dataset_path = Path("dataset/exports/epi-v1")
    yaml_file = dataset_path / "data.yaml"
    
    print("\n" + "="*50)
    print(" INSPEÇÃO DE INTEGRIDADE DO DATASET YOLOv8")
    print("="*50)
    
    if not yaml_file.exists():
        print("[ERRO] Arquivo data.yaml não encontrado!")
        return

    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        print(f" Caminho base: {data.get('path', 'NÃO DEFINIDO')}")
        print(f" Classes ({data.get('nc', 0)}): {', '.join(data.get('names', []))}")
        print("-" * 50)
        
        total_imgs = 0
        for split in ['train', 'valid', 'test']:
            imgs, lbls = check_split(dataset_path, split)
            total_imgs += imgs
            status = "OK" if imgs == lbls and imgs > 0 else "ERRO"
            print(f" {split.upper():<6} | Imagens: {imgs:>4} | Labels: {lbls:>4} | {status}")
            
        print("-" * 50)
        print(f" TOTAL DE IMAGENS: {total_imgs}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"[ERRO] Falha ao ler data.yaml: {e}")

if __name__ == "__main__":
    main()
