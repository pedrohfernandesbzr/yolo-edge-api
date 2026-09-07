import yaml
from pathlib import Path

def test_dataset_integrity():
    dataset_path = Path("dataset/exports/epi-v1")
    yaml_file = dataset_path / "data.yaml"
    
    # 1. Verifica se o arquivo de configuração existe
    assert yaml_file.exists(), "Arquivo data.yaml não encontrado!"
    
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
        
    # 2. Valida as classes e os dados do YAML
    assert data is not None, "data.yaml vazio"
    assert data.get('nc', 0) == 3, "Número de classes incorreto (deve ser 3: capacete, colete, pessoa)"
    assert 'capacete' in data.get('names', []), "Classe 'capacete' ausente"
    
    # 3. Varre as pastas garantindo que as imagens e as marcações batem
    for split in ['train', 'valid', 'test']:
        img_dir = dataset_path / split / 'images'
        lbl_dir = dataset_path / split / 'labels'
        
        assert img_dir.exists(), f"Diretório de imagens ausente no split: {split}"
        assert lbl_dir.exists(), f"Diretório de labels ausente no split: {split}"
        
        imgs = len(list(img_dir.glob('*.jpg')))
        lbls = len(list(lbl_dir.glob('*.txt')))
        
        assert imgs > 0, f"Split vazio. Nenhuma imagem encontrada em: {split}"
        assert imgs == lbls, f"Inconsistência em {split}: {imgs} imagens vs {lbls} labels"
