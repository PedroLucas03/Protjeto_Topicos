"""Script para preparar e dividir o dataset em train/validation/test"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm

DATASET_ORIGINAL = r"c:\Users\Pedro\Downloads\archive (1)\images\Images"
DATASET_PREPARADO = r"c:\Users\Pedro\Documents\Identificacao_Racas\dataset"
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

def criar_estrutura_diretorios():
    for split in ['train', 'validation', 'test']:
        (Path(DATASET_PREPARADO) / split).mkdir(parents=True, exist_ok=True)

def preparar_dataset():
    criar_estrutura_diretorios()
    
    racas = sorted([d for d in os.listdir(DATASET_ORIGINAL) 
                    if os.path.isdir(os.path.join(DATASET_ORIGINAL, d))])
    
    estatisticas = {
        'train': 0,
        'validation': 0,
        'test': 0,
        'total': 0,
        'racas': {}
    }
    
    for raca in tqdm(racas, desc="Processando raças"):
        raca_path = os.path.join(DATASET_ORIGINAL, raca)
        imagens = [f for f in os.listdir(raca_path) if f.endswith('.jpg')]
        
        if len(imagens) < 3:
            continue
        
        train_imgs, temp_imgs = train_test_split(
            imagens, test_size=(VAL_SIZE + TEST_SIZE), random_state=42
        )
        val_imgs, test_imgs = train_test_split(
            temp_imgs, test_size=0.5, random_state=42
        )
        
        for split in ['train', 'validation', 'test']:
            (Path(DATASET_PREPARADO) / split / raca).mkdir(parents=True, exist_ok=True)
        
        splits_imgs = {'train': train_imgs, 'validation': val_imgs, 'test': test_imgs}
        
        for split, imgs in splits_imgs.items():
            for img in imgs:
                shutil.copy2(
                    os.path.join(raca_path, img),
                    Path(DATASET_PREPARADO) / split / raca / img
                )
        
        estatisticas['train'] += len(train_imgs)
        estatisticas['validation'] += len(val_imgs)
        estatisticas['test'] += len(test_imgs)
        estatisticas['total'] += len(imagens)
        
        estatisticas['racas'][raca] = {
            'total': len(imagens),
            'train': len(train_imgs),
            'validation': len(val_imgs),
            'test': len(test_imgs)
        }
    
    with open(Path(DATASET_PREPARADO) / 'dataset_stats.json', 'w', encoding='utf-8') as f:
        json.dump(estatisticas, f, indent=2, ensure_ascii=False)
    
    print(f"\nTotal: {estatisticas['total']:,} imagens")
    print(f"Train: {estatisticas['train']:,} | Val: {estatisticas['validation']:,} | Test: {estatisticas['test']:,}")
    
    return estatisticas

if __name__ == "__main__":
    preparar_dataset()
