"""
Script para preparar e dividir o dataset em train/validation/test
"""
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
    print("Criando estrutura de diretórios...")
    
    for split in ['train', 'validation', 'test']:
        split_path = Path(DATASET_PREPARADO) / split
        split_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Estrutura criada em: {DATASET_PREPARADO}")

def preparar_dataset():
    """Divide o dataset em train/validation/test mantendo proporção de classes"""
    
    criar_estrutura_diretorios()
    
    racas = sorted([d for d in os.listdir(DATASET_ORIGINAL) 
                    if os.path.isdir(os.path.join(DATASET_ORIGINAL, d))])
    
    print(f"\nProcessando {len(racas)} raças...")
    
    estatisticas = {
        'train': 0,
        'validation': 0,
        'test': 0,
        'total': 0,
        'racas': {}
    }
    
    for raca in tqdm(racas, desc="Dividindo dataset"):
        raca_path = os.path.join(DATASET_ORIGINAL, raca)
        
        imagens = [f for f in os.listdir(raca_path) if f.endswith('.jpg')]
        
        if len(imagens) < 3:
            print(f"Raça {raca} tem poucas imagens ({len(imagens)}), pulando...")
            continue
        
        train_imgs, temp_imgs = train_test_split(
            imagens, 
            test_size=(VAL_SIZE + TEST_SIZE),
            random_state=42
        )
        
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            test_size=0.5,
            random_state=42
        )
        
        for split in ['train', 'validation', 'test']:
            split_raca_path = Path(DATASET_PREPARADO) / split / raca
            split_raca_path.mkdir(parents=True, exist_ok=True)
        
        splits_imgs = {
            'train': train_imgs,
            'validation': val_imgs,
            'test': test_imgs
        }
        
        for split, imgs in splits_imgs.items():
            for img in imgs:
                src = os.path.join(raca_path, img)
                dst = Path(DATASET_PREPARADO) / split / raca / img
                shutil.copy2(src, dst)
        
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
    
    # Salvar estatísticas
    stats_path = Path(DATASET_PREPARADO) / 'dataset_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(estatisticas, f, indent=2, ensure_ascii=False)
    
    # Exibir resumo
    print("\n" + "="*70)
    print("RESUMO DA DIVISÃO DO DATASET")
    print("="*70)
    print(f"\nTotal de imagens: {estatisticas['total']:,}")
    print(f"\nTreinamento:  {estatisticas['train']:,} ({estatisticas['train']/estatisticas['total']*100:.1f}%)")
    print(f"Validação:    {estatisticas['validation']:,} ({estatisticas['validation']/estatisticas['total']*100:.1f}%)")
    print(f"Teste:        {estatisticas['test']:,} ({estatisticas['test']/estatisticas['total']*100:.1f}%)")
    print(f"\n✓ Dataset preparado com sucesso!")
    print(f"Localização: {DATASET_PREPARADO}")
    print("="*70)
    
    return estatisticas

if __name__ == "__main__":
    print("Iniciando preparação do dataset...")
    print(f"Divisão: {TRAIN_SIZE*100:.0f}% train / {VAL_SIZE*100:.0f}% val / {TEST_SIZE*100:.0f}% test\n")
    
    estatisticas = preparar_dataset()
