"""
Script para criar um dataset reduzido com as 30 raças mais distintas
para obter melhores resultados de classificação
"""
import os
import shutil
import json
from collections import defaultdict


RACAS_SELECIONADAS = [
    # Cães grandes
    'n02091831-Saluki',           
    'n02090721-Irish_wolfhound',  
    'n02092002-Scottish_deerhound', 
    'n02088094-Afghan_hound',     
    'n02087394-Rhodesian_ridgeback', 
    
    # Cães médios
    'n02100583-vizsla',          
    'n02092339-Weimaraner',      
    'n02089973-English_foxhound', 
    'n02091467-Norwegian_elkhound', 
    'n02093754-Border_terrier',   
    
    # Cães pequenos 
    'n02085620-Chihuahua',       
    'n02112018-Pomeranian',      
    'n02086240-Shih-Tzu',        
    'n02085936-Maltese_dog',    
    'n02087046-toy_terrier',     
    
    # Cães com características únicas
    'n02115641-dingo',          
    'n02116738-African_hunting_dog', 
    'n02093647-Bedlington_terrier', 
    'n02094258-Norwich_terrier',  
    'n02096051-Airedale',        
    
    # Bulldogs e similares
    'n02108089-boxer',           
    'n02108422-bull_mastiff',    
    'n02109047-Great_Dane',      
    
    # Pastores e similares
    'n02106030-collie',          
    'n02105641-Old_English_sheepdog', 
    
    # Spaniels 
    'n02102318-cocker_spaniel',  
    
    # Outros distintos
    'n02099601-golden_retriever', 
    'n02099712-Labrador_retriever', 
    'n02111889-Samoyed',         
    'n02098413-Lhasa'            
]

def criar_dataset_reduzido():
    """Cria um dataset reduzido com raças mais distintas"""
    
    print("="*70)
    print(" RAÇAS DISTINTAS")
    print("="*70)
    
    # Diretórios
    dataset_original = "dataset"
    output_dir = "dataset_30_racas"
    
    # Verificar se dataset original existe
    if not os.path.exists(dataset_original):
        print("Dataset original não encontrado!")
        return
    
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir)
    
    for split in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(output_dir, split))
    

    stats = defaultdict(lambda: defaultdict(int))
    total_copiados = 0
    racas_encontradas = []
    
    print(f"\\nCopiando {len(RACAS_SELECIONADAS)} raças selecionadas...")
    
    # Copiar cada raça selecionada
    for i, raca in enumerate(RACAS_SELECIONADAS, 1):
        print(f"\\n{i:2d}. Processando: {raca}")
        
        raca_encontrada = False
        

        for split in ['train', 'validation', 'test']:
            origem_dir = os.path.join(dataset_original, split, raca)
            destino_dir = os.path.join(output_dir, split, raca)
            
            if os.path.exists(origem_dir):
                raca_encontrada = True
                
                # Copiar diretório
                shutil.copytree(origem_dir, destino_dir)
                
                # Contar arquivos
                num_files = len([f for f in os.listdir(destino_dir) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                
                stats[raca][split] = num_files
                stats[raca]['total'] += num_files
                total_copiados += num_files
                
                print(f"    {split}: {num_files} imagens")
        
        if raca_encontrada:
            racas_encontradas.append(raca)
        else:
            print(f"Raça não encontrada no dataset!")
    
    # Salvar estatísticas
    stats_final = {
        'total_racas': len(racas_encontradas),
        'total_imagens': total_copiados,
        'racas': dict(stats)
    }
    
    # Calcular totais por split
    for split in ['train', 'validation', 'test']:
        stats_final[split] = sum(info[split] for info in stats.values())
    
    with open(os.path.join(output_dir, 'dataset_stats_reduzido.json'), 'w') as f:
        json.dump(stats_final, f, indent=2)
    
    print(f"\\n" + "="*70)
    print("DATASET REDUZIDO CRIADO COM SUCESSO!")
    print("="*70)
    
    print(f"\\n📊 ESTATÍSTICAS:")
    print(f"   • Raças selecionadas: {len(racas_encontradas)}")
    print(f"   • Total de imagens: {total_copiados:,}")
    print(f"   • Train: {stats_final['train']:,}")
    print(f"   • Validation: {stats_final['validation']:,}")
    print(f"   • Test: {stats_final['test']:,}")
    
    # Média por raça
    if len(racas_encontradas) > 0:
        media_por_raca = total_copiados / len(racas_encontradas)
        print(f"   • Média por raça: {media_por_raca:.0f} imagens")
    
    print(f"\nDataset salvo em: {output_dir}/")
    print(f"\nPróximo passo: Treinar modelo com dataset reduzido")
    print(f"  python treinar_modelo.py")

def listar_racas_disponiveis():
    """Lista todas as raças disponíveis no dataset original"""
    
    dataset_original = "dataset/train"
    if not os.path.exists(dataset_original):
        print("Dataset original não encontrado!")
        return
    
    racas = [d for d in os.listdir(dataset_original) 
             if os.path.isdir(os.path.join(dataset_original, d))]
    
    print(f"\\nRAÇAS DISPONÍVEIS NO DATASET ({len(racas)}):")
    print("-" * 50)
    
    for i, raca in enumerate(sorted(racas), 1):
        num_images = len([f for f in os.listdir(os.path.join(dataset_original, raca))
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"{i:3d}. {raca:<35} ({num_images:3d} imagens)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--listar":
        listar_racas_disponiveis()
    else:
        criar_dataset_reduzido()