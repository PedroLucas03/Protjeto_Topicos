import os
import json

def listar_racas(dataset_path='dataset'):
    """Lista todas as raças disponíveis"""
    train_path = os.path.join(dataset_path, 'train')
    racas = sorted(os.listdir(train_path))
    
    print(f"Total de raças: {len(racas)}\n")
    
    for i, raca in enumerate(racas, 1):
        nome = raca.split('-', 1)[1].replace('_', ' ') if '-' in raca else raca
        print(f"{i:3d}. {nome}")
    
    return racas

def estatisticas_dataset(dataset_path='dataset'):
    """Mostra estatísticas do dataset preparado"""
    
    stats_file = os.path.join(dataset_path, 'dataset_stats.json')
    
    if not os.path.exists(stats_file):
        print("ERRO: Arquivo de estatísticas não encontrado!")
        print("Execute primeiro: python preparar_dataset.py")
        return
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    print("="*70)
    print("ESTATÍSTICAS DO DATASET")
    print("="*70)
    print(f"\nTotal de imagens: {stats['total']:,}")
    print(f"\nTreinamento:  {stats['train']:,} ({stats['train']/stats['total']*100:.1f}%)")
    print(f"Validação:    {stats['validation']:,} ({stats['validation']/stats['total']*100:.1f}%)")
    print(f"Teste:        {stats['test']:,} ({stats['test']/stats['total']*100:.1f}%)")
    print("="*70)
    
    return stats

def listar_modelos_treinados(models_dir='models'):
    """Lista modelos treinados disponíveis"""
    
    if not os.path.exists(models_dir):
        print("Pasta de modelos não encontrada!")
        print("Nenhum modelo foi treinado ainda.")
        return []
    
    modelos = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    
    if not modelos:
        print("Nenhum modelo treinado encontrado!")
        return []
    
    print("="*70)
    print("MODELOS TREINADOS")
    print("="*70)
    
    for i, modelo in enumerate(modelos, 1):
        size = os.path.getsize(os.path.join(models_dir, modelo)) / (1024*1024)
        print(f"{i}. {modelo:40s} ({size:.1f} MB)")
    
    print("="*70)
    
    return modelos

def resumo_treinamento(model_name='resnet50'):
    """Mostra resumo do último treinamento"""
    
    history_file = f'history_{model_name}.json'
    
    if not os.path.exists(history_file):
        print(f"Histórico não encontrado: {history_file}")
        print("O modelo ainda não foi treinado.")
        return None
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    print("="*70)
    print(f"RESUMO DO TREINAMENTO - {model_name.upper()}")
    print("="*70)
    
    print(f"\nNúmero de épocas: {len(history['accuracy'])}")
    
    print(f"\nAcurácia:")
    print(f"  Train:      {history['accuracy'][-1]*100:.2f}%")
    print(f"  Validation: {history['val_accuracy'][-1]*100:.2f}%")
    print(f"  Melhor Val: {max(history['val_accuracy'])*100:.2f}% (época {history['val_accuracy'].index(max(history['val_accuracy']))+1})")
    
    print(f"\nTop-5 Accuracy:")
    print(f"  Train:      {history['top_5_accuracy'][-1]*100:.2f}%")
    print(f"  Validation: {history['val_top_5_accuracy'][-1]*100:.2f}%")
    print(f"  Melhor Val: {max(history['val_top_5_accuracy'])*100:.2f}%")
    
    print(f"\nLoss:")
    print(f"  Train:      {history['loss'][-1]:.4f}")
    print(f"  Validation: {history['val_loss'][-1]:.4f}")
    print(f"  Menor Val:  {min(history['val_loss']):.4f} (época {history['val_loss'].index(min(history['val_loss']))+1})")
    
    print("="*70)
    
    return history

def comparar_modelos():
    """Compara resultados de todos os modelos treinados"""
    
    import glob
    
    history_files = glob.glob('history_*.json')
    
    if not history_files:
        print("❌ Nenhum histórico de treinamento encontrado!")
        return
    
    print("="*70)
    print("COMPARAÇÃO DE MODELOS")
    print("="*70)
    
    results = []
    
    for hist_file in history_files:
        model_name = hist_file.replace('history_', '').replace('.json', '')
        
        with open(hist_file, 'r') as f:
            history = json.load(f)
        
        results.append({
            'modelo': model_name,
            'val_acc': max(history['val_accuracy']) * 100,
            'val_top5': max(history['val_top_5_accuracy']) * 100,
            'val_loss': min(history['val_loss']),
            'epocas': len(history['accuracy'])
        })
    
    # Ordenar por val_acc
    results.sort(key=lambda x: x['val_acc'], reverse=True)
    
    print(f"\n{'Modelo':<20} {'Val Acc':<10} {'Top-5':<10} {'Val Loss':<10} {'Épocas':<10}")
    print("-"*70)
    
    for r in results:
        print(f"{r['modelo']:<20} {r['val_acc']:>8.2f}%  {r['val_top5']:>8.2f}%  {r['val_loss']:>8.4f}  {r['epocas']:>8}")
    
    print("="*70)
    
    return results

def verificar_gpu():
    """Verifica se GPU está disponível"""
    try:
        import tensorflow as tf
        
        print("="*70)
        print("VERIFICAÇÃO DE GPU")
        print("="*70)
        
        print(f"\nTensorFlow versão: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"\n✓ {len(gpus)} GPU(s) detectada(s):")
            for i, gpu in enumerate(gpus, 1):
                print(f"  {i}. {gpu.name}")
                
            # Verificar memória
            for gpu in gpus:
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    print(f"\nDetalhes: {details}")
                except:
                    pass
        else:
            print("\n Nenhuma GPU detectada!")
            print("O treinamento será realizado na CPU (mais lento).")
            print("\nPara usar GPU, instale:")
            print(" CUDA Toolkit")
            print(" cuDNN")
            print("  Tensorflow-gpu")
        
        print("="*70)
        
        return len(gpus) > 0
        
    except ImportError:
        print("TensorFlow não está instalado!")
        print("Execute: pip install tensorflow")
        return False

def menu_principal():
    """Menu interativo principal"""
    
    while True:
        print("\n" + "="*70)
        print("CLASSIFICADOR DE RAÇAS DE CÃES - MENU PRINCIPAL")
        print("="*70)
        print("\n1. Verificar GPU")
        print("2. Estatísticas do Dataset")
        print("3. Listar Raças")
        print("4. Listar Modelos Treinados")
        print("5. Resumo do Último Treinamento")
        print("6. Comparar Modelos")
        print("7. Treinar Novo Modelo")
        print("8. Fazer Predições")
        print("9. Visualizar Resultados")
        print("0. Sair")
        
        escolha = input("\nEscolha uma opção: ").strip()
        
        if escolha == '1':
            verificar_gpu()
        elif escolha == '2':
            estatisticas_dataset()
        elif escolha == '3':
            listar_racas()
        elif escolha == '4':
            listar_modelos_treinados()
        elif escolha == '5':
            model_name = input("Nome do modelo (padrão: resnet50): ").strip() or 'resnet50'
            resumo_treinamento(model_name)
        elif escolha == '6':
            comparar_modelos()
        elif escolha == '7':
            print("\nIniciando treinamento...")
            print("Execute: python modelo_transfer_learning.py")
        elif escolha == '8':
            print("\nSistema de predições...")
            print("Execute: python preditor.py")
        elif escolha == '9':
            model_name = input("Nome do modelo (padrão: resnet50): ").strip() or 'resnet50'
            print(f"\n Visualizando resultados de {model_name}...")
            print(f"Execute: python visualizar_resultados.py {model_name}")
        elif escolha == '0':
            print("\n Até logo!")
            break
        else:
            print("\nOpção inválida!")
        
        input("\nPressione ENTER para continuar...")

if __name__ == "__main__":
    menu_principal()
