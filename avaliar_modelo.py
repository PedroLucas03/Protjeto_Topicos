import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

def avaliar_modelo_corrigido():    
    modelo_path = "models/modelo_efficientnet_corrigido_final.h5"
    dataset_path = "dataset_30_racas"
    
    if not os.path.exists(modelo_path):
        raise FileNotFoundError(f"Modelo não encontrado: {modelo_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")
        
    try:
        model = load_model(modelo_path)
        print(f"Modelo carregado: {modelo_path}")
    except Exception as e:
        print(f"ERRO: Erro ao carregar modelo: {e}")
        return
    
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(dataset_path, 'test'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False  
    )
    
    print(f"Test samples: {test_generator.samples}")
    print(f"Número de classes: {test_generator.num_classes}")
    
    class_mapping_path = "class_mapping.json"
    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, 'r') as f:
            class_indices = json.load(f)
        indices_to_classes = {v: k for k, v in class_indices.items()}
    else:
        print("AVISO: Usando mapeamento do gerador (pode estar incorreto)")
        indices_to_classes = {v: k for k, v in test_generator.class_indices.items()}
    
    print("\nAvaliando modelo...")
    test_loss, test_acc = model.evaluate(test_generator, verbose=1)

    print("\n" + "="*70)
    print("RESULTADOS FINAIS")
    print("="*70)
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    print("\nAnalisando predições por classe...")
    
    test_generator.reset()
    
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    true_classes = test_generator.classes
    
    class_names = [indices_to_classes[i].split('-')[1].replace('_', ' ') for i in range(len(indices_to_classes))]
    
    print("\nRELATORIO DE CLASSIFICACAO:")
    print(classification_report(
        true_classes, 
        predicted_classes, 
        target_names=class_names,
        digits=4
    ))
    
    print("\nACURACIA POR CLASSE:")
    class_accuracies = {}
    class_counts = {}
    
    for i in range(len(indices_to_classes)):
        class_mask = true_classes == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(predicted_classes[class_mask] == true_classes[class_mask])
            class_count = np.sum(class_mask)
            
            class_name = indices_to_classes[i]
            breed_name = class_name.split('-')[1].replace('_', ' ')
            
            class_accuracies[class_name] = class_acc
            class_counts[class_name] = class_count
            
            print(f"   {breed_name:25} - {class_acc:6.1%} ({class_count:2d} amostras)")
    
    sorted_accs = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTOP 5 MELHORES CLASSES:")
    for i, (class_name, acc) in enumerate(sorted_accs[:5]):
        breed_name = class_name.split('-')[1].replace('_', ' ')
        count = class_counts[class_name]
        print(f"   {i+1}. {breed_name:25} - {acc:6.1%} ({count} amostras)")
    
    print("\nTOP 5 PIORES CLASSES:")
    for i, (class_name, acc) in enumerate(sorted_accs[-5:]):
        breed_name = class_name.split('-')[1].replace('_', ' ')
        count = class_counts[class_name]
        print(f"   {i+1}. {breed_name:25} - {acc:6.1%} ({count} amostras)")
    
    accuracies = list(class_accuracies.values())
    zero_acc_classes = [k for k, v in class_accuracies.items() if v == 0.0]
    
    print("\nESTATISTICAS GERAIS:")
    print(f" Acurácia média por classe: {np.mean(accuracies):.2%}")
    print(f" Desvio padrão: {np.std(accuracies):.2%}")
    print(f" Classes com 0% acurácia: {len(zero_acc_classes)}/30")
    print(f" Classes com >50% acurácia: {sum([1 for acc in accuracies if acc > 0.5])}/30")
    
    print("\nCOMPARACAO COM MODELO ANTERIOR:")
    print(" Modelo Anterior (EfficientNet): 12.86%")
    print(f" Modelo Corrigido: {test_acc*100:.2f}%")
    print(f" Melhoria: {(test_acc - 0.1286)*100:+.2f} pontos percentuais")
    
    results = {
        "model_path": modelo_path,
        "dataset": "reduzido_30_classes_corrigido",
        "test_samples": test_generator.samples,
        "num_classes": test_generator.num_classes,
        "loss": float(test_loss),
        "accuracy": float(test_acc),
        "class_accuracies": {k: float(v) for k, v in class_accuracies.items()},
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "statistics": {
            "mean_class_accuracy": float(np.mean(accuracies)),
            "std_class_accuracy": float(np.std(accuracies)),
            "zero_accuracy_classes": len(zero_acc_classes),
            "above_50_percent_classes": sum([1 for acc in accuracies if acc > 0.5])
        }
    }
    
    with open('resultados_modelo_corrigido.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\nResultados salvos em 'resultados_modelo_corrigido.json'")
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.hist(accuracies, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(accuracies), color='red', linestyle='--', linewidth=2, 
                label=f'Média: {np.mean(accuracies):.2%}')
    plt.xlabel('Acurácia por Classe')
    plt.ylabel('Número de Classes')
    plt.title('Distribuição de Acurácias por Classe')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    top_10 = sorted_accs[:10]
    breeds = [name.split('-')[1].replace('_', ' ') for name, _ in top_10]
    accs = [acc for _, acc in top_10]
    
    plt.barh(range(len(breeds)), accs, color='lightgreen')
    plt.yticks(range(len(breeds)), breeds)
    plt.xlabel('Acurácia')
    plt.title('Top 10 Classes - Melhores Acurácias')
    plt.gca().invert_yaxis()
    
    plt.subplot(2, 2, 3)
    top_10_indices = [class_indices[name] for name, _ in top_10]
    
    mask = np.isin(true_classes, top_10_indices)
    true_subset = true_classes[mask]
    pred_subset = predicted_classes[mask]
    
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(top_10_indices)}
    true_remapped = np.array([index_map[idx] for idx in true_subset])
    pred_remapped = np.array([index_map[idx] for idx in pred_subset])
    
    cm = confusion_matrix(true_remapped, pred_remapped)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=False, cmap='Blues', 
                xticklabels=[b[:10] for b in breeds], 
                yticklabels=[b[:10] for b in breeds])
    plt.title('Matriz de Confusão (Top 10 Classes)')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    
    plt.subplot(2, 2, 4)
    models = ['Original\nResNet50', 'Anterior\nEfficientNet', 'Corrigido\nEfficientNet']
    accuracies_comp = [0.1340, 0.1286, test_acc]
    colors = ['red', 'orange', 'green']
    
    bars = plt.bar(models, accuracies_comp, color=colors, alpha=0.7)
    plt.ylabel('Acurácia')
    plt.title('Comparação de Modelos')
    plt.ylim(0, max(accuracies_comp) + 0.1)
    
    for bar, acc in zip(bars, accuracies_comp):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('avaliacao_modelo_corrigido.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Gráficos salvos em 'avaliacao_modelo_corrigido.png'")
    
    return results

def main():
    
    print(f"TensorFlow version: {tf.__version__}")
    if tf.config.list_physical_devices('GPU'):
        print("GPU disponível")
    else:
        print("AVISO: Usando CPU")
        results = avaliar_modelo_corrigido()
    
    if results:
        print("\n" + "="*70)
        print("RESUMO EXECUTIVO")
        print("="*70)
        
        acc = results['accuracy']
        if acc > 0.7:
            print(f"Acurácia de {acc:.2%} - Objetivo alcançado!")
        elif acc > 0.5:
            print(f"Acurácia de {acc:.2%} - Modelo funcional")
        elif acc > 0.3:
            print(f"Acurácia de {acc:.2%} - Ainda há melhorias possíveis")
        else:
            print(f"Acurácia de {acc:.2%} - Modelo precisa de mais ajustes")

if __name__ == "__main__":
    main()