import os
import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt

class PreditorRacasCachorros:
    def __init__(self):
        self.model_path = "models/modelo_efficientnet_final.h5"
        self.class_mapping_path = "class_mapping_correto.json"
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
            
        if not os.path.exists(self.class_mapping_path):
            raise FileNotFoundError(f"Mapeamento não encontrado: {self.class_mapping_path}")
        
        self.model = load_model(self.model_path)
        
        with open(self.class_mapping_path, 'r', encoding='utf-8') as f:
            self.class_indices = json.load(f)
        
        self.indices_to_classes = {v: k for k, v in self.class_indices.items()}
        
    def preprocessar_imagem(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_original = img.copy()
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img_original
    
    def predizer(self, img_path, top_k=5):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {img_path}")
        
        img_array, img_original = self.preprocessar_imagem(img_path)
        predictions = self.model.predict(img_array, verbose=0)[0]
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            class_code = self.indices_to_classes[idx]
            breed_name = class_code.split('-', 1)[1].replace('_', ' ').title() if '-' in class_code else class_code
            confidence = predictions[idx] * 100
            results.append((breed_name, confidence, class_code))
        
        return results, img_original
    
    def predizer_e_mostrar(self, img_path, top_k=3):
        results, img_original = self.predizer(img_path, top_k)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_original)
        plt.title(f"{os.path.basename(img_path)}", fontsize=12)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        breeds = [result[0] for result in results]
        confidences = [result[1] for result in results]
        colors = ['lightgreen' if c > 50 else 'gold' if c > 25 else 'lightcoral' for c in confidences]
        
        bars = plt.barh(range(len(breeds)), confidences, color=colors)
        plt.yticks(range(len(breeds)), breeds)
        plt.xlabel('Confiança (%)')
        plt.title(f'Top {top_k}', fontsize=12)
        plt.xlim(0, 100)
        
        for bar, conf in zip(bars, confidences):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{conf:.1f}%', va='center', fontweight='bold')
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        output_name = f"predicao_{os.path.splitext(os.path.basename(img_path))[0]}.png"
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        plt.show()
        return results

def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else input("Caminho da imagem: ").strip()
    if not img_path:
        return
    
    try:
        preditor = PreditorRacasCachorros()
        results = preditor.predizer_e_mostrar(img_path, top_k=3)
        if results:
            print(f"{results[0][0]} ({results[0][1]:.1f}%)")
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()