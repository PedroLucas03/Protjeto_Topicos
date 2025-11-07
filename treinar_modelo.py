import os
import json
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, 
    TensorBoard
)

class EfficientNetModeloCorrigido:
    def __init__(self):

        self.DATASET_PATH = "dataset"
        self.IMG_SIZE = (224, 224)
        self.BATCH_SIZE = 16 
        self.NUM_CLASSES = 120
        self.CHECKPOINT_FILE = "checkpoint_treinamento.json"
        self.MODEL_CHECKPOINT = "models/checkpoint_modelo.h5"
        
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
    
    def salvar_checkpoint(self, fase, epoca, history, modelo):
        checkpoint_data = {
            'fase': fase,
            'epoca': epoca,
            'history': history,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        modelo.save(self.MODEL_CHECKPOINT)
    
    def carregar_checkpoint(self):
        if os.path.exists(self.CHECKPOINT_FILE):
            with open(self.CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
            
            print(f"\nCheckpoint: Fase {checkpoint['fase']}, Época {checkpoint['epoca']}")
            resposta = input("Continuar? (s/n): ").lower()
            
            if resposta == 's':
                modelo = load_model(self.MODEL_CHECKPOINT)
                return checkpoint, modelo
        
        return None, None
        
    def criar_geradores(self):
        
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input, 
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.DATASET_PATH, 'train'),
            target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        validation_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.DATASET_PATH, 'validation'),
            target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.DATASET_PATH, 'test'),
            target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        class_indices = train_generator.class_indices
        with open('class_mapping_correto.json', 'w') as f:
            json.dump(class_indices, f, indent=2, ensure_ascii=False)
        
        return train_generator, validation_generator, test_generator
    
    def criar_modelo(self):
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.IMG_SIZE, 3)
        )
        
        base_model.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.NUM_CLASSES, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        return model
    
    def configurar_callbacks(self, fase):
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            ModelCheckpoint(
                f'models/best_model_efficientnet_corrigido_fase{fase}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=f'logs/efficientnet_corrigido_{timestamp}_fase{fase}',
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def treinar_modelo(self, epocas_por_vez=5):
        checkpoint, model = self.carregar_checkpoint()
        train_gen, val_gen, test_gen = self.criar_geradores()
        
        if checkpoint:
            fase_atual = checkpoint['fase']
            epoca_inicial = checkpoint['epoca'] + 1
            history = checkpoint['history']
        else:
            fase_atual = 1
            epoca_inicial = 0
            history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
            model = self.criar_modelo()
        
        fases_config = {
            1: {'max_epocas': 30, 'lr': 1e-3, 'descricao': 'FEATURE EXTRACTION', 'descongelar': None},
            2: {'max_epocas': 40, 'lr': 1e-4, 'descricao': 'FINE-TUNING PARCIAL', 'descongelar': -40},
            3: {'max_epocas': 50, 'lr': 5e-5, 'descricao': 'FINE-TUNING COMPLETO', 'descongelar': 'all'}
        }
        
        while fase_atual <= 3:
            config = fases_config[fase_atual]
            print(f"\nFase {fase_atual}: {config['descricao']} - Época {epoca_inicial}/{config['max_epocas']}")
            
            if epoca_inicial == 0: 
                if config['descongelar'] is not None:
                    base_model = None
                    for layer in model.layers:
                        if hasattr(layer, 'layers') and len(layer.layers) > 10:
                            base_model = layer
                            break
                    
                    if base_model:
                        base_model.trainable = True
                        if config['descongelar'] == -40:
                            for layer in base_model.layers[:-40]:
                                layer.trainable = False
                
                model.compile(
                    optimizer=Adam(learning_rate=config['lr']),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
            
            epoca_final = min(epoca_inicial + epocas_por_vez, config['max_epocas'])
            
            try:
                for epoca in range(epoca_inicial, epoca_final):
                    hist = model.fit(
                        train_gen,
                        epochs=1,
                        validation_data=val_gen,
                        verbose=1
                    )
                    
                    for key in history.keys():
                        history[key].extend(hist.history[key])
                    
                    self.salvar_checkpoint(fase_atual, epoca, history, model)
                
                if epoca_final >= config['max_epocas']:
                    fase_atual += 1
                    epoca_inicial = 0
                else:
                    epoca_inicial = epoca_final
                    return model, history
                    
            except KeyboardInterrupt:
                print("\nInterrompido - checkpoint salvo")
                return model, history
        
        model.save('models/modelo_efficientnet_final.h5')
        
        with open('history_efficientnet.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        test_loss, test_acc = model.evaluate(test_gen, verbose=1)
        print(f"\nTreinamento completo - Acurácia de teste: {test_acc:.2%}")
        
        if os.path.exists(self.CHECKPOINT_FILE):
            os.remove(self.CHECKPOINT_FILE)
        if os.path.exists(self.MODEL_CHECKPOINT):
            os.remove(self.MODEL_CHECKPOINT)
        
        return model, history

def main():    
    import sys
    
    epocas = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    
    trainer = EfficientNetModeloCorrigido()
    model, history = trainer.treinar_modelo(epocas_por_vez=epocas)
    
    if history['val_accuracy']:
        print(f"Melhor val_acc: {max(history['val_accuracy']):.2%}")

if __name__ == "__main__":
    main()