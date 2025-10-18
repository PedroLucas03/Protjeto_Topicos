"""
Versão Corrigida do Modelo EfficientNet-B0 - Dataset Reduzido
============================================================

Esta versão corrige os problemas críticos identificados:
1. Remove normalização duplicada (EfficientNet usa preprocess_input)
2. Corrige mapeamento de classes
3. Usa learning rates apropriados
4. Data augmentation mais conservador
"""

import os
import json
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, 
    TensorBoard
)

class EfficientNetModeloCorrigido:
    def __init__(self):

        self.DATASET_PATH = "dataset_30_racas"
        self.IMG_SIZE = (224, 224)
        self.BATCH_SIZE = 16 
        self.NUM_CLASSES = 30
        
        # Criar diretório para logs
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
    def criar_geradores(self):
        """Cria geradores de dados CORRIGIDOS"""
        
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
        """Cria modelo EfficientNet-B0 otimizado"""
        
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
        
        print(f"Parâmetros treináveis: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
        
        return model
    
    def configurar_callbacks(self, fase):
        """Configura callbacks para treinamento"""
        
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
    
    def treinar_modelo(self):
        """Treina modelo com abordagem de 3 fases"""
        
        train_gen, val_gen, test_gen = self.criar_geradores()
        
        model = self.criar_modelo()
        
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks_fase1 = self.configurar_callbacks(1)
        
        history_fase1 = model.fit(
            train_gen,
            epochs=15,
            validation_data=val_gen,
            callbacks=callbacks_fase1,
            verbose=1
        )
        
        for key in history.keys():
            history[key].extend(history_fase1.history[key])
        
        base_model = None
        for layer in model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 10:  
                base_model = layer
                break
        
        if base_model is not None:
            base_model.trainable = True
            for layer in base_model.layers[:-40]:  
                layer.trainable = False
            print("Fine-tuning: Últimas 40 camadas desbloqueadas")
        else:
            print("AVISO: Modelo base não encontrado, usando fine-tuning completo")
        
        model.compile(
            optimizer=Adam(learning_rate=1e-4),  
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks_fase2 = self.configurar_callbacks(2)
        
        history_fase2 = model.fit(
            train_gen,
            epochs=20,
            validation_data=val_gen,
            callbacks=callbacks_fase2,
            verbose=1
        )
        
        for key in history.keys():
            history[key].extend(history_fase2.history[key])

        if base_model is not None:
            for layer in base_model.layers:
                layer.trainable = True
            print("Todas as camadas desbloqueadas para fine-tuning completo")
        else:
            for layer in model.layers:
                layer.trainable = True
        
        model.compile(
            optimizer=Adam(learning_rate=5e-5), 
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks_fase3 = self.configurar_callbacks(3)
        
        history_fase3 = model.fit(
            train_gen,
            epochs=15,
            validation_data=val_gen,
            callbacks=callbacks_fase3,
            verbose=1
        )
        
        for key in history.keys():
            history[key].extend(history_fase3.history[key])
        
        model.save('models/modelo_efficientnet_corrigido_final.h5')
        
        with open('history_efficientnet_corrigido.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print("\nTreinamento concluído!")
        print("Modelo salvo: models/modelo_efficientnet_corrigido_final.h5")
        print("História salva: history_efficientnet_corrigido.json")
        
        print("\nAvaliando modelo no conjunto de teste...")
        test_loss, test_acc = model.evaluate(test_gen, verbose=1)
        print(f"Acurácia de teste: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Loss de teste: {test_loss:.4f}")
        
        return model, history

def main():
    """Função principal"""
    
    trainer = EfficientNetModeloCorrigido()
    model, history = trainer.treinar_modelo()
    
    best_val_acc = max(history['val_accuracy'])
    print(f"Concluído - Acurácia: {best_val_acc:.2%}")

if __name__ == "__main__":
    main()