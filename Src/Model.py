import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class DeforestationModel:
    def __init__(self, model_type='cnn'):
        self.model_type = model_type
        self.model = None

    def build_cnn(self, input_shape, num_classes=3):
        """Constrói uma Rede Neural Convolucional para classificação de imagens."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Treina o modelo com os dados fornecidos."""
        if self.model_type == 'cnn':
            input_shape = X_train.shape[1:]
            self.model = self.build_cnn(input_shape, num_classes=len(np.unique(y_train)))
            self.model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
        else:
            raise ValueError(f"Tipo de modelo '{self.model_type}' não suportado.")

    def predict(self, image_data):
        """Faz a previsão em uma única imagem."""
        if self.model is None:
            raise ValueError("Modelo não foi treinado ou carregado.")
        
        # Adiciona uma dimensão de lote para a previsão
        if len(image_data.shape) == 3:
            image_data = np.expand_dims(image_data, axis=0)

        prediction = self.model.predict(image_data)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]

        class_map = {0: 'Preservada', 1: 'Em Risco', 2: 'Desmatada'}
        status = class_map.get(predicted_class, 'Desconhecido')
        
        return {'status': status, 'confidence': float(confidence)}

    def load_model(self, filepath):
        """Carrega um modelo salvo do disco."""
        if self.model_type == 'cnn':
            self.model = tf.keras.models.load_model(filepath)
        else:
            raise ValueError(f"Tipo de modelo '{self.model_type}' não suportado para carregamento.")

    def save_model(self, filepath):
        """Salva o modelo treinado no disco."""
        if self.model:
            self.model.save(filepath)
        else:
            raise ValueError("O modelo não foi treinado e não pode ser salvo.")
