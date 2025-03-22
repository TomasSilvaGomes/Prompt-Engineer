import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import roc_curve, auc
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical

# Diretório onde as imagens estão armazenadas
IMAGE_DIR = "both_eyes"
CSV_PATH = "comparacoes_10000_shuffled.csv"
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 30

# Carregar dataset
df = pd.read_csv(CSV_PATH)


# Função para carregar e processar imagens
def load_image(image_name):
    image_path = os.path.join(IMAGE_DIR, image_name)
    image = Image.open(image_path).convert('RGB')
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0  # Normalização
    return image


# Criar dataset de pares de imagens
X1, X2, y = [], [], []
for _, row in df.iterrows():
    img1, img2, label = row['img1'], row['img2'], row['identicas']
    X1.append(load_image(img1))
    X2.append(load_image(img2))
    y.append(label)

X1, X2, y = np.array(X1), np.array(X2), np.array(y)
y = to_categorical(y, num_classes=2)  # Converter rótulos para one-hot encoding

# Separar conjuntos de treino, validação e teste
train_idx = df[df['fase'] == 0].index
test_idx = df[df['fase'] == 2].index
val_idx = df[df['fase'] == 1].index

X1_train, X2_train, y_train = X1[train_idx], X2[train_idx], y[train_idx]
X1_val, X2_val, y_val = X1[val_idx], X2[val_idx], y[val_idx]
X1_test, X2_test, y_test = X1[test_idx], X2[test_idx], y[test_idx]

# Criar base de uma CNN (VGG16 pré-treinada)
base_cnn = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base_cnn.trainable = False


def build_siamese_network():
    input_shape = (*IMG_SIZE, 3)
    base_model = keras.models.Sequential([
        base_cnn,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu')
    ])

    input_1 = keras.Input(shape=input_shape)
    input_2 = keras.Input(shape=input_shape)

    feature_1 = base_model(input_1)
    feature_2 = base_model(input_2)

    # Distância entre embeddings
    distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([feature_1, feature_2])
    output = layers.Dense(2, activation='softmax')(distance)  # Softmax com duas saídas

    model = keras.Model(inputs=[input_1, input_2], outputs=output)
    return model


# Criar e compilar modelo
siamese_model = build_siamese_network()
siamese_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinar modelo
history = siamese_model.fit(
    [X1_train, X2_train], y_train,
    validation_data=([X1_val, X2_val], y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Avaliação no conjunto de teste
preds = siamese_model.predict([X1_test, X2_test])

# Calcular curva ROC
fpr, tpr, _ = roc_curve(y_test[:, 1], preds[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Linha de separação')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()


# Ajuste de limiar de decisão
def evaluate_threshold(threshold):
    predictions = (preds[:, 1] >= threshold).astype(int)
    FAR = np.mean((predictions == 1) & (y_test[:, 1] == 0))
    FRR = np.mean((predictions == 0) & (y_test[:, 1] == 1))
    print(f'Threshold: {threshold:.2f} | Falsos positivos: {FAR:.4f} | Falsos negativos: {FRR:.4f}')


thresholds = 0.22
evaluate_threshold(thresholds)