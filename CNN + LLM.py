import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications import VGG16
import umap.umap_ as umap
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import load_model


# Redutor de dimensionalidade
reducer = umap.UMAP()

# ----------------------------------------------- Importações dos dados ------------------------------------------------------------------- #

img_dir = "both_eyes"
csv_file = "comparacoes_10000_shuffled.csv"



# ---------------------------------------------- Funções de carregamento e pré-processamento ---------------------------------------------- #
def load_and_filter_image(img_path, target_size=(300, 104), method='sobel'):
    # Carregar em escala de cinzentos
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)

    # Aplicar CLAHE (opcional, melhora contraste)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Aplicar filtro de realce
    if method == 'sobel':
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        filtered = np.sqrt(sobelx**2 + sobely**2)
    elif method == 'laplacian':
        filtered = cv2.Laplacian(img, cv2.CV_64F)
    else:
        raise ValueError("Método não reconhecido. Usa 'sobel' ou 'laplacian'.")

    # Normalizar para [0, 1]
    filtered = np.clip(filtered, 0, 255)
    filtered = filtered / 255.0
    filtered = np.expand_dims(filtered, axis=-1)  # shape (104, 300, 1)

    return filtered


def load_csv(csv_file):
    df = pd.read_csv(csv_file)
    img1_paths = [os.path.join(img_dir, img) for img in df['img1'].values]
    img2_paths = [os.path.join(img_dir, img) for img in df['img2'].values]
    labels = df['identicas'].values
    fases = df['fase'].astype(int).values
    return np.array(img1_paths), np.array(img2_paths), np.array(labels), np.array(fases)


def split_data(img1_paths, img2_paths, labels, fases):
    train_idx = fases == 0
    val_idx = fases == 1
    test_idx = fases == 2
    return (
        (img1_paths[train_idx], img2_paths[train_idx], labels[train_idx]),
        (img1_paths[val_idx], img2_paths[val_idx], labels[val_idx]),
        (img1_paths[test_idx], img2_paths[test_idx], labels[test_idx])
    )


def concatenate_filtered_images(img1, img2, method='sobel'):
    img1 = load_and_filter_image(img1, method=method)
    img2 = load_and_filter_image(img2, method=method)
    return np.concatenate((img1, img2), axis=-1)  # shape (104, 300, 2)


def create_base_network(input_shape):
    base_model = VGG16(weights=None , include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    embending_output = x

    output = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

    embendding_model = models.Model(inputs=base_model.input, outputs=embending_output)

    return model, embendding_model


# ----------------------------------------------- Carregamento e pré-processamento dos dados ------------------------------------------ #
img1_paths, img2_paths, labels, fases = load_csv(csv_file)
X_train, X_val, Y_test = split_data(img1_paths, img2_paths, labels, fases)
X_train_concat = np.array([concatenate_filtered_images(img1, img2, method='sobel') for img1, img2 in zip(X_train[0], X_train[1])])
X_val_concat = np.array([concatenate_filtered_images(img1, img2, method='sobel') for img1, img2 in zip(X_val[0], X_val[1])])
Y_test_concat = np.array([concatenate_filtered_images(img1, img2, method='sobel') for img1, img2 in zip(Y_test[0], Y_test[1])])




print(f"Shape das imagens{X_train_concat.shape}")

# ----------------------------------------------- Criação e treino do modelo ----------------------------------------------------- #
model = load_model("modelo_treinado4.h5")
model.summary()

# ----------------------------------------------- Avaliação do modelo ----------------------------------------------------- #
# Avaliação do modelo
y_pred = model.predict(Y_test_concat)
y_pred_bin = (y_pred > 0.40).astype(int)


def get_image_names_and_predictions(img_path1, img_path2, predictions):
    results = []
    img_path1 = [os.path.basename(path) for path in img_path1]
    img_path2 = [os.path.basename(path) for path in img_path2]
    predictions = [pred[0] for pred in predictions]
    for img_path1, img_path2, pred in zip(img_path1,img_path2, predictions):
        results.append((img_path1,img_path2, int(pred)))
    return results


image_names_and_predictions = get_image_names_and_predictions(Y_test[0][:100], Y_test[1][:100], y_pred_bin)

print(f"Imagens e respetivas previsões: {image_names_and_predictions}")

df = pd.DataFrame(image_names_and_predictions, columns=['img1', 'img2', 'prediction'])