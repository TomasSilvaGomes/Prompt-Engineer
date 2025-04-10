import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
import umap.umap_ as umap
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50


# Redutor de dimensionalidade
reducer = umap.UMAP()

# ----------------------------------------------- Importações dos dados ------------------------------------------------------------------- #

img_dir = "both_eyes"
csv_file = "comparacoes_10000_shuffled.csv"



# ---------------------------------------------- Funções de carregamento e pré-processamento ---------------------------------------------- #
# load images using CV2 instead of PIL
def load_and_preprocess_image(img_path, target_size=(104, 300)):
    img = image.load_img(img_path, target_size=target_size).convert('RGB')
    img_array = image.img_to_array(img) / 255.0
    return img_array


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


def concatenate_images(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    img1 = cv2.resize(img1, (300, 104)) # the original size of the images
    img2 = cv2.resize(img2, (300, 104))
    concatenated_img = np.concatenate((img1, img2), axis=-1)
    return concatenated_img

# use the Resnet50 model as a backbone and
def create_base_network(input_shape):
    base_model = ResNet50(weights=None , include_top=False, input_shape=input_shape)
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

    embending_model = models.Model(inputs=base_model.input, outputs=embending_output)

    return model, embending_model

def evaluate_thresholds(y_true, y_pred, thresholds):
    for threshold in thresholds:
        # Convertendo as previsões para 0 ou 1 com base no threshold
        y_pred_binary = (y_pred >= threshold).astype(int)

        # Calculando o False Acceptance Rate (FAR)
        FAR = np.mean((y_pred_binary == 1) & (y_true == 0))

        # Calculando o False Rejection Rate (FRR)
        FRR = np.mean((y_pred_binary == 0) & (y_true == 1))

        print(f'Threshold: {threshold:.2f} | Falsos positivos (FAR): {FAR:.4f} | Falsos negativos (FRR): {FRR:.4f}')


img1_paths, img2_paths, labels, fases = load_csv(csv_file)
X_train, X_val, X_test = split_data(img1_paths, img2_paths, labels, fases)
X_train_concat = np.array([concatenate_images(img1, img2) for img1, img2 in zip(X_train[0], X_train[1])])
X_val_concat = np.array([concatenate_images(img1, img2) for img1, img2 in zip(X_val[0], X_val[1])])
X_test_concat = np.array([concatenate_images(img1, img2) for img1, img2 in zip(X_test[0], X_test[1])])

print(f"Shape das imagens{X_train_concat.shape}")
# i want to see an image concatenated
img1 = X_train_concat[0][..., :3]
img2 = X_train_concat[0][..., 3:]
side_by_side = np.concatenate((img1, img2), axis=1)  # eixo horizontal

plt.imshow(side_by_side)
plt.title("Duas imagens concatenadas")
plt.show()


model, embending_model = create_base_network((104, 300, 6))
# do me a summary of the model
Model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train_concat, X_train[2],
                    validation_data=(X_test_concat, X_test[2]),
                    batch_size=8, epochs=30, callbacks=[early_stopping])

# Avaliação do modelo
y_pred = model.predict(X_test_concat)
y_pred = (y_pred > 0.5).astype(int)
precision = precision_score(X_test[2], y_pred)
recall = recall_score(X_test[2], y_pred)
f1 = f1_score(X_test[2], y_pred)

print(f'Precisão: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Definindo uma lista de thresholds para testar
thresholds = np.arange(0.0, 1.1, 0.1)
evaluate_thresholds(test_data[2],y_pred, thresholds)

test_embeddings = embedding_model.predict(X_test_concat)
umap = umap.UMAP(random_state=42)
embedding_2d = umap.fit_transform(test_embeddings)



# Plotar a curva ROC
fpr, tpr, thresholds = roc_curve(X_test[2], y_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'r--')  # linha de referência
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.figure()


plt.figure(figsize=(10, 8))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=X_test[2], cmap='coolwarm', s=10, alpha=0.7)
plt.title('Visualização dos Embeddings com UMAP')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar(label='Classe (0 = não idênticas, 1 = idênticas)')
plt.grid(True)
plt.show()



