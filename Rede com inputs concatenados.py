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
'''
# load images using CV2 instead of PIL
def load_and_preprocess_image(img_path, target_size=(104, 300)):
    img = image.load_img(img_path, target_size=target_size).convert('RGB')
    img_array = image.img_to_array(img) / 255.0
    return img_array

 usar quando for novamente treinar o modelo
'''
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

'''
def concatenate_images(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    img1 = cv2.resize(img1, (300, 104)) # the original size of the images
    img2 = cv2.resize(img2, (300, 104))
    concatenated_img = np.concatenate((img1, img2), axis=-1)
    return concatenated_img
'''

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

def evaluate_thresholds(y_true, y_pred, thresholds):
    for threshold in thresholds:
        # Convertendo as previsões para 0 ou 1 com base no threshold
        y_pred_binary = (y_pred >= threshold).astype(int)

        # Calculando o False Acceptance Rate (FAR)
        FAR = np.mean((y_pred_binary == 1) & (y_true == 0))

        # Calculando o False Rejection Rate (FRR)
        FRR = np.mean((y_pred_binary == 0) & (y_true == 1))

        print(f'Threshold: {threshold:.2f} | Falsos positivos (FAR): {FAR:.4f} | Falsos negativos (FRR): {FRR:.4f}')



# ----------------------------------------------- Carregamento e pré-processamento dos dados ------------------------------------------ #
img1_paths, img2_paths, labels, fases = load_csv(csv_file)
X_train, X_val, Y_test = split_data(img1_paths, img2_paths, labels, fases)
#X_train_concat = np.array([concatenate_images(img1, img2) for img1, img2 in zip(X_train[0], X_train[1])])
#X_val_concat = np.array([concatenate_images(img1, img2) for img1, img2 in zip(X_val[0], X_val[1])])
#Y_test_concat = np.array([concatenate_images(img1, img2) for img1, img2 in zip(Y_test[0], Y_test[1])])
X_train_concat = np.array([concatenate_filtered_images(img1, img2, method='sobel') for img1, img2 in zip(X_train[0], X_train[1])])
X_val_concat = np.array([concatenate_filtered_images(img1, img2, method='sobel') for img1, img2 in zip(X_val[0], X_val[1])])
Y_test_concat = np.array([concatenate_filtered_images(img1, img2, method='sobel') for img1, img2 in zip(Y_test[0], Y_test[1])])




print(f"Shape das imagens{X_train_concat.shape}")

# ----------------------------------------------- Criação e treino do modelo ----------------------------------------------------- #
TRAIN_MODEL = False  # True = treinar o modelo, False = carregar o modelo salvo
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
if TRAIN_MODEL:
    model, embendding_model = create_base_network((104, 300, 2))
    model.summary()

    history = model.fit(X_train_concat, X_train[2],
                        validation_data=(X_val_concat, X_val[2]),
                        batch_size=16, epochs=100, callbacks=[early_stopping])
    model.save("modelo_treinado4.h5")
else:
    model = load_model("modelo_treinado4.h5")
    model.summary()

# ----------------------------------------------- Avaliação do modelo ----------------------------------------------------- #
# Avaliação do modelo
y_pred = model.predict(Y_test_concat)
y_pred_bin = (y_pred > 0.40).astype(int)
precision = precision_score(Y_test[2], y_pred_bin)
recall = recall_score(Y_test[2], y_pred_bin)
f1 = f1_score(Y_test[2], y_pred_bin)

print(f'Precisão no conjunto teste: {precision:.2f}')
print(f'Recall conjunto teste: {recall:.2f}')
print(f'F1 Score no conjunto teste: {f1:.2f}')

# Definindo uma lista de thresholds para testar
#thresholds = np.arange(0.0, 1.1, 0.1)
#evaluate_thresholds(Y_test[2],y_pred, thresholds)


# ----------------------------------------------- Grad-CAM Visualização ----------------------------------------------------- #
last_conv_layer_name = 'block5_conv3'
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Criar um modelo que mapeia a imagem de entrada para as ativações da última camada convolucional
    # e a predição do modelo
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for the input image
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradiente da classe em relação à saída da camada convolucional
    grads = tape.gradient(class_channel, conv_outputs)

    # Vetor médio do gradiente em cada canal
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiplicar cada canal da feature map pelo gradiente médio
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalizar para 0-1 para visualização
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

for i in range(2):
    img_input = Y_test_concat[i:i+1]

    # Geração do heatmap
    heatmap = make_gradcam_heatmap(img_input, model, last_conv_layer_name)

    # Mostrar o heatmap puro
    plt.figure(figsize=(6, 4))
    plt.imshow(heatmap, cmap='jet')
    plt.title(f"Grad-CAM Heatmap for Test Image {i}")
    plt.axis('off')
    plt.show()

    # Assumindo que queres sobrepor na imagem da esquerda (canal 0)
    original_img = img_input[0][..., 0]  # canal esquerdo em cinzento (104, 300)
    original_img = np.uint8(255 * original_img)

    # Converter para 3 canais (BGR) para combinar com o heatmap_color
    original_img_3ch = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    # Redimensionar heatmap para o mesmo tamanho
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Combinar heatmap com imagem original
    superimposed_img = cv2.addWeighted(original_img_3ch, 0.6, heatmap_color, 0.4, 0)

    # Mostrar
    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Grad-CAM Overlay for Test Image {i}")
    plt.axis('off')
    plt.show()

# ----------------------------------------------- UMAP para visualização ----------------------------------------------------- #
embedding_layer_output = model.layers[-3].output  # ajusta se necessário
embedding_model = models.Model(inputs=model.input, outputs=embedding_layer_output)
test_embeddings = embedding_model.predict(Y_test_concat)
umap = umap.UMAP(random_state=42)
embedding_2d = umap.fit_transform(test_embeddings)



# ----------------------------------------------- Visualização dos resultados ----------------------------------------------------- #
fpr, tpr, thresholds = roc_curve(Y_test[2], y_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'r--')  # linha de referência
plt.xlabel('False Positive Rate')
plt.legend()
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.figure()


plt.figure(figsize=(10, 8))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=Y_test[2], cmap='coolwarm', s=10, alpha=0.7)
plt.title('Visualização dos Embeddings com UMAP')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar(label='Classe (0 = não idênticas, 1 = idênticas)')
plt.grid(True)
plt.show()