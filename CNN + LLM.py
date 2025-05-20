import base64
from io import BytesIO
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import umap.umap_ as umap
from PIL import Image
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import os
import requests
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import time



os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Redutor de dimensionalidade
reducer = umap.UMAP()

# ----------------------------------------------- Importações dos dados ------------------------------------------------------------------- #

img_dir = "both_eyes"
csv_file = "comparacoes_10000_shuffled.csv"
#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
api = "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"


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

# -----------------------------------------------  Chamada do modelo ----------------------------------------------------- #
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


# Função para carregar e pré-processar imagens
def preprocess_image(img_path1, img_path2, target_size=(300, 104)):
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, target_size)
    img2 = cv2.resize(img2, target_size)

    # Concatenar as duas imagens
    concat_image = np.concatenate((img1, img2), axis=-1)
    concat_image = np.clip(concat_image, 0, 255)
    concat_image = concat_image / 255.0  # Normalizar para [0, 1]
    concat_image = np.expand_dims(concat_image, axis=-1)  # Formato (altura, largura, 2)

    return concat_image

# Função para converter imagem para base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def justify_with_internvl(pil_image, prompt, openrouter_api_key):
    # Converter imagem PIL para base64
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_data_url = f"data:image/png;base64,{img_b64}"

    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
        # Estes são opcionais:
        "X-Title": "CNN+LLM Justification Tool"         # opcional
    }

    payload = {
        "model": "opengvlab/internvl3-2b:free",  # modelo correto
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]
            }
        ],
        "max_tokens": 150
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise RuntimeError(f"Erro na API OpenRouter: {response.status_code}\n{response.text}")





# ----------------------------------------------- Carregamento e pré-processamento da imagem para BLIP2 ------------------------------------- #
# Aqui, usamos a primeira linha do dataframe df
first_row = df.iloc[1]
img1_path = os.path.join(img_dir, first_row['img1'])
img2_path = os.path.join(img_dir, first_row['img2'])
label = int(first_row['prediction'])  # usa prediction do modelo binário (0 ou 1)

# Pré-processar as imagens
processed_image = preprocess_image(img1_path, img2_path)

# Converter a imagem concatenada para formato adequado para o modelo (RGB)
concat_image = (processed_image * 255).astype(np.uint8)

# Se for uma imagem monocromática (1 canal), convertê-la para RGB (3 canais)
if concat_image.ndim == 3 and concat_image.shape[-1] == 1:
    concat_image = np.repeat(concat_image, 3, axis=-1)  # Replicar o canal para 3

# Agora, crie o objeto PIL Image
concat_pil = Image.fromarray(concat_image)
concat_pil.show()

prompt = (
    "Porque é que as imagens pertencem à mesma pessoa?"
    if label == 1 else
    "Porque é que as imagens não pertencem à mesma pessoa?"
)


# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------- Justificação com InternVL ----------------------------------------------------- #
# for cicle to generate 50 responses and storage in a list
responses = []
for i in range(15):
    try:
        response = justify_with_internvl(concat_pil, prompt, api)
        responses.append(response)
        print(f"[{i+1}/15] Justificação obtida.")
        time.sleep(20)  # Espera 20 segundos antes da próxima chamada
    except Exception as e:
        print(f"Erro na iteração {i+1}: {e}")
        break


print(responses[0])


# ----------------------------------------------- Text Encoder  ----------------------------------------------------- #
# Codificador de texto
text_encoder = SentenceTransformer('all-MiniLM-L6-v2')  # modelo leve e eficaz

scaler = StandardScaler()
response_embeddings = text_encoder.encode(responses)
if not np.all(np.isfinite(response_embeddings)):
    print("Erro: Existem NaNs ou Inf nos embeddings!")
    response_embeddings = np.nan_to_num(response_embeddings)

response_embeddings = scaler.fit_transform(response_embeddings)
n_samples = len(responses)
perplexity = min(10, n_samples - 1)  # garantir que perplexidade < n_samples

# -------------------------------------------------- Tsne para redução de Visualização ----------------------------------------------------- #
# t-sne with responses
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
X_embedded = tsne.fit_transform(response_embeddings)
plt.figure(figsize=(10, 8))
for i, txt in enumerate(responses):
    if len(txt) > 20:  # Limitar o comprimento do texto para evitar sobreposição
        txt = txt[:20] + "..."
    plt.text(X_embedded[i, 0] + 0.5, X_embedded[i, 1], txt, fontsize=9)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c='blue', marker='o')
plt.title("t-SNE Visualization of Justification Responses")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()

