import base64
import os
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
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM

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

def justify_with_clip_llm(pil_image, prompt):
    # Extrair embedding da imagem com CLIP
    inputs = clip_processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)

    # Gerar prompt textual final
    final_prompt = (
        f"{prompt}\n\nDescrição da imagem codificada: {image_features[0].tolist()[:10]}...\n"
        "Explica com base visual concreta se são ou não da mesma pessoa."
    )

    # Codificar e gerar resposta com LLM
    input_ids = llm_tokenizer.encode(final_prompt, return_tensors="pt").to(device)
    output = llm_model.generate(input_ids, max_new_tokens=100, do_sample=True, top_k=50)

    decoded = llm_tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split(final_prompt)[-1].strip()




# ----------------------------------------------- Carregamento e pré-processamento da imagem para BLIP2 ------------------------------------- #
# Aqui, usamos a primeira linha do dataframe df
first_row = df.iloc[0]
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


# Criar o prompt com base na classificação
prompt = (
    "A imagem mostra duas regiões dos olhos extraídas de duas pessoas, Compara cuidadosamente a forma dos olhos, as sobrancelhas e o espaçamento entre eles. Justifica com detalhes visuais concretos, porque é que pertecem à mesma pessoa. "
    if label == 1 else
    "A imagem mostra duas regiões dos olhos extraídas de duas pessoas. Compara cuidadosamente a forma dos olhos, as sobrancelhas e o espaçamento entre eles. Justifica com detalhes visuais concretos, porque é que não pertecem à mesma pessoa. "
)

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP: carregar modelo e processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# LLM: modelo de linguagem (podes trocar por mistral ou outro se quiseres local)
llm_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
llm_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

response = justify_with_clip_llm(concat_pil, prompt)
print(response)

