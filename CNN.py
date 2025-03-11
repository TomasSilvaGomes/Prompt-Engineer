import pandas as pd
from keras.src.applications.resnet import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
import numpy as np
import os

# Definindo o caminho para as imagens
img_dir = "both_eyes"  # Defina o diretório onde as imagens estão armazenadas

# Função para carregar e pré-processar as imagens
def load_and_preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalização
    return img_array


# Função para carregar os dados do CSV e preparar os dados de treino/teste
def load_data_from_csv(csv_file):
    df = pd.read_csv(csv_file)

    # Cria listas para armazenar as imagens e rótulos
    img1_paths = []
    img2_paths = []
    labels = []
    fases = []

    for index, row in df.iterrows():
        img1_path = os.path.join(img_dir, row['img1'])
        img2_path = os.path.join(img_dir, row['img2'])
        label = row['identicas']
        fase = row['fase']

        img1_paths.append(img1_path)
        img2_paths.append(img2_path)
        labels.append(label)
        fases.append(fase)

    return np.array(img1_paths), np.array(img2_paths), np.array(labels), np.array(fases)


# Função para dividir os dados em conjuntos de treino, validação e teste com base na fase
def split_data(img1_paths, img2_paths, labels, fases):
    # Divisão direta com base nos valores da fase
    train_img1 = img1_paths[fases == 0]
    train_img2 = img2_paths[fases == 0]
    train_labels = labels[fases == 0]

    val_img1 = img1_paths[fases == 1]
    val_img2 = img2_paths[fases == 1]
    val_labels = labels[fases == 1]

    test_img1 = img1_paths[fases == 2]
    test_img2 = img2_paths[fases == 2]
    test_labels = labels[fases == 2]

    return (train_img1, train_img2, train_labels), (val_img1, val_img2, val_labels), (test_img1, test_img2, test_labels)


# Função para criar e treinar a rede siamês
def create_siamese_network(input_shape=(128, 128, 6)):
    # Utilizando ResNet50 com 6 canais de entrada (duas imagens RGB)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Congelar as camadas da ResNet
    base_model.trainable = False

    # Construir a rede a partir da ResNet
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())  # Camada de pooling para reduzir a dimensionalidade
    model.add(layers.Dense(128, activation='relu'))  # Camada densa para classificação
    model.add(layers.Dense(1, activation='sigmoid'))  # Saída binária (0 ou 1)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Exemplo de uso

# Carregar os dados do CSV
csv_file = 'comparacoes_10000.csv'
img1_paths, img2_paths, labels, fases = load_data_from_csv(csv_file)

# Dividir os dados em treino, validação e teste com base na fase
train_data, val_data, test_data = split_data(img1_paths, img2_paths, labels, fases)

# Carregar as imagens para o conjunto de teste
test_img1, test_img2, test_labels = test_data
test_img1 = np.array([load_and_preprocess_image(img_path) for img_path in test_img1])
test_img2 = np.array([load_and_preprocess_image(img_path) for img_path in test_img2])

# Concatenar as duas imagens RGB para criar entradas de 6 canais
test_input = np.concatenate([test_img1, test_img2], axis=-1)

# Criar o modelo
model = create_siamese_network()

# Treino do modelo
model.fit(test_input, test_labels, batch_size=32, epochs=10, validation_split=0.2)

# Avaliar o modelo no conjunto de teste
predictions = model.predict(test_input)

# Converter as previsões em rótulos binários (0 ou 1)
predicted_labels = (predictions > 0.5).astype(int)

# Calcular a acurácia
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_labels, predicted_labels)
print(f'Acurácia no conjunto de teste: {accuracy * 100:.2f}%')