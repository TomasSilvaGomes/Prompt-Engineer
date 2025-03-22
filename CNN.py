import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.src.applications.resnet import ResNet50
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Definir diretório das imagens
img_dir = "both_eyes"

# Função para carregar e processar imagens
def load_and_preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalização
    return img_array

# Função para carregar dados do CSV
def load_data_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    img1_paths = [os.path.join(img_dir, img) for img in df['img1'].values]
    img2_paths = [os.path.join(img_dir, img) for img in df['img2'].values]
    labels = df['identicas'].values
    fases = df['fase'].astype(int).values  # 0 = treino, 1 = validação, 2 = teste
    return np.array(img1_paths), np.array(img2_paths), np.array(labels), np.array(fases)

# Dividir os dados em treino, validação e teste
def split_data(img1_paths, img2_paths, labels, fases):
    train_idx = fases == 0
    val_idx = fases == 1
    test_idx = fases == 2

    train_data = (img1_paths[train_idx], img2_paths[train_idx], labels[train_idx])
    val_data = (img1_paths[val_idx], img2_paths[val_idx], labels[val_idx])
    test_data = (img1_paths[test_idx], img2_paths[test_idx], labels[test_idx])

    return train_data, val_data, test_data


def euclidean_distance(vectors):
    (featA, featB) = vectors
    sum_square = K.sum(K.square(featA - featB), axis=1, keepdims=True)
    distance = K.sqrt(K.maximum(sum_square, K.epsilon()))  # Removido K.exp(-distance)
    return distance

# Criar a CNN para extração de embeddings
def create_embedding_model(input_shape=(128, 128, 3)):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=True)
    for layer in base_model.layers[:100]:
        layer.trainable = False
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    model = models.Model(inputs, x)
    return model

# Criar o modelo de similaridade com softmax
def create_similarity_network(input_shape=(128, 128, 3)):
    embedding_model = create_embedding_model(input_shape)

    inputA = layers.Input(shape=input_shape)
    inputB = layers.Input(shape=input_shape)

    embeddingA = embedding_model(inputA)
    embeddingB = embedding_model(inputB)

    diff = layers.Lambda(euclidean_distance)([embeddingA, embeddingB])
    output = layers.Dense(1, activation='sigmoid')(diff)

    similarity_model = models.Model(inputs=[inputA, inputB], outputs=output)
    similarity_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return similarity_model

# Carregar os dados
df_csv = 'comparacoes_10000_shuffled.csv'
img1_paths, img2_paths, labels, fases = load_data_from_csv(df_csv)
train_data, val_data, test_data = split_data(img1_paths, img2_paths, labels, fases)

# Carregar imagens
train_img1 = np.array([load_and_preprocess_image(p) for p in train_data[0]])
train_img2 = np.array([load_and_preprocess_image(p) for p in train_data[1]])
test_img1 = np.array([load_and_preprocess_image(p) for p in test_data[0]])
test_img2 = np.array([load_and_preprocess_image(p) for p in test_data[1]])

# Criar o modelo de similaridade
model = create_similarity_network()
model.summary()

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

# Treinar o modelo
history = model.fit(
    [train_img1, train_img2], train_data[2],
    validation_data=([test_img1, test_img2], test_data[2]),
    batch_size=8, epochs=20, verbose=1
)

# Avaliar no conjunto de teste
y_pred = model.predict([test_img1, test_img2])
y_pred_binary = np.argmax(y_pred, axis=1)
y_true_binary = np.argmax(test_data[2], axis=1)

precision = precision_score(y_true_binary, y_pred_binary)
recall = recall_score(y_true_binary, y_pred_binary)
f1 = f1_score(y_true_binary, y_pred_binary)
print(f'Precisão: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')

# Curva ROC
fpr, tpr, _ = roc_curve(test_data[2][:, 1], y_pred[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (Área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate (FAR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()
