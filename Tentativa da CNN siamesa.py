import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score

from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import MobileNetV2


# Diretório das imagens
img_dir = "both_eyes"

# Carregamento e processamento das imagens
def load_and_preprocess_image(img_path, target_size=(104, 300)):
    img = image.load_img(img_path, target_size=target_size).convert('RGB')
    img_array = image.img_to_array(img) / 255.0
    return img_array

# Carregamento do CSV
def load_data_from_csv(csv_file):
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

# Função para criar o backbone (ResNet50 sem top)
def create_base_network(input_shape):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    model = models.Model(inputs=base_model.input, outputs=x)
    return model

# Distância euclidiana
def euclidean_distance(vectors):
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# Construção da rede siamesa
def create_siamese_network(input_shape):
    base_network = create_base_network(input_shape)

    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    embedding_a = base_network(input_a)
    embedding_b = base_network(input_b)

    distance = layers.Lambda(euclidean_distance)([embedding_a, embedding_b])
    neg_distance = layers.Lambda(lambda x: -x)(distance)
    output = layers.Dense(1, activation='sigmoid')(neg_distance)

    model = models.Model(inputs=[input_a, input_b], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
    return model

# Carregamento dos dados
df_csv = 'comparacoes_10000_shuffled.csv'
img1_paths, img2_paths, labels, fases = load_data_from_csv(df_csv)
train_data, val_data, test_data = split_data(img1_paths, img2_paths, labels, fases)

# Preprocessamento das imagens
def process_pairs(img_paths1, img_paths2):
    return (
        np.array([load_and_preprocess_image(p) for p in img_paths1]),
        np.array([load_and_preprocess_image(p) for p in img_paths2])
    )

train_img1, train_img2 = process_pairs(train_data[0], train_data[1])
val_img1, val_img2 = process_pairs(val_data[0], val_data[1])
test_img1, test_img2 = process_pairs(test_data[0], test_data[1])

input_shape = (104, 300, 3)
model = create_siamese_network(input_shape)
model.summary()

# Treino
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    [train_img1, train_img2], train_data[2],
    validation_data=([val_img1, val_img2], val_data[2]),
    batch_size=16, epochs=1000, callbacks=[early_stopping]
)

# Avaliação
test_loss, test_acc = model.evaluate([test_img1, test_img2], test_data[2])
print(f"Test Accuracy: {test_acc:.3f}, Test Loss: {test_loss:.3f}")

# Previsões
y_pred = model.predict([test_img1, test_img2])
y_pred_binary = (y_pred >= 0.5).astype(int)

# Métricas
precision = precision_score(test_data[2], y_pred_binary)
recall = recall_score(test_data[2], y_pred_binary)
f1 = f1_score(test_data[2], y_pred_binary)

print(f'Precisão: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')

# Curva ROC
fpr, tpr, _ = roc_curve(test_data[2], y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (Área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend()
plt.show()

# Extração dos embeddings (usando uma das torres)
embedding_model = model.layers[2]  # camada base_network

test_embeddings_a = embedding_model.predict(test_img1)
test_embeddings_b = embedding_model.predict(test_img2)
avg_embeddings = (test_embeddings_a + test_embeddings_b) / 2

# Guardar os embeddings e as labels num CSV
embeddings_df = pd.DataFrame(avg_embeddings)
embeddings_df['label'] = test_data[2]
embeddings_df['img1'] = test_data[0]
embeddings_df['img2'] = test_data[1]

embeddings_df.to_csv("embeddings_teste.csv", index=False)
print("Embeddings guardados em 'embeddings_teste.csv'")


# t-SNE
tsne = TSNE(n_components=2, random_state=42)
Tsne = tsne.fit_transform(avg_embeddings)
plt.figure(figsize=(10, 8))
plt.scatter(Tsne[:, 0], Tsne[:, 1], c=test_data[2], cmap='viridis', s=10)
plt.colorbar(label='Classe (0 = não idênticas, 1 = idênticas)')
plt.title('t-SNE dos Embeddings da Fase de Teste')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()
