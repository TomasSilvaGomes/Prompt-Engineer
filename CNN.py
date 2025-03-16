import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.applications import ResNet50
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.preprocessing import image

# Definindo o caminho para as imagens
img_dir = "both_eyes"  # Defina o diretório onde as imagens estão armazenadas


def load_and_preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalização
    return img_array


def load_data_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    img1_paths = [os.path.join(img_dir, img) for img in df['img1'].values]
    img2_paths = [os.path.join(img_dir, img) for img in df['img2'].values]
    labels = df['identicas'].values
    fases = df['fase'].astype(int).values  # Certificar que é inteiro
    return np.array(img1_paths), np.array(img2_paths), np.array(labels), np.array(fases)


def split_data(img1_paths, img2_paths, labels, fases):
    train_idx = fases == 0
    val_idx = fases == 1
    test_idx = fases == 2

    train_data = (img1_paths[train_idx], img2_paths[train_idx], labels[train_idx])
    val_data = (img1_paths[val_idx], img2_paths[val_idx], labels[val_idx])
    test_data = (img1_paths[test_idx], img2_paths[test_idx], labels[test_idx])

    return train_data, val_data, test_data


def contrastive_loss(y_true, y_pred, margin=1.0):
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_siamese_network(input_shape=(128, 128, 3)):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)


    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    model = models.Model(inputs, x)

    inputA = layers.Input(shape=input_shape)
    inputB = layers.Input(shape=input_shape)

    encodedA = model(inputA)
    encodedB = model(inputB)

    subtracted = layers.Subtract()([encodedA, encodedB])
    abs_distance = layers.Activation('relu')(subtracted)
    output = layers.Dense(1, activation='sigmoid')(abs_distance)

    siamese_model = models.Model(inputs=[inputA, inputB], outputs=output)
    siamese_model.compile(optimizer='adam', loss=contrastive_loss, metrics=['accuracy'])

    return siamese_model


# Carregar os dados
df_csv = 'comparacoes_10000.csv'
img1_paths, img2_paths, labels, fases = load_data_from_csv(df_csv)
train_data, val_data, test_data = split_data(img1_paths, img2_paths, labels, fases)

# Carregar imagens
train_img1 = np.array([load_and_preprocess_image(p) for p in train_data[0]])
train_img2 = np.array([load_and_preprocess_image(p) for p in train_data[1]])
test_img1 = np.array([load_and_preprocess_image(p) for p in test_data[0]])
test_img2 = np.array([load_and_preprocess_image(p) for p in test_data[1]])

# Criar o modelo
model = create_siamese_network()

# Treinar o modelo e armazenar métricas
history = model.fit(
    [train_img1, train_img2], train_data[2],
    validation_data=([test_img1, test_img2], test_data[2]),
    batch_size=32, epochs=10, verbose=1
)

# Exibir acurácia e loss no treino e teste
train_loss = history.history['loss'][-1]
train_acc = history.history['accuracy'][-1]
test_loss = history.history['val_loss'][-1]
test_acc = history.history['val_accuracy'][-1]

print(f'Acurácia no treino: {train_acc * 100:.2f}%, Loss no treino: {train_loss:.4f}')
print(f'Acurácia no teste: {test_acc * 100:.2f}%, Loss no teste: {test_loss:.4f}')

# Aplicar PCA para visualizar os embeddings
embeddings = model.predict([test_img1, test_img2])
if embeddings.shape[0] > 1:
    n_components = min(2, embeddings.shape[0])
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings)

    # Plot dos embeddings com PCA
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=test_data[2], cmap='coolwarm', alpha=0.6)
    plt.colorbar(label='Classe (0 = Diferente, 1 = Igual)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Visualização dos Embeddings com PCA')
    plt.show()
else:
    print("PCA não pode ser aplicado pois há apenas uma amostra de embeddings.")


# Calcular a curva ROC
fpr, tpr, _ = roc_curve(test_data[2], model.predict([test_img1, test_img2]))
roc_auc = auc(fpr, tpr)

# Plot da curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (Área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate (FAR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()
