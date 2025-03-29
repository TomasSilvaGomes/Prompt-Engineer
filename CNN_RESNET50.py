import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from PIL import Image

IMAGE_DIR = "both_eyes"
CSV_PATH = "comparacoes_10000_shuffled.csv"
IMG_SIZE = (104, 300)
BATCH_SIZE = 8

df = pd.read_csv(CSV_PATH)

# lê as imagens e normaliza
def load_image(image_name):
    image_path = os.path.join(IMAGE_DIR, image_name)
    image = Image.open(image_path).convert('RGB')
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0  # Normalização
    return image

# cria os pares de imagens
X1, X2, y = [], [], []
for _, row in df.iterrows():
    img1, img2, label = row['img1'], row['img2'], row['identicas']
    X1.append(load_image(img1))
    X2.append(load_image(img2))
    y.append(label)

train_idx = df[df['fase'] == 0].index
test_idx = df[df['fase'] == 2].index
val_idx = df[df['fase'] == 1].index

X1, X2, y = np.array(X1), np.array(X2), np.array(y)
X1_train, X2_train, y_train = X1[train_idx], X2[train_idx], y[train_idx]
X1_val, X2_val, y_val = X1[val_idx], X2[val_idx], y[val_idx]
X1_test, X2_test, y_test = X1[test_idx], X2[test_idx], y[test_idx]

# concatenar os arrays das imagens comparadas no eixo da profundidade
X_train = np.concatenate([X1_train, X2_train], axis=-1)
X_val = np.concatenate([X1_val, X2_val], axis=-1)
X_test = np.concatenate([X1_test, X2_test], axis=-1)

# Usar ResNet50 como base da CNN
base_cnn = ResNet50(weights='imagenet', include_top=False, input_shape=(104,300,3))
base_cnn.trainable = True

# Criar um modelo para capturar as representações latentes (camada GlobalAveragePooling2D)
latent_model = models.Model(inputs=base_cnn.input, outputs=base_cnn.output)

# Criar o modelo de classificação final
model = models.Sequential([
    base_cnn,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=30, callbacks=[early_stopping])


latent_test = latent_model.predict(y_test, batch_size=BATCH_SIZE)

# Reduzir as dimensões com t-SNE
tsne = TSNE(n_components=2, random_state=42)
latent_tsne = tsne.fit_transform(latent_test)

# Visualização com t-SNE
plt.figure(figsize=(10, 8))
plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=y_test, cmap='viridis', s=50)
plt.colorbar()
plt.title('T-SNE (Test set)')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.show()

# Avaliação do modelo no conjunto de teste
y_pred = model.predict(X_test, batch_size=BATCH_SIZE).ravel()
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print(f"Test accuracy: {test_acc:.3f}")
print(f"Test loss: {test_loss:.3f}")
