import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.applications import MobileNetV2
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam

# Definir diretório das imagens
img_dir = "both_eyes"

# Função para carregar e processar imagens
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalização
    return img_array

# Função para carregar dados do CSV
def load_data_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    img1_paths = [os.path.join(img_dir, img) for img in df['img1'].values]
    img2_paths = [os.path.join(img_dir, img) for img in df['img2'].values]
    labels = df['identicas'].values  # 0 = diferente, 1 = mesma pessoa
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

# Função de perda contrastiva
def contrastive_loss(y_true, y_pred, margin=1.0):
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# Criar a rede siamesa com ResNet50 como backbone
def create_siamese_network(input_shape=(224, 224, 3)):
    base_model = MobileNetV2 (weights='imagenet', include_top=False, input_shape=input_shape)

    # Criar modelo base
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(224, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    model = models.Model(inputs, x)

    # Criar entradas para a rede siamesa
    inputA = layers.Input(shape=input_shape)
    inputB = layers.Input(shape=input_shape)

    encodedA = model(inputA)
    encodedB = model(inputB)

    # Distância entre embeddings
    subtracted = layers.Subtract()([encodedA, encodedB])
    abs_distance = layers.Activation('relu')(subtracted)
    output = layers.Dense(1, activation='sigmoid')(abs_distance)

    siamese_model = models.Model(inputs=[inputA, inputB], outputs=output)
    siamese_model.compile(optimizer=Adam(learning_rate=0.00001), loss=contrastive_loss, metrics=['accuracy'])

    return siamese_model

# Carregar os dados
df_csv = 'comparacoes_10000_shuffled.csv'
img1_paths, img2_paths, labels, fases = load_data_from_csv(df_csv)
train_data, val_data, test_data = split_data(img1_paths, img2_paths, labels, fases)

# Carregar imagens
train_img1 = np.array([load_and_preprocess_image(p) for p in train_data[0]])
train_img2 = np.array([load_and_preprocess_image(p) for p in train_data[1]])
test_img1 = np.array([load_and_preprocess_image(p) for p in test_data[0]])
test_img2 = np.array([load_and_preprocess_image(p) for p in test_data[1]])

# Criar o modelo
model = create_siamese_network()

# Treinar o modelo
history = model.fit(
    [train_img1, train_img2], train_data[2],
    validation_data=([test_img1, test_img2], test_data[2]),
    batch_size=32, epochs=10, verbose=1
)

# Calcular métricas no conjunto de teste
y_pred = model.predict([test_img1, test_img2]).flatten()
y_pred_binary = (y_pred >= 0.5).astype(int)  # Converter para 0 ou 1

precision = precision_score(test_data[2], y_pred_binary)
recall = recall_score(test_data[2], y_pred_binary)
f1 = f1_score(test_data[2], y_pred_binary)

print(f'Precisão: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')

# Calcular a curva ROC
fpr, tpr, _ = roc_curve(test_data[2], y_pred)
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