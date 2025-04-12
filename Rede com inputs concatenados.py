import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications import VGG16
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
import umap.umap_ as umap
import tensorflow as tf
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
    base_model = VGG16(weights=None , include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base_model.output)
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



# ----------------------------------------------- Carregamento e pré-processamento dos dados ------------------------------------------ #
img1_paths, img2_paths, labels, fases = load_csv(csv_file)
X_train, X_val, Y_test = split_data(img1_paths, img2_paths, labels, fases)
X_train_concat = np.array([concatenate_images(img1, img2) for img1, img2 in zip(X_train[0], X_train[1])])
X_val_concat = np.array([concatenate_images(img1, img2) for img1, img2 in zip(X_val[0], X_val[1])])
Y_test_concat = np.array([concatenate_images(img1, img2) for img1, img2 in zip(Y_test[0], Y_test[1])])

print(f"Shape das imagens{X_train_concat.shape}")
# i want to see an image concatenated
img1 = X_train_concat[0][..., :3]
img2 = X_train_concat[0][..., 3:]
side_by_side = np.concatenate((img1, img2), axis=1)  # eixo horizontal

plt.imshow(side_by_side)
plt.title("Duas imagens concatenadas")
plt.show()



# ----------------------------------------------- Criação e treino do modelo ----------------------------------------------------- #
TRAIN_MODEL = True  # True = treinar o modelo, False = carregar o modelo salvo
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
if TRAIN_MODEL:
    model, embending_model = create_base_network((104, 300, 6))
    model.summary()
    history = model.fit(X_train_concat, X_train[2],
                        validation_data=(Y_test_concat, Y_test[2]),
                        batch_size=16, epochs=100, callbacks=[early_stopping])
    model.save("modelo_treinado.h5")
else:
    from tensorflow.keras.models import load_model
    model = load_model("modelo_treinado.h5")

# ----------------------------------------------- Avaliação do modelo ----------------------------------------------------- #
# Avaliação do modelo
y_pred = model.predict(Y_test_concat)
y_pred = (y_pred > 0.22).astype(int)
precision = precision_score(Y_test[2], y_pred)
recall = recall_score(Y_test[2], y_pred)
f1 = f1_score(Y_test[2], y_pred)

print(f'Precisão no conjunto teste: {precision:.2f}')
print(f'Recall conjunto teste: {recall:.2f}')
print(f'F1 Score no conjunto teste: {f1:.2f}')

# Definindo uma lista de thresholds para testar
thresholds = np.arange(0.0, 1.1, 0.1)
evaluate_thresholds(test_data[2],y_pred, thresholds)


# ----------------------------------------------- Grad-CAM Visualização ----------------------------------------------------- #
# For Grad-CAM, we define a simple loss function. For a binary classification, we can just use the output value.
def loss_function(output):
    return output[:, 0]


# Choose the penultimate convolutional layer to use for Grad-CAM.
# In VGG16, 'block5_conv3' is a common choice. Ensure that this layer exists in your model.
last_conv_layer_name = 'block5_conv3'

# Instantiate the Gradcam object
gradcam = Gradcam(model, model_modifier=lambda m: m, clone=True)

# Loop through the first 5 test images (from Y_test_concat)
for i in range(5):
    # Select one image and ensure it is in batch form (shape: (1, 104, 300, 6))
    img_input = Y_test_concat[i:i + 1]

    # Compute the Grad-CAM heatmap; you can specify the penultimate layer either by name or index.
    # Here we pass the name.
    heatmap = gradcam(loss_function, img_input, penultimate_layer=last_conv_layer_name)

    # Normalize the heatmap for display
    heatmap = normalize(heatmap)[0]

    # Display the heatmap using matplotlib
    plt.figure(figsize=(6, 4))
    plt.imshow(heatmap, cmap='jet')
    plt.title(f"Grad-CAM Heatmap for Test Image {i}")
    plt.axis('off')
    plt.savefig(f"gradcam_heatmap_{i}.png", bbox_inches='tight')
    plt.show()

    # Optional: Overlay the Grad-CAM heatmap on the original image.
    # Since the concatenated image has 6 channels, we use the first 3 channels (the first image) for overlay.
    original_img = img_input[0][..., :3]
    # Resize the heatmap to match the original image size
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    # Blend the heatmap with the original image
    overlay = cv2.addWeighted(np.uint8(original_img), 0.6, heatmap_color, 0.4, 0)

    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Grad-CAM Overlay for Test Image {i}")
    plt.axis('off')
    plt.savefig(f"gradcam_overlay_{i}.png", bbox_inches='tight')
    plt.show()



# ----------------------------------------------- UMAP para visualização ----------------------------------------------------- #
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