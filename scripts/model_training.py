# ============================================
# X-RAY ANOMALY DETECTION – RETRAINING + GRAD-CAM
# ============================================

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import matplotlib.pyplot as plt

# --------------------------------------------
# 1. PATH SETUP
# --------------------------------------------
DATA_DIR = r"C:\Users\PC\xray_project\dataset"
CSV_PATH = os.path.join(DATA_DIR, "Data_Entry_2017.csv")

IMG_SIZE = 224
BATCH_SIZE = 32

# --------------------------------------------
# 2. LOAD CSV + MAP IMAGE PATHS
# --------------------------------------------
df = pd.read_csv(CSV_PATH)

def find_image_path(img_name):
    for folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder)
        potential_path = os.path.join(folder_path, img_name)
        if os.path.exists(potential_path):
            return potential_path
    return None

df['image_path'] = df['Image Index'].apply(find_image_path)
df = df.dropna(subset=['image_path'])

# Split multi-labels into lists
df['Finding Labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))

# Encode labels
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(df['Finding Labels'])
print("✅ Classes detected:", mlb.classes_)

# --------------------------------------------
# 3. TRAIN/TEST SPLIT
# --------------------------------------------
train_df, test_df, y_train, y_test = train_test_split(
    df['image_path'], labels, test_size=0.2, random_state=42
)

# --------------------------------------------
# 4. DATA GENERATOR FUNCTION
# --------------------------------------------
def generator(features, labels, batch_size):
    while True:
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            batch_paths = features[start:end]
            batch_labels = np.array(labels[start:end])
            
            batch_images = []
            for img_path in batch_paths:
                img = keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = keras.preprocessing.image.img_to_array(img)
                img_array = img_array / 255.0
                batch_images.append(img_array)
            
            yield np.array(batch_images), batch_labels

train_gen = generator(list(train_df), y_train, BATCH_SIZE)
test_gen = generator(list(test_df), y_test, BATCH_SIZE)

steps_per_epoch = len(train_df) // BATCH_SIZE
val_steps = len(test_df) // BATCH_SIZE

# --------------------------------------------
# 5. MODEL DEFINITION
# --------------------------------------------
num_classes = len(mlb.classes_)

base_model = keras.applications.DenseNet121(
    weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = keras.layers.GlobalAveragePooling2D()(base_model.output)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.3)(x)
output = keras.layers.Dense(num_classes, activation='sigmoid')(x)

model = keras.Model(inputs=base_model.input, outputs=output)

# Freeze base layers first
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --------------------------------------------
# 6. TRAIN MODEL (INITIAL TRAINING)
# --------------------------------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)

print("\n🚀 Starting initial training (frozen base)...")
history1 = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_gen,
    validation_steps=val_steps,
    epochs=15,
    callbacks=[early_stop]
)

# --------------------------------------------
# 7. FINE-TUNE (UNFREEZE LAST LAYERS)
# --------------------------------------------
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n🔧 Fine-tuning last 50 layers...")
history2 = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_gen,
    validation_steps=val_steps,
    epochs=8,
    callbacks=[early_stop]
)

# --------------------------------------------
# 8. SAVE MODEL
# --------------------------------------------
os.makedirs("models", exist_ok=True)
model.save("models/xray_densenet_retrained.h5")
print("\n✅ Model retrained and saved successfully!")

# --------------------------------------------
# 9. GRAD-CAM VISUALIZATION
# --------------------------------------------
def get_gradcam_heatmap(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, model, layer_name="conv5_block16_concat"):
    img = keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    heatmap = get_gradcam_heatmap(model, img_array, layer_name)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Grad-CAM Visualization")
    plt.show()

# --------------------------------------------
# 10. TEST GRAD-CAM ON SAMPLE IMAGE
# --------------------------------------------
sample_img = test_df.iloc[0]
print("\n🩻 Displaying Grad-CAM for:", sample_img)
display_gradcam(sample_img, model)
