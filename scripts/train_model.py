import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

X_train = np.load("X_train.npy", allow_pickle=True)
y_train = np.load("y_train.npy", allow_pickle=True)
X_val = np.load("X_val.npy", allow_pickle=True)
y_val = np.load("y_val.npy", allow_pickle=True)
label_classes = np.load("label_classes.npy", allow_pickle=True)

print(f"Loaded {len(X_train)} training images and {len(X_val)} validation images.")
print(f"Number of classes: {len(label_classes)}")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

def generator(features, labels, batch_size):
    while True:
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            batch_paths = features[start:end]
            batch_labels = np.array(labels[start:end])
            
            batch_images = []
            for img_path in batch_paths:
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                batch_images.append(img_array)
            
            batch_images = np.array(batch_images)
            yield datagen.flow(batch_images, batch_labels, batch_size=batch_size, shuffle=False).next()

base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(len(label_classes), activation='sigmoid')(x) 

model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    "models/xray_densenet_best.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

lr_reduce = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-7
)

train_steps = len(X_train) // BATCH_SIZE
val_steps = len(X_val) // BATCH_SIZE

history = model.fit(
    generator(X_train, y_train, BATCH_SIZE),
    steps_per_epoch=train_steps,
    validation_data=generator(X_val, y_val, BATCH_SIZE),
    validation_steps=val_steps,
    epochs=EPOCHS,
    callbacks=[checkpoint, lr_reduce]
)

model.save("models/xray_densenet_final.h5")

print("✅ Training complete! Model saved.")
