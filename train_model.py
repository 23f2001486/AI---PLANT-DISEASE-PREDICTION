import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models
import os
import json

# Paths
DATA_DIR = r"C:\Users\divya\OneDrive\Documents\betel_leaf\Comprehensive Betel Leaf Disease Dataset for Advanced Pathology Research\Betel Leaf Dataset\Betel Leaf Dataset\Betel Leaf Dataset\Augmented_Dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = "model/densenet_model.h5"
CLASS_NAMES_PATH = "model/class_names.json"

# Load dataset (train/val split)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


class_names = train_ds.class_names
print(" Classes found:", class_names)

# Prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Model
base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=IMG_SIZE+(3,))
base_model.trainable = False  # transfer learning (freeze base)

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(256, activation="relu"),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Save model + class names
os.makedirs("model", exist_ok=True)
model.save(MODEL_PATH)

with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)

print(f" Model saved at {MODEL_PATH}")
print(f" Class names saved at {CLASS_NAMES_PATH}")
