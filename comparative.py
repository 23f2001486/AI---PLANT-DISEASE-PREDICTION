# ======================================
# 📊 Comparative Analysis of CNN Models
# ======================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, VGG16, ResNet50
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import os

# ===============================
# 1️⃣ Dataset Setup
# ===============================
data_dir = r"C:\Users\divya\Downloads\Betel Leaf Dataset\Betel Leaf Dataset\Original_Dataset"
img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(val_generator.class_indices)
print(f"Detected {num_classes} classes: {val_generator.class_indices}")

# ===============================
# 2️⃣ Function to Build Model
# ===============================
def build_model(base_model):
    base_model.trainable = False  # freeze convolutional base
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ===============================
# 3️⃣ Define Models
# ===============================
models_dict = {
    "VGG16": VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    "ResNet50": ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    "DenseNet121": DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
}

# ===============================
# 4️⃣ Evaluate Each Model
# ===============================
results = {}

for name, base_model in models_dict.items():
    print(f"\n🔹 Evaluating {name} ...")
    model = build_model(base_model)
    
    # Load your trained DenseNet weights if you already have them
    if name == "DenseNet121":
        try:
            model.load_weights(r"C:\Users\divya\OneDrive\Documents\Node\AI\model\densenet_model_improved.h5")
            print("Loaded pre-trained DenseNet121 weights.")
        except:
            print("No saved DenseNet weights found — using default ImageNet weights.")
    
    # Evaluate
    val_loss, val_acc = model.evaluate(val_generator, verbose=1)
    results[name] = val_acc * 100  # convert to percentage
    print(f"{name} Accuracy: {val_acc * 100:.2f}%")

# ===============================
# 5️⃣ Plot Comparison
# ===============================
plt.figure(figsize=(7, 5))
plt.bar(results.keys(), results.values(), color=['#90caf9', '#64b5f6', '#1e88e5'])
plt.title("Model Comparison - Accuracy (%)")
plt.xlabel("Model")
plt.ylabel("Validation Accuracy (%)")
plt.ylim(0, 100)
plt.show()

# ===============================
# 6️⃣ Print Summary
# ===============================
print("\n Comparative Analysis Summary:")
for model_name, acc in results.items():
    print(f"{model_name}: {acc:.2f}%")

best_model = max(results, key=results.get)
print(f"\n Best Performing Model: {best_model} ({results[best_model]:.2f}%)")

