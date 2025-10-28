import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

# ===============================
# 1️ Load Model
# ===============================
model_path = r"C:\Users\divya\OneDrive\Documents\Node\AI\model\densenet_model_improved.h5"
model = load_model(model_path)
print(" Model loaded successfully!")

# ===============================
# 2️ Dataset Setup
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

# ===============================
# 3️ Predict and Evaluate
# ===============================
Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# Print overall accuracy
accuracy = np.sum(y_pred == y_true) / len(y_true)
print(f"\n Overall Model Accuracy: {accuracy * 100:.2f}%\n")

# Classification report
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - DenseNet Model")
plt.show()

# ===============================
# 4️ Qualitative Analysis - Correct & Incorrect Predictions
# ===============================
print("\n🔍 Displaying sample correct and incorrect predictions...\n")

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for ax in axes.flat:
    idx = np.random.randint(len(y_true))
    img_path = val_generator.filepaths[idx]
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    pred = class_labels[y_pred[idx]]
    actual = class_labels[y_true[idx]]
    color = "green" if pred == actual else "red"
    ax.imshow(img_array)
    ax.set_title(f"Pred: {pred}\nActual: {actual}", color=color)
    ax.axis('off')
plt.tight_layout()
plt.show()

# ===============================
# 5️ Grad-CAM Visualization
# ===============================
def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Pick one random image
img_path = val_generator.filepaths[np.random.randint(len(val_generator.filepaths))]
img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array / 255.0, axis=0)

# Generate Grad-CAM
heatmap = get_gradcam_heatmap(model, img_array, "conv5_block16_concat")

# Display Grad-CAM overlay
plt.figure(figsize=(6, 6))
plt.imshow(tf.keras.preprocessing.image.array_to_img(img_array[0]))
plt.imshow(cv2.resize(heatmap, (224, 224)), cmap='jet', alpha=0.4)
plt.title("Grad-CAM Visualization")
plt.axis("off")
plt.show()
