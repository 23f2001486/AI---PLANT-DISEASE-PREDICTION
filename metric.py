import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===============================
# 1️⃣ Paths and Parameters
# ===============================
data_dir = r"C:\Users\divya\Downloads\Betel Leaf Dataset\Betel Leaf Dataset\Original_Dataset"
model_path = r"C:\Users\divya\OneDrive\Documents\Node\AI\model\densenet_model_improved.h5"
img_size = (224, 224)
batch_size = 32

# ===============================
# 2️⃣ Data Generator (Validation Only)
# ===============================
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
# 3️⃣ Load Model
# ===============================
model = load_model(model_path)
print("\n Model loaded successfully!")

# ===============================
# 4️⃣ Evaluate Performance
# ===============================
loss, accuracy = model.evaluate(val_generator)
print(f"\n Overall Validation Accuracy: {accuracy * 100:.2f}%")

Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)
class_labels = list(val_generator.class_indices.keys())

print("\nClassification Report:\n")
print(classification_report(val_generator.classes, y_pred, target_names=class_labels))

cm = confusion_matrix(val_generator.classes, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - DenseNet Model")
plt.show()
