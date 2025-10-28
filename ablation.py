import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import gc

# ====================================================




# 1️⃣ Common Settings
# ====================================================
data_dir = r"C:\Users\divya\Downloads\Betel Leaf Dataset\Betel Leaf Dataset\Original_Dataset"
img_size = (224, 224)
batch_size = 8  # reduced from 32
epochs = 5
val_split = 0.2

# ✅ Prevent TensorFlow from allocating all GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ====================================================
# 2️⃣ Validation Generator (fixed)
# ====================================================
base_datagen = ImageDataGenerator(rescale=1./255, validation_split=val_split)
val_generator = base_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# ====================================================
# 3️⃣ Helper Function (Memory-Efficient)
# ====================================================
def build_and_train(model_type, augment=True, transfer=True, dropout=True, img_resize=(224,224), epochs=5):
    print(f"\n🔹 Training: {model_type}")

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25 if augment else 0,
        width_shift_range=0.2 if augment else 0,
        height_shift_range=0.2 if augment else 0,
        zoom_range=0.2 if augment else 0,
        shear_range=0.2 if augment else 0,
        horizontal_flip=augment,
        vertical_flip=augment,
        validation_split=val_split
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_resize,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_resize,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    base_model = DenseNet121(weights='imagenet' if transfer else None,
                             include_top=False, input_shape=img_resize + (3,))
    
    # Freeze most layers to reduce computation
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    if dropout:
        x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(train_generator.num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    start = time.time()
    history = model.fit(train_generator, validation_data=val_generator, epochs=epochs, verbose=0)
    end = time.time()

    val_loss, val_acc = model.evaluate(val_generator, verbose=0)
    train_time = round((end - start)/60, 2)

    print(f" {model_type}: {val_acc*100:.2f}% | Time: {train_time} min")

    # ✅ Clear memory
    tf.keras.backend.clear_session()
    del model, base_model, train_generator, val_generator
    gc.collect()

    return val_acc * 100, train_time


# ====================================================
# 4️⃣ Run Ablation Variants
# ====================================================
results = []

configs = [
    ("Full Model (Our Approach)", True, True, True, (224,224), 5),
    ("Without Data Augmentation", False, True, True, (224,224), 5),
    ("Without Transfer Learning", True, False, True, (224,224), 5),
    ("Without Dropout", True, True, False, (224,224), 5),
    ("Smaller Input Size (128x128)", True, True, True, (128,128), 5),
    ("Reduced Epochs (3)", True, True, True, (224,224), 3),
]

for cfg in configs:
    results.append([cfg[0], *build_and_train(*cfg)])


# ====================================================
# 5️⃣ Display Table
# ====================================================
df = pd.DataFrame(results, columns=["Configuration", "Accuracy", "Training Time (min)"])
df["Performance Drop"] = df["Accuracy"].max() - df["Accuracy"]

print("\n==================== ABLATION STUDY RESULTS ====================")
print(df.to_string(index=False))

# ====================================================
# 6️⃣ Visualize
# ====================================================
plt.figure(figsize=(10,5))
plt.bar(df["Configuration"], df["Accuracy"], color=['green'] + ['red']*(len(df)-1))
plt.xticks(rotation=45, ha='right')
plt.ylabel("Accuracy (%)")
plt.title("Ablation Study: Accuracy Impact")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.bar(df["Configuration"], -df["Performance Drop"], color='orange')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Performance Drop (%)")
plt.title("Impact of Component Removal")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.bar(df["Configuration"], df["Training Time (min)"], color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Training Time (min)")
plt.title("Training Time Comparison")
plt.tight_layout()
plt.show()
