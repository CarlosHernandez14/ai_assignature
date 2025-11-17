# train_optimized.py
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# ----- CONFIG -----
DATASET_DIR = r"C:/Users/carlo/Documents/ai_projects/ai_assignature/projects/cnn_project/datasets"
MODEL_DIR = r"C:/Users/carlo/Documents/ai_projects/ai_assignature/projects/cnn_project/models"
IMG_SIZE = (100, 100)
BATCH_SIZE = 64   # try 64, then 128 if memory allows
SEED = 42
EPOCHS = 38
INIT_LR = 1e-3

# Optionally enable mixed precision (test if it helps; comment out if problems)
USE_MIXED_PRECISION = False
if USE_MIXED_PRECISION:
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled: mixed_float16")
    except Exception as e:
        print("Mixed precision not enabled:", e)

# Set CPU threading to reasonable values for Ryzen 3700X (8 cores / 16 threads)
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Print devices
print("TF version:", tf.__version__)
print("Physical devices:", tf.config.list_physical_devices())

# ----- DATASETS: use image_dataset_from_directory to stream data -----
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=SEED
)

# If you have a separate test folder, load similarly; otherwise split further from val or train.
# For demonstration let's create a small test set from val (or assume separate test folder)
# test_ds = ...

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# ----- PERFORMANCE: caching, prefetching, parallel mapping -----
AUTOTUNE = tf.data.AUTOTUNE

# Example: if dataset fits in memory, uncomment caching to memory (fast). Otherwise use cache(filename).
# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ----- Data augmentation as model layers (runs on GPU) -----
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
        # Add more augmentation if useful
    ],
    name="data_augmentation",
)

# ----- Build model: lightweight, batchnorm, GAP, dropout ----- 
def make_model(input_shape=IMG_SIZE + (3,), num_classes=num_classes, lr=INIT_LR):
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)  # augment on GPU
    x = layers.Rescaling(1./255)(x)  # normalize

    # Block 1
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.SeparableConv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3 (optional — uncomment if you need more capacity)
    # x = layers.SeparableConv2D(128, 3, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.LeakyReLU()(x)
    # x = layers.MaxPooling2D(2)(x)
    # x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.4)(x)

    # Last layer: if using mixed_float16, outputs need to be float32 for numeric stability
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs, name="optimized_cnn")
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

model = make_model()
model.summary()

# ----- Callbacks -----
callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.h5"), save_best_only=True, monitor="val_loss"),
]

# ----- Training -----
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save final model
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, "final_model.h5"))

# If you want predictions/evaluation on a test set:
# test_loss, test_acc = model.evaluate(test_ds)
# print("Test loss:", test_loss, "Test acc:", test_acc)
