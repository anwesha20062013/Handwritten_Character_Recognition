import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)

print("Loading EMNIST Balanced Dataset...")

# ----------------------------
# Read CSV Files
# ----------------------------

train_df = pd.read_csv("emnist-balanced-train.csv", header=None)
test_df = pd.read_csv("emnist-balanced-test.csv", header=None)

print("Training Shape :", train_df.shape)
print("Testing Shape  :", test_df.shape)

# ----------------------------
# Split Labels & Images
# ----------------------------

y_train = train_df.iloc[:, 0].values
x_train = train_df.iloc[:, 1:].values

y_test = test_df.iloc[:, 0].values
x_test = test_df.iloc[:, 1:].values

# Normalize

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape

x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)

# ---------------------------------
# Fix EMNIST Orientation
# ---------------------------------

x_train = np.array([
    np.fliplr(np.rot90(img))
    for img in x_train
])

x_test = np.array([
    np.fliplr(np.rot90(img))
    for img in x_test
])

# Add channel dimension

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("Training Images :", x_train.shape)

# ----------------------------
# CNN Model
# ----------------------------

model = Sequential([

    tf.keras.Input(shape=(28,28,1)),

    Conv2D(32, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        47,
        activation="softmax"
    )

])

model.summary()

# ----------------------------
# Compile
# ----------------------------

# ----------------------------
# Data Augmentation
# ----------------------------

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomRotation(0.05),

    tf.keras.layers.RandomZoom(0.10),

    tf.keras.layers.RandomTranslation(
        0.10,
        0.10
    )

])

# ----------------------------
# Compile
# ----------------------------

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

# ----------------------------
# Train
# ----------------------------

history = model.fit(

    data_augmentation(x_train),

    y_train,

    epochs=25,

    validation_data=(x_test, y_test)

)

# ----------------------------
# Evaluate
# ----------------------------

loss,accuracy = model.evaluate(

    x_test,

    y_test

)

print()

print("Character Accuracy :",accuracy*100)

# ----------------------------
# Save Model
# ----------------------------

model.save(

    "models/character_model.h5"

)

print()

print("Character Model Saved Successfully!")