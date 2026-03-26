# train_model.py
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = "processed_dataset"

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

num_features = X_train.shape[1]
num_classes = len(np.unique(y_train))

model = keras.Sequential([
    layers.Input(shape=(num_features,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test),
)

model.save("sign_numbers_classifier.h5")
print("Saved model to sign_numbers_classifier.h5")
