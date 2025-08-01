import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Dataset directory
dataset_path = "ASL_Dataset"
asl_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

X = []
y = []

# Load data
for idx, letter in enumerate(asl_labels):
    letter_folder = os.path.join(dataset_path, letter)
    if not os.path.isdir(letter_folder):
        print(f"Folder missing for letter: {letter}")
        continue

    for i in range(200):  # Assuming 0.npy to 199.npy
        file_path = os.path.join(letter_folder, f"{i}.npy")
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            continue

        try:
            data = np.load(file_path)

            # Reshape or flatten if needed
            if len(data.shape) == 2:  # e.g. (21, 2)
                data = data.flatten()
                X.append(data)
                y.append(idx)

            elif len(data.shape) == 3:  # e.g. (batch_size, 21, 2)
                data = data.reshape(data.shape[0], -1)
                X.extend(data)
                y.extend([idx] * data.shape[0])

            elif len(data.shape) == 1:  # Already flat
                X.append(data)
                y.append(idx)

            else:
                print(f"Unexpected shape {data.shape} in {file_path}")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define model
model = keras.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(26, activation='softmax')  # 26 letters A-Z
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save
model.save("asl_model.h5")
print("✅ Model training complete. Saved as asl_model.h5")
