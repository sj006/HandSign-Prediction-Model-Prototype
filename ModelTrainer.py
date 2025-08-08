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
        print(f"❌ Folder missing for letter: {letter}")
        continue

    # List all .npy files in the letter folder
    files = sorted([f for f in os.listdir(letter_folder) if f.endswith(".npy")])
    if len(files) == 0:
        print(f"⚠️ No data files for letter: {letter}")
        continue

    for file_name in files:
        file_path = os.path.join(letter_folder, file_name)
        try:
            data = np.load(file_path)

            if len(data.shape) == 2:      # (21, 2) format
                data = data.flatten()
                X.append(data)
                y.append(idx)

            elif len(data.shape) == 3:    # (batch_size, 21, 2)
                data = data.reshape(data.shape[0], -1)
                X.extend(data)
                y.extend([idx] * data.shape[0])

            elif len(data.shape) == 1:    # Already flat
                X.append(data)
                y.append(idx)

            else:
                print(f"⚠️ Unexpected shape {data.shape} in {file_path}")

        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"✅ Total samples loaded: {len(X)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

# Define model
model = keras.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(26, activation='softmax')  # 26 classes A-Z
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Evaluate and save
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Final Test Accuracy: {accuracy * 100:.2f}%")
model.save("asl_model.h5")
print("✅ Model training complete. Saved as asl_model.h5")
