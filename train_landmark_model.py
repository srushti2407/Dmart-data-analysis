import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

# ==============================
# LOAD DATA
# ==============================
data = pd.read_csv("landmarks.csv")

# Fix labels: "A-samples" -> "A"
data['label'] = data['label'].astype(str).str.split('-').str[0]

# ==============================
# NORMALIZE LANDMARKS (IMPORTANT)
# ==============================
def normalize_landmarks(row):
    row = row.values.astype(float)
    coords = row.reshape(21, 3)

    base = coords[0]  # wrist
    coords = coords - base

    return coords.flatten()

X = data.drop('label', axis=1).apply(normalize_landmarks, axis=1)
X = np.stack(X.values)

y = data['label'].values

print("X shape:", X.shape)

# ==============================
# ENCODE LABELS
# ==============================
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = keras.utils.to_categorical(y_encoded)

# ==============================
# SCALE
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)
np.save("labels.npy", le.classes_)

# ==============================
# SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_cat, test_size=0.2, random_state=42
)

# ==============================
# MODEL
# ==============================
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(63,)),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(y_cat.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==============================
# TRAIN
# ==============================
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ==============================
# SAVE
# ==============================
model.save("model.h5")

print("\n✅ Training complete")