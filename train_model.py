import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# -------------------------
# 1. Load dataset
# -------------------------
train_df = pd.read_csv("sign_mnist_train.csv")
test_df = pd.read_csv("sign_mnist_test.csv")

# -------------------------
# 2. Use only A–E (labels 0–4)
# -------------------------
train_df = train_df[train_df['label'] <= 4]
test_df = test_df[test_df['label'] <= 4]

X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values

X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

# -------------------------
# 3. Normalize & reshape
# -------------------------
X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train, 5)
y_test = to_categorical(y_test, 5)

# -------------------------
# 4. Build CNN model
# -------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# -------------------------
# 5. Compile & train
# -------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

# -------------------------
# 6. Save model
# -------------------------
model.save("asl_model_A_E.h5")

print("✅ Model trained & saved as asl_model_A_E.h5")
