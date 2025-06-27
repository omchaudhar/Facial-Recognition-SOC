
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# 1. Prepare Data (Example: XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 2. Build the Model
model = keras.Sequential([
    layers.Input(shape=(2,)), # Input layer with 2 features
    layers.Dense(4, activation=\'relu\'), # Hidden layer with 4 neurons and ReLU activation
    layers.Dense(1, activation=\'sigmoid\') # Output layer with 1 neuron and Sigmoid activation
])

# 3. Compile the Model
model.compile(optimizer=\'adam\', # Optimization algorithm
              loss=\'binary_crossentropy\', # Loss function for binary classification
              metrics=[\'accuracy\']) # Metric to monitor during training

# 4. Train the Model
# model.fit(X, y, epochs=1000, verbose=0) # Train for 1000 epochs

# 5. Make Predictions
# predictions = model.predict(X)
# print("\nPredictions:")
# print(predictions.round())

# Model Summary
model.summary()


