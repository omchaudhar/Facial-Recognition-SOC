import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Lambda, Flatten
from tensorflow.keras.optimizers import Adam
import os
import cv2
import numpy as np

DATA_DIR = ".cache/kagglehub/datasets/vasukipatel/face-recognition-dataset/versions/1/Faces/Faces"
IMG_SIZE = (100, 100)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return img

def load_data():
    X = []
    y = []
    label_map = {}
    current_label = 0

    for person_name in os.listdir(DATA_DIR):
        person_dir = os.path.join(DATA_DIR, person_name)
        if os.path.isdir(person_dir):
            if person_name not in label_map:
                label_map[person_name] = current_label
                current_label += 1
            
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                processed_img = preprocess_image(img_path)
                if processed_img is not None:
                    X.append(processed_img)
                    y.append(label_map[person_name])
    
    return np.array(X), np.array(y), label_map

# Siamese Neural Network Architecture
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)  # Output: ~45x45

    # Second block
    c2 = Conv2D(128, (7, 7), activation='relu')(p1)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)  # Output: ~19x19

    # Third block
    c3 = Conv2D(128, (4, 4), activation='relu')(p2)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)  # Output: ~8x8

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation='relu')(p3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')

embedding = make_embedding()

# L1 distance function
def L1_dist(embedding):
    return Lambda(lambda x: tf.math.abs(x[0] - x[1]))

# Siamese model
def make_siamese_model():
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))
    
    # Positive image input in the network
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    # Combine siamese distance embedding
    siamese_network = L1_dist(embedding)([embedding(input_image), embedding(validation_image)])
    
    # Output
    output = Dense(1, activation='sigmoid')(siamese_network)
    
    return Model(inputs=[input_image, validation_image], outputs=output, name='SiameseNetwork')

siamese_model = make_siamese_model()

siamese_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# This part needs to be adapted for triplet loss or contrastive loss based on the actual video implementation
# For now, a placeholder for training loop

# Load data (this will be used to create triplets for training)
X, y, label_map = load_data()

# Placeholder for creating triplets - this is a simplified version and needs to be expanded
# to create actual anchor, positive, negative triplets as shown in the videos.
# For demonstration, we'll create some dummy pairs.

# Create dummy pairs for training (replace with actual triplet generation)
anchors = []
positives = []
negatives = []

# Simple example: for each image, create a positive pair with itself and a negative pair with a random image
for i in range(len(X)):
    anchor_img = X[i]
    anchor_label = y[i]

    # Positive: another image of the same person
    positive_indices = np.where(y == anchor_label)[0]
    positive_indices = positive_indices[positive_indices != i] # Exclude itself
    if len(positive_indices) > 0:
        positive_img = X[np.random.choice(positive_indices)]
    else:
        # If no other image of the same person, use the anchor itself (not ideal for training)
        positive_img = anchor_img

    # Negative: an image of a different person
    negative_indices = np.where(y != anchor_label)[0]
    if len(negative_indices) > 0:
        negative_img = X[np.random.choice(negative_indices)]
    else:
        # Fallback if only one person in dataset (shouldn't happen with this dataset)
        negative_img = anchor_img

    anchors.append(anchor_img)
    positives.append(positive_img)
    negatives.append(negative_img)

anchors = np.array(anchors)
positives = np.array(positives)
negatives = np.array(negatives)

# Combine into training data
X_train = [anchors, positives]
y_train = np.ones((len(anchors), 1)) # All are positive pairs for this simplified example

# Train the model (this will be a very basic training, not using triplets yet)
# You would typically use a custom training loop with triplet loss or contrastive loss
# siamese_model.fit(X_train, y_train, epochs=10, batch_size=16)

print("Model building complete. Training setup is a placeholder and needs actual triplet generation and loss function.")

# Save the model (example)
siamese_model.save("siamesemodel.h5")
print("Model saved as siamesemodel.h5")


