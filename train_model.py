import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from emnist import extract_training_samples, extract_test_samples
import matplotlib.pyplot as plt

print("Downloading and loading EMNIST 'letters' dataset...")
train_images, train_labels = extract_training_samples('letters')
test_images, test_labels = extract_test_samples('letters')

print("Reshaping and normalizing...")
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

train_labels = train_labels - 1
test_labels = test_labels - 1

print("Building model")

model = models.Sequential()

# Block 1: Find edges and features
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Block 2: Find complex shapes (curves, loops)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Block 3: Decision making
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
# Output layer: 26 neurons for A-Z
model.add(layers.Dense(26, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Start training...")
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

print("Training complete! Saving model to disk...")
model.save('air_writing_model.keras')
print("Model successfully saved as 'air_writing_model.keras'")