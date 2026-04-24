import numpy as np
import matplotlib.pyplot as plt
from emnist import extract_training_samples

print("Loading data...")
train_images, train_labels = extract_training_samples('letters')

# 1. Look at the raw image straight from the library
raw_image = train_images[0]

# 2. Look at what our train_model.py code did to it
fixed_image = np.fliplr(np.rot90(raw_image, axes=(1,0)))

# Show them side-by-side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(raw_image, cmap='gray')
plt.title("1. Raw from Library")

plt.subplot(1, 2, 2)
plt.imshow(fixed_image, cmap='gray')
plt.title("2. What we fed the AI")

plt.show()