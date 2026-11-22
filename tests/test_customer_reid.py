import numpy as np
from src.customer_reid import CustomerReID

# Initialize OSNet
reid = CustomerReID()

# Create a sample person image
person_image = np.random.randint(
    0, 255,
    (256, 128, 3),
    dtype=np.uint8
)

# Extract features
features = reid.extract_features(person_image)

print("Feature Shape:", features.shape)

# Compare same image features
similarity = reid.compare_features(
    features,
    features
)

print("Similarity:", similarity)

print("OSNet Test Successful!")
