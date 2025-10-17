import numpy as np
import pickle
import matplotlib.pyplot as plt
from train_fetal_2d_cyclegan import build_2d_generator

# Load one test image
with open('processed_data_4slice_fixed/test_4slice_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

test_img = test_data['images'][0:1] / 255.0
test_ga = test_data['gestational_age'][0:1].reshape(-1, 1)

print(f"Input image stats:")
print(f"  Shape: {test_img.shape}")
print(f"  Min: {test_img.min():.4f}, Max: {test_img.max():.4f}, Mean: {test_img.mean():.4f}")

# Load model
gen = build_2d_generator((138, 176, 1), 16)
gen.load_weights('weights/cyclegan_2d/gen_B2A_epoch_50.weights.h5')

# Generate
output = gen([test_img, test_ga], training=False).numpy()

print(f"\nOutput image stats:")
print(f"  Min: {output.min():.4f}, Max: {output.max():.4f}, Mean: {output.mean():.4f}")
print(f"  Unique values: {len(np.unique(output))}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(test_img[0, :, :, 0], cmap='gray')
axes[0].set_title('Input')
axes[1].imshow(output[0, :, :, 0], cmap='gray')
axes[1].set_title('Output')
axes[2].hist(output.flatten(), bins=50)
axes[2].set_title('Output Distribution')
plt.savefig('debug_output.png')
print("\nSaved: debug_output.png")