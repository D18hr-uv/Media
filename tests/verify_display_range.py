import numpy as np

# Simulate image in [0, 255] float
image_255 = np.array([[[0.0, 127.5, 255.0]]]).astype('float32')

# Test proposed fix: convert to [0, 1] for display
display_img = image_255 / 255.0
display_img = np.clip(display_img, 0.0, 1.0)

# Check values
assert display_img[0][0][0] == 0.0, f"Expected 0.0, got {display_img[0][0][0]}"
assert display_img[0][0][1] == 0.5, f"Expected 0.5, got {display_img[0][0][1]}"
assert display_img[0][0][2] == 1.0, f"Expected 1.0, got {display_img[0][0][2]}"

print("Display range verification passed.")
