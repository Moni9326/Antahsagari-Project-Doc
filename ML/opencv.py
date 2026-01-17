
import numpy as np
import cv2
import os

# Paths
input_path = r"/filtered images/WhatsApp Image 2025-04-17 at 11_EnhancedLR.png"
save_dir = r"/filtered images"

# Load image
image = cv2.imread(input_path)

# Convert to LAB color space and split channels
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# Apply CLAHE to the L-channel
clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(8, 8))
cl = clahe.apply(l)

# Merge and convert back to BGR
limg = cv2.merge((cl, a, b))
enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# Create side-by-side comparison
comparison = np.hstack((image, enhanced))

# Save the comparison image
comparison_filename = "CLAHE_comparison.png"
comparison_path = os.path.join(save_dir, comparison_filename)
cv2.imwrite(comparison_path, comparison)

print(f"Comparison image saved at: {comparison_path}")
