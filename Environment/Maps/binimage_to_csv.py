import cv2
import numpy as np

# Load the image
name = 'acoruna_port'
image = cv2.imread(f'Environment/Maps/{name}.png', 0)

# Binarize the image
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Convert the binary image to 0s and 1s
binary_image = np.where(binary_image > 0, 1, 0)

# Save the binary image as a CSV file
np.savetxt(f'Environment/Maps/{name}.csv', binary_image, delimiter=',', fmt='%d')
