import cv2
import numpy as np

# Load the degraded image
img = cv2.imread('degraded_image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 100, 200)

# Display the original and edges images side by side
res = np.hstack((gray, edges))
cv2.imshow('Original vs. Edges', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
