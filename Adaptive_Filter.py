import cv2
import numpy as np

# Load the degraded image
img = cv2.imread('degraded_image.jpg')

# Apply adaptive filter
J = cv2.bilateralFilter(img, 5, 75, 75)

# Display the original and restored images side by side
res = np.hstack((img, J))
cv2.imshow('Original vs. Restored', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
