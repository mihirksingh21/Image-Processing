import cv2
import numpy as np
# Load the degraded image
img = cv2.imread('degraded_image.jpg', 0)
# Estimate the noise power spectrum
N = img.shape[0]*img.shape[1]
noise = img - cv2.GaussianBlur(img, (5,5), 0)
noise = np.abs(noise)**2
noise = np.sum(noise)
noise /= N
noise /= 2

# Apply Wiener filter
K = 0.1
H = np.ones((5,5))/25
G = np.fft.fft2(img)
H = np.fft.fft2(H, s=img.shape)
W = np.conj(H) / (np.abs(H)**2 + K/noise)
F = W * G
J = np.fft.ifft2(F)
J = np.real(J)

# Convert back to uint8 and display the restored image
J = np.uint8(np.clip(J, 0, 255))
cv2.imshow('Restored Image', J)
cv2.waitKey(0)
cv2.destroyAllWindows()
