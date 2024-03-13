import cv2

img = cv2.imread("images/allpieces/all.jpg")  # Read image

# Setting parameter values
t_lower = 100  # Lower Threshold
t_upper = 300  # Upper threshold

# Applying the Canny Edge filter
edge = cv2.Canny(img, t_lower, t_upper)

#cv2.imshow('original', img)
cv2.imshow('edge', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()