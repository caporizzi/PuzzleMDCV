import cv2 as cv
import numpy as np

# Load an image
image = cv.imread("images/allpiecesflash/contorno.png")

# Check if the image was loaded successfully
if image is None:
    print("Error: Unable to load image.")
    exit()

# Convert to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Find contours
contours, hierarchy = cv.findContours(gray, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

# Iterate through the contours and hierarchy to find the first child of the external contour
external_contours = []
first_child_contour = None
for i, contour in enumerate(contours):
    if hierarchy[0][i][3] == -1:  # No parent contour, thus it's an external contour
        external_contours.append(contour)
        first_child_index = hierarchy[0][i][2]  # Index of the first child contour
        if first_child_index != -1:  # Ensure there is a child contour
            first_child_contour = contours[first_child_index]
        break  # Only consider the first external contour


# Draw the first child contour
if first_child_contour is not None:
    cv.drawContours(image, [first_child_contour], -1, (0, 0, 255), 1)

# Display the image
cv.imshow("External Contour and its First Child", image)
cv.waitKey(0)
cv.destroyAllWindows()
