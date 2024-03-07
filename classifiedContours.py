import cv2
import numpy as np


def classify_contour(contour, threshold_length):
    # Create a blank image
    contour_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Draw the contour on the blank image
    cv2.drawContours(contour_image, [contour], -1, 255, thickness=cv2.FILLED)

    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(contour_image, rho=1, theta=np.pi / 180, threshold=100, minLineLength=threshold_length,
                            maxLineGap=10)

    if lines is None:
        return "Inner"

    num_lines = len(lines)

    if num_lines > 2:
        return "Corner"
    elif num_lines == 1:
        return "Edge"
    else:
        return "Inner"


# Load image
image = cv2.imread('images/cornerTraining/allTrue.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

min_contour_length = 500
filtered_contours = []
for contour in contours:
    contour_length = cv2.arcLength(contour, closed=True)
    if contour_length > min_contour_length:
        filtered_contours.append(contour)
# Define the minimum line length threshold for considering as a line
threshold_length = 4

# Classify contours
contour_classifications = []
for contour in filtered_contours:
    classification = classify_contour(contour, threshold_length)
    contour_classifications.append(classification)

# Print classifications
for i, classification in enumerate(contour_classifications):
    print(f"Contour {i}: {classification} piece")

# Draw contours on the original image
image_copy = image.copy()
cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)

# Display the results
cv2.imshow('Classified Contours', image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()