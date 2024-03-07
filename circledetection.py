import math
from sys import argv
import numpy as np
import cv2 as cv

training_file = 'images/testing/img.png'

filename = argv[1] if len(argv) > 1 else training_file

# Loads an image
src = cv.imread(cv.samples.findFile(training_file), cv.IMREAD_COLOR)

# Read image.
if src is None:
    print('Error opening image!')
    print('Usage: hough_lines.py [image_name -- default ' + training_file + '] \n')
    exit()

# Convert to grayscale.
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

# Blur using 3 * 3 kernel.
gray_blurred = cv.blur(gray, (3, 3))

# Apply Hough transform on the blurred image.
detected_circles = cv.HoughCircles(gray_blurred,
                                    cv.HOUGH_GRADIENT, 1, 150, param1=50,
                                    param2=30, minRadius=20, maxRadius=50)

# Draw circles that are detected.
if detected_circles is not None:
    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))

    print(detected_circles)

    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]

        # Draw the circumference of the circle.
        cv.circle(src, (a, b), r, (0, 255, 0), 2)

        # Draw a small circle (of radius 1) to show the center.
        cv.circle(src, (a, b), 1, (0, 0, 255), 3)

# Display the detected circles.
cv.imshow("Detected Circles", src)
cv.waitKey(0)
