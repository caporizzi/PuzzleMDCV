"""

Function which looks for straight line in a picture and return 4 points which are the corner
"""

import cv2
import numpy as np


def find_straight_line_corners(image, error_range=100):
    # Preprocess the image (e.g., convert to grayscale, apply Gaussian blur)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

    # Filter lines based on their orientation to identify potential straight lines
    straight_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > error_range or abs(y2 - y1) > error_range:
            continue
        straight_lines.append(line[0])
    print(straight_lines)
    for x1, y1, x2, y2 in straight_lines:
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


    """# Find intersections of straight lines to determine corners
    corners = cv2.goodFeaturesToTrack(gray, 16, 0.01, 50)
    corners = np.intp(corners)

    # Draw corners on the image
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)"""

    return image

image = cv2.imread('images/testing/1200px-Checkerboard_pattern.svg.png')
result_image = find_straight_line_corners(image, error_range=10)
cv2.imshow('Detected Corners', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()