import math
from sys import argv
import numpy as np
import cv2 as cv


training_file = 'images/cornerTraining/allTrue.jpg'
filename = argv[0] if len(argv) > 0 else training_file

# Loads an image
src = cv.imread(cv.samples.findFile(training_file), cv.IMREAD_GRAYSCALE)
# Check if image is loaded fine
if src is None:
    print('Error opening image!')
    print('Usage: hough_lines.py [image_name -- default ' + training_file + '] \n')

dst = cv.Canny(src, 50, 200, None, 3)

# Copy edges to the images that will display the results in BGR
cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)


lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 200, 61)

print(linesP)
"""
threshold: This is the minimum number of intersections to detect a line. It's essentially a voting threshold. It should be set based on the complexity of the image and the density of lines you're expecting.
minLineLength: The minimum length of a line in pixels. Lines shorter than this will be rejected.
maxLineGap: The maximum allowed gap between line segments to treat them as a single line.
"""
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
        cv.putText(cdstP, f'({l[0]}, {l[1]})', (l[0], l[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
        cv.putText(cdstP, f'({l[2]}, {l[3]})', (l[2], l[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)


#show image
cv.imshow("Source", src)
#cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
cv.waitKey()