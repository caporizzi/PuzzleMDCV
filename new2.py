import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def cannyTransform(image):
    # Setting parameter values
    t_lower = 100  # Lower Threshold
    t_upper = 300  # Upper threshold

    # Applying the Canny Edge filter
    edges = cv2.Canny(image, t_lower, t_upper)
    return edges

def calculate_angle(x1, y1, x2, y2, x3, y3):
    """Calculate angle between three points."""
    a = np.array([x1, y1])
    b = np.array([x2, y2])
    c = np.array([x3, y3])
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    return angle

def findContours(image):
    contours, _ = cv2.findContours(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    min_contour_length = 500

    filtered_contours = []

    for contour in contours:
        contour_length = cv2.arcLength(contour, closed=True)
        if contour_length > min_contour_length:
            filtered_contours.append(contour)
    print("Number of filtered contours :" + str(len(filtered_contours)))

    # Draw contours on the original image
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rstimage = cv2.drawContours(image=contour_image, contours=filtered_contours, contourIdx=-1, color=(255, 0, 0), thickness=1,
                     lineType=cv2.LINE_AA)

    if filtered_contours:
        first_contour_coordinates = filtered_contours[0].reshape(-1, 2)  # Reshape to Nx2 array
        x = [point[0] for point in first_contour_coordinates]
        y = [point[1] for point in first_contour_coordinates]

        segment_length = 5
        num_segments = len(x) // segment_length
        remainder = len(x) % segment_length
        if remainder > 0:
            num_segments += 1

        print("Number of color changes on the plot:", num_segments)

        # Check angles between consecutive sets of 5 points
        angles = []
        for i in range(num_segments - 2):
            angle = calculate_angle(x[i], y[i], x[i+1], y[i+1], x[i+2], y[i+2])
            angles.append(angle)

        # Check if any angle falls within the desired range (70° to 120°)
        for idx, angle in enumerate(angles):
            if 70 <= angle <= 120:
                print(f"Angle {idx+1}: {angle}°")

        # Plot the line
        plt.plot(x, y, marker='o', color='blue')

        # Add labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Line Plot')

        # Show the plot
        plt.grid(True)
        plt.show()

    else:
        print("No contours found.")

    return rstimage

# Read the image
img = cv2.imread("images/testing/rect.PNG")

# Perform Canny edge detection
cannyRst = cannyTransform(img)

# Find and draw contours
contour_image = findContours(cannyRst)

# Display the resulting image
cv2.imshow("Canny Converted", cannyRst)
cv2.imshow("Contours Detected", contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
