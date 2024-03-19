import numpy as np
import cv2
import matplotlib.pyplot as plt

def linearRegression(x_values, y_values):
    # Perform linear regression
    slope, intercept = np.polyfit(x_values, y_values, 1)
    return slope, intercept

def findContours(image):
    contours, _ = cv2.findContours(image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    min_contour_length = 500
    filtered_contours = []

    for contour in contours:
        contour_length = cv2.arcLength(contour, closed=True)
        if contour_length > min_contour_length:
            filtered_contours.append(contour)
    print(len(filtered_contours))

    # Draw contours on the original image
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rstimage = cv2.drawContours(image=contour_image, contours=filtered_contours, contourIdx=-1, color=(255, 0, 0), thickness=2,
                     lineType=cv2.LINE_AA)

    if filtered_contours:
        first_contour_coordinates = filtered_contours[0].reshape(-1, 2)  # Reshape to Nx2 array
        print("Coordinates of the first contour:")
        tc = 0
        for x, y in first_contour_coordinates:
            print(f"({x}, {y})")
            tc += 1

        # Extract x and y values from contour coordinates
        x_values = [point[0] for point in first_contour_coordinates]
        y_values = [point[1] for point in first_contour_coordinates]

        # Perform linear regression
        slope, intercept = linearRegression(x_values, y_values)

        # Generate y values for the regression line
        line_x_values = np.array([min(x_values), max(x_values)])
        line_y_values = slope * line_x_values + intercept

        # Plot the linear regression line
        plt.plot(line_x_values, line_y_values, color='red', label='Regression Line')
        plt.legend()

    else:
        print("No contours found.")

    print(tc)
    return rstimage

def cannyTransform(image):
    # Setting parameter values
    t_lower = 100  # Lower Threshold
    t_upper = 300  # Upper threshold

    # Applying the Canny Edge filter
    edges = cv2.Canny(image, t_lower, t_upper)
    return edges

# Read the image
img = cv2.imread("images/allpieces/all.jpg")

# Perform Canny edge detection
cannyRst = cannyTransform(img)

# Find and draw contours
contour_image = findContours(cannyRst)

# Save the resulting image
# cv2.imwrite("contours_detected.jpg", contour_image)

# Display the resulting image
cv2.imshow("Canny Converted", cannyRst)
cv2.imshow("Contours Detected", contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
