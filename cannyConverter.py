import cv2
import numpy as np
import matplotlib.pyplot as plt

def cannyTransform(image):
    # Setting parameter values
    t_lower = 100  # Lower Threshold
    t_upper = 300  # Upper threshold

    # Applying the Canny Edge filter
    edges = cv2.Canny(image, t_lower, t_upper)
    return edges


def findContours(image):
    contours, _ = cv2.findContours(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    min_contour_length = 500

    filtered_contours = []

    for contour in contours:
        contour_length = cv2.arcLength(contour, closed=True)
        if contour_length > min_contour_length:
            filtered_contours.append(contour)
    print(len(filtered_contours))

    # Draw contours on the original image
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rstimage = cv2.drawContours(image=contour_image, contours=filtered_contours, contourIdx=-1, color=(255, 0, 0), thickness=1,
                     lineType=cv2.LINE_AA)

    if filtered_contours:
        first_contour_coordinates = filtered_contours[1].reshape(-1, 2)  # Reshape to Nx2 array
        x = [point[0] for point in first_contour_coordinates]
        y = [point[1] for point in first_contour_coordinates]

        for i in range(0, len(x), 50):
            plt.plot(x[i:i + 50], y[i:i + 50], marker='o', color='C{}'.format(i // 50))  # Using color cycling

        # Add labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Line Plot')

        # Show the plot
        plt.grid(True)
        plt.show()
        print("Coordinates of the first contour:")
        tc = 0
        for x, y in first_contour_coordinates:
            print(f"({x}, {y})")
            tc += 1

    else:
        print("No contours found.")
    print(tc)
    return rstimage


# Read the image
img = cv2.imread("images/allpieces/all.jpg")

# Perform Canny edge detection
cannyRst = cannyTransform(img)

# Find and draw contours
contour_image = findContours(cannyRst)

# Save the resulting image
#cv2.imwrite("contours_detected.jpg", contour_image)

# Display the resulting image
cv2.imshow("Canny Converted", cannyRst)
cv2.imshow("Contours Detected", contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
