import cv2
import numpy as np
import matplotlib.pyplot as plt
import anglecalc
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
    print("Number of filtered contours :" + str(len(filtered_contours)))

    # Draw contours on the original image
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rstimage = cv2.drawContours(image=contour_image, contours=filtered_contours, contourIdx=-1, color=(255, 0, 0), thickness=1,
                     lineType=cv2.LINE_AA)

    if filtered_contours:
        first_contour_coordinates = filtered_contours[0].reshape(-1, 2)  # Reshape to Nx2 array
        x = [point[0] for point in first_contour_coordinates]
        y = [point[1] for point in first_contour_coordinates]

        segment_length = 100
        num_segments = len(x) // segment_length
        remainder = len(x) % segment_length
        if remainder > 0:
            num_segments += 1

        print("Number of color changes on the plot:", num_segments)

        # Plot the line with color change every 50 coordinates
        for i in range(0, len(x), segment_length):
            plt.plot(x[i:i + segment_length], y[i:i + segment_length], marker='o',
                     color='C{}'.format(i // segment_length))  # Using color cycling

        # Add labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Line Plot')

        # Show the plot
        plt.grid(True)
        plt.show()
        #print("Coordinates of the contour:")
        tc = 0
        for x, y in first_contour_coordinates:
            print(f"({x}, {y})")
            tc += 1
        #for x,y in first_contour_coordinates:
        #    print(anglecalc.calculate_angle(x,y,x+1,y+1,x+20,y+20,x+21,y+21))
            # Calculate angles between 70° and 110°
        seen_points = set()

        for i in range(len(first_contour_coordinates) - 4):
                x1, y1 = first_contour_coordinates[i]
                x2, y2 = first_contour_coordinates[i + 1]
                x4, y4 = first_contour_coordinates[i + 3]
                x5, y5 = first_contour_coordinates[i + 4]

                angle = anglecalc.calculate_angle(x1, y1, x2, y2, x4, y4, x5, y5)
                if 170 <= angle <= 190:
                    points = [(x2, y2), (x4, y4), (x5, y5)]
                    if not any(point in seen_points for point in points):
                        print(f"Angle: {angle}°, Points: {points}")
                        seen_points.update(points)



    else:
        print("No contours found.")
    print("Total tuple of (x,y) values: " + str(tc))
    return rstimage

def findCorner():
    pass
# Read the image
img = cv2.imread("images/allpieces/allpiecesCannypaint.png")

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
