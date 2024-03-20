import cv2
import matplotlib.pyplot as plt
import anglecalc
originalImage = cv2.imread('images/allpieces/detourPieces.png')

def applyBinaryAndDrawContours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    ret, thresh = cv2.threshold(gray, 1, 999, cv2.THRESH_BINARY)
    cv2.imshow('Binary', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    min_contour_length = 500

    filtered_contours = []

    for contour in contours:
        contour_length = cv2.arcLength(contour, closed=True)
        if contour_length > min_contour_length:
            filtered_contours.append(contour)
    image_copy = image.copy()
    for idx, contour in enumerate(filtered_contours):
        cv2.drawContours(image=image_copy, contours=[contour], contourIdx=-1, color=(0, 255, 0), thickness=2,
                         lineType=cv2.LINE_AA)
    print("Length of filtered contours: " + str(len(filtered_contours)))
    cv2.imshow('None approximation', image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return filtered_contours


def checkContoursAndDraw(contours_data, index):
    ravelled_contour = contours_data[index].reshape(-1,2)

    x = [point[0] for point in ravelled_contour]
    y = [point[1] for point in ravelled_contour]
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
    return ravelled_contour

def findAngles(contours_data):
    seen_points = set()
    seen_angles = {}

    for i in range(len(contours_data) - 4):
        x1, y1 = contours_data[i]
        x2, y2 = contours_data[i + 1]
        x4, y4 = contours_data[i + 3]
        x5, y5 = contours_data[i + 4]
        # print(x4,y4)
        angle = anglecalc.calculate_angle(x1, y1, x2, y2, x4, y4, x5, y5)
        # Check if the angle has been seen before
        if 80 <= angle <= 100:

            if angle in seen_angles:
                # Add points to seen_points only if the same angle is obtained twice
                points = [(x1, y1), (x2, y2), (x4, y4), (x5, y5)]
                if not any(point in seen_points for point in points):
                    print(f"Angle: {angle}°, Points: {points}")
                    seen_points.update(points)
            else:
                # Add the angle to the dictionary
                seen_angles[angle] = 1
    print("Number of seen points: " + str(seen_points))





contours = applyBinaryAndDrawContours(originalImage)
ravelledContours = checkContoursAndDraw(contours, 5)
findAngles(ravelledContours)


