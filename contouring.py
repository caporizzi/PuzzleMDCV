import cv2
import matplotlib.pyplot as plt
import anglecalc
import numpy as np
from itertools import permutations
from collections import Counter

def applyBinaryAndDrawContours(image, draw=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    ret, thresh = cv2.threshold(gray, 1, 999, cv2.THRESH_BINARY)
    if draw:
        cv2.imshow('Binary', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    min_contour_length = 200

    filtered_contours = []

    for contour in contours:
        contour_length = cv2.arcLength(contour, closed=True)
        if contour_length > min_contour_length:
            filtered_contours.append(contour)
    image_copy = image.copy()
    for idx, contour in enumerate(filtered_contours):
        cv2.drawContours(image=image_copy, contours=[contour], contourIdx=-1, color=(0, 255, 0), thickness=1,
                         lineType=cv2.LINE_AA)
        #add annotation to each contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Add annotation
            cv2.putText(image_copy, f"Cont {idx}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
    print("Length of filtered contours: " + str(len(filtered_contours)))

    if draw:
        cv2.imshow('None approximation', image_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return filtered_contours


def checkContoursAndDraw(contours_data, index, draw=True):
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

    if draw:
        # Add labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Line Plot')

        # Show the plot
        plt.grid(True)
        plt.show()
        """
        for x, y in ravelled_contour:
            print(f"({x}, {y})")
        """
    return ravelled_contour

def findAngles(contours_data):
    seen_points = set()
    seen_angles = {}
    useful_points = []
    for i in range(len(contours_data) - 4):
        x1, y1 = contours_data[i]
        x2, y2 = contours_data[i + 1]
        x3, y3 = contours_data[i + 2]
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
                    useful_points.append((x3, y3))
            else:
                # Add the angle to the dictionary
                seen_angles[angle] = 1

    print("My informative points of length: " + str(len(useful_points)) + " are: " + str(useful_points) )
    #print("Number of seen points: " + str(seen_points))

    return useful_points #y wannabe corners

def calcAngleKwarg(x1,y1,x2,y2,x3,y3):
    # Define vectors representing the two lines
    line1_vector = np.array([x2 - x1, y2 - y1])
    line2_vector = np.array([x3 - x2, y3 - y2])

    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(line1_vector, line2_vector)
    magnitude_line1 = np.linalg.norm(line1_vector)
    magnitude_line2 = np.linalg.norm(line2_vector)

    # Calculate the cosine of the angle between the two lines
    cosine_angle = dot_product / (magnitude_line1 * magnitude_line2)

    # Calculate the angle in degrees
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)

    return angle

def find_90_deg_angles(setOfPoints):
    valid_permutations = []
    for permutation in permutations(setOfPoints, 3):  # Considering permutations of 3 points
        x1, y1 = permutation[0]
        x2, y2 = permutation[1]
        x3, y3 = permutation[2]
        vector1_length = np.linalg.norm(np.array([x2 - x1, y2 - y1]))
        vector2_length = np.linalg.norm(np.array([x3 - x2, y3 - y2]))
        # Check if both vector lengths are within the specified range
        if (65 <= vector1_length <= 150) and (65 <= vector2_length <= 200) and \
                (0.6 * vector1_length <= vector2_length <= 1.5 * vector1_length) and \
                (0.6 * vector2_length <= vector1_length <= 1.5 * vector2_length):
            # Check if permutation is not a reverse of another permutation
            if (x1, y1) < (x3, y3):
                angle = calcAngleKwarg(x1, y1, x2, y2, x3, y3)
                if 81 < angle < 99:  # Check if angle is close to 90°
                    valid_permutations.append(permutation)
    print(valid_permutations)
    return valid_permutations

def find_hidden_square(valid_permutations):
    # Flatten the valid permutations to get all points
    all_points = [point for perm in valid_permutations for point in perm]
    # Count the occurrences of each point
    point_counts = Counter(all_points)
    # Select the four coordinates with the highest counts
    most_common_points = point_counts.most_common(4)
    # Extract only the coordinates
    hidden_square = [point[0] for point in most_common_points]
    print(hidden_square)
    return hidden_square

def plot_points(setOfPoints, title):
    xpoints = [point[0] for point in setOfPoints]
    ypoints = [point[1] for point in setOfPoints]
    plt.plot(xpoints, ypoints, 'o')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_hidden_square(setOfPoints, hidden_square, title):
    xpoints = [point[0] for point in setOfPoints]
    ypoints = [point[1] for point in setOfPoints]

    plt.plot(xpoints, ypoints, 'o')

    # Extract the x and y coordinates of the hidden square
    square_x = [point[0] for point in hidden_square]
    square_y = [point[1] for point in hidden_square]

    # Plot lines connecting the points of the hidden square
    for i in range(len(square_x)):
        plt.plot([square_x[i], square_x[(i + 1) % 4]], [square_y[i], square_y[(i + 1) % 4]], 'r-')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(title)
    plt.grid(True)
    plt.show()

def segment_contour(ravelled_contour, square_points, draw=True):
    segments = [[] for _ in range(4)]  # Create four empty segments

    current_segment_index = 0

    for point in ravelled_contour:
        if any((point == corner).all() for corner in square_points):
            # If we encounter a corner point, move to the next segment
            current_segment_index = (current_segment_index + 1) % 4

        segments[current_segment_index].append(point)
    
    colors = ['r', 'g', 'b', 'y']  # Different colors for each segment

    plt.figure(figsize=(6, 6))

    for i, segment in enumerate(segments):
        x, y = zip(*segment)
        plt.plot(x, y, color=colors[i], label=f"Segment {i + 1}")

    if draw:
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.show()

    return segments

# Example usage:
# segmented_sides = segment_contour(ravelled_contour, hidden_square)

def removeBorders(image, contour):
    pass



def ravelContours(index):
    contoursss = contours
    ravelled_contour = contoursss[index].reshape(-1,2)

if __name__ == "__main__":
    originalImage = cv2.imread('images/allpieces/all.png')
    contours = applyBinaryAndDrawContours(originalImage)
    for i in range(5):
        ravelledContours = checkContoursAndDraw(contours, i)
        #removeBorders(originalImage, ravelledContours)
        useful_points = findAngles(ravelledContours)
        allPermutations = find_90_deg_angles(useful_points)
        hiddenSquare = find_hidden_square(allPermutations)
        #plot_points(useful_points, "Informative Points Kwarg")
        segmented_sides = segment_contour(ravelledContours, hiddenSquare)
        #plot_hidden_square(useful_points, hiddenSquare, 'Hidden Square')



