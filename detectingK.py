import cv2
import matplotlib.pyplot as plt
import anglecalc
import numpy as np
from itertools import permutations

setOfPoints = [(125, 1130), (46, 1110), (48, 1227), (120, 1193), (119, 1221), (175, 1214), (201, 1136), (168, 1102)]

xpoints = np.array([125,46,48,120,119,175,201,168])
ypoints = np.array([1130,1110, 1227,1193,1221,1214,1136,1102])
plt.plot(xpoints,ypoints, 'o')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Informative Points Kwarg')

# Show the plot
plt.grid(True)
plt.show()


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

# Function to find all permutations of points and check for 90° angles
def find_90_deg_angles(setOfPoints):
    valid_permutations = []
    for permutation in permutations(setOfPoints, 3):  # Considering permutations of 3 points
        x1, y1 = permutation[0]
        x2, y2 = permutation[1]
        x3, y3 = permutation[2]
        vector1_length = np.linalg.norm(np.array([x2 - x1, y2 - y1]))
        # Check if permutation is not a reverse of another permutation
        if 0.8 * vector1_length <= np.linalg.norm(np.array([x3 - x2, y3 - y2])) <= 1.2 * vector1_length:
            # Check if permutation is not a reverse of another permutation
            if (x1, y1) < (x3, y3):
                angle = calcAngleKwarg(x1, y1, x2, y2, x3, y3)
                if 85 < angle < 95:  # Check if angle is close to 90°
                    valid_permutations.append(permutation)
    return valid_permutations

from collections import Counter

def find_hidden_square(valid_permutations):
    # Flatten the valid permutations to get all points
    all_points = [point for perm in valid_permutations for point in perm]
    # Count the occurrences of each point
    point_counts = Counter(all_points)
    # Select the four coordinates with the highest counts
    most_common_points = point_counts.most_common(4)
    # Extract only the coordinates
    hidden_square = [point[0] for point in most_common_points]
    return hidden_square


valid_permutations = find_90_deg_angles(setOfPoints)
print("Length of Valid perm:", len(valid_permutations))
print("Permutations with 90° angles:", valid_permutations)
hidden_square = find_hidden_square(find_90_deg_angles(setOfPoints))
print("Hidden square coordinates:", hidden_square)

