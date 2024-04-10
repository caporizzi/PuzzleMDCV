import numpy as np

def calculate_angle(x1, y1, x2, y2, x4, y4, x5, y5):
    """Calculate angle between two lines formed by five points."""
    # Define vectors representing the two lines
    line1_vector = np.array([x2 - x1, y2 - y1])
    line2_vector = np.array([x5 - x4, y5 - y4])

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


