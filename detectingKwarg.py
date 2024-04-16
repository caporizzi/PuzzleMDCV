import cv2
import matplotlib.pyplot as plt
import anglecalc
import numpy as np

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


def calcAngle(x1,y1,x2,y2,x3,y3):
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

print(calcAngle(46,1110,48,1227,170,1214))

#Once we found set of points which are 90° between them.
#We can check if there is a symetric point

setOfValidPoints = [(46,1110),(48,1227),(170,1214)]

def findOppositePoint(x1,y1,x2,y2,x3,y3):
    line1_vector = np.array([x2 - x1, y2 - y1])
    line2_vector = np.array([x3 - x2, y3 - y2])
    newx2,newy2 = (48,1227)-line1_vector
    new_x2,new_y2 = (newx2,newy2)-line2_vector
    print(new_x2,new_y2)
findOppositePoint(46,1110,48,1227,170,1214)


#Once we foudn the 4 corners, we can classify using the rest of the key points

#if therer's a keypoint outside my conrer it has 1


