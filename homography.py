import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class Homography:
    MIN_MATCH_COUNT = 700

    import numpy as np
    import cv2
    def find_homography(self):
        if len(self.good) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([self.kp1[m.queryIdx].pt for m in self.good]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.kp2[m.trainIdx].pt for m in self.good]).reshape(-1, 1, 2)
            # Estimate homography
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            # Apply homography to warp one image onto the other
            h, w, _ = self.img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, M)
            self.img2 = cv.polylines(self.img2, [np.int32(dst)], True, (0, 255, 0), 3, cv.LINE_AA)
        else:
            print("Not enough matches are found - {}/{}".format(len(self.good), self.MIN_MATCH_COUNT))
            matchesMask = None
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        img3 = cv.drawMatches(self.img1, self.kp1, self.img2, self.kp2, self.good, None, **draw_params)
        plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB)), plt.show()


    # Load the image
    imageFrame = cv2.imread('./images/allpieces/all.jpg')

    # Convert the imageFrame in BGR(RGB color space) to HSV(hue-saturation-value) color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set range for red color and define mask
    red_lower = np.array([0,120,70], np.uint8)
    red_upper = np.array([10, 160, 200], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Set range for yellow color and define mask
    yellow_lower = np.array([20, 100, 100], np.uint8)
    yellow_upper = np.array([30, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    # Set range for blue color and define mask
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Morphological Transform, Dilation for each color and bitwise_and operator between imageFrame and mask determines to detect only that particular color
    kernel = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv2.dilate(red_mask, kernel)
    res_red = cv2.bitwise_and(imageFrame, imageFrame, mask=red_mask)

    # For yellow color
    yellow_mask = cv2.dilate(yellow_mask, kernel)
    res_green = cv2.bitwise_and(imageFrame, imageFrame, mask=yellow_mask)

    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernel)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame, mask=blue_mask)

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(imageFrame, "Waldo", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

    # Creating contour to track yellow color
    contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    yellow_color_bgr = (56, 245, 255)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), yellow_color_bgr, 2)
            cv2.putText(imageFrame, "Maldo", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (yellow_color_bgr))

    # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(imageFrame, "Pants", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0))

    # Display the result
    cv2.imshow("Color Detection", imageFrame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    def __init__(self):
        self.img1 = cv.imread('./images/contourTraining/edge.jpg', cv.IMREAD_COLOR)  # queryImage
        self.img2 = cv.imread('./images/allpieces/all.jpg', cv.IMREAD_COLOR)  # trainImage
        # Preprocess images if necessary
        # Feature extraction
        self.sift = cv.SIFT_create(nfeatures=1000)  # Adjust nfeatures to detect more keypoints
        gray_img1 = cv.cvtColor(self.img1, cv.COLOR_BGR2GRAY)
        gray_img2 = cv.cvtColor(self.img2, cv.COLOR_BGR2GRAY)
        self.kp1, self.des1 = self.sift.detectAndCompute(gray_img1, None)
        self.kp2, self.des2 = self.sift.detectAndCompute(gray_img2, None)
        # Feature matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(index_params, search_params)
        self.matches = self.flann.knnMatch(self.des1, self.des2, k=2)
        # Good matches based on Lowe's ratio test
        self.good = []
        for m, n in self.matches:
            if m.distance < 0.7 * n.distance:
                self.good.append(m)



# Create an instance of the class and execute the code
homography_instance = Homography()
homography_instance.find_homography()

