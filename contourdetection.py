import cv2


image = cv2.imread('images/allpieces/allpiecesdetourata.png')

# convert the image to grayscale format
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply binary thresholding
ret, thresh = cv2.threshold(img_gray, 1, 999, cv2.THRESH_BINARY)
# visualize the binary image
cv2.imshow('Binary image', thresh)
cv2.waitKey(0)
cv2.imwrite('image_thres1.jpg', thresh)
cv2.destroyAllWindows()

# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)


min_contour_length = 500

filtered_contours = []

for contour in contours:
    contour_length = cv2.arcLength(contour, closed=True)
    if contour_length > min_contour_length:
        filtered_contours.append(contour)
# draw contours on the original image
print(filtered_contours)
image_copy = image.copy()
for idx, contour in enumerate(filtered_contours):
    cv2.drawContours(image=image_copy, contours=[contour], contourIdx=-1, color=(0, 255, 0), thickness=2,
                     lineType=cv2.LINE_AA)
    """
        M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # Add annotation
        cv2.putText(image_copy, f"Contour {idx}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    """

# see the results
cv2.imshow('None approximation', image_copy)
cv2.waitKey(0)
cv2.imwrite('images/result/successfulDetect.jpg', image_copy)
cv2.destroyAllWindows()