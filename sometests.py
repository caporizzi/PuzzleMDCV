import cv2

image = cv2.imread('images/allpieces/allpiecesflash.jpg')

def binaryTransform(image):
    _, binary_image = cv2.threshold(image, 100, 150, cv2.THRESH_BINARY)
    return binary_image

binary_image = binaryTransform(image)

cv2.imshow('Binary Converter', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()