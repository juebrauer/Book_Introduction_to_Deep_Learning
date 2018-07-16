import sys
import cv2

print("Your Python version is: " + sys.version)
print("Your OpenCV version is: " + cv2.__version__)

# Load an color image as a grayscale image
img = cv2.imread('coins.jpg',0)

# Show the image
cv2.imshow('Here is a first image displayed with the help of OpenCV',img)

# Wait for user to press a key
cv2.waitKey(0)
