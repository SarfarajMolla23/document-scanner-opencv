import cv2
import numpy as np

def initializeTrackbars():
    # Create a window for the trackbars
    cv2.namedWindow("Trackbars")
    # Create trackbars for the Canny edge detection thresholds
    cv2.createTrackbar("Threshold1", "Trackbars", 50, 255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 150, 255, nothing)

def nothing(x):
    pass

def valTrackbars():
    # Get the current positions of the trackbars
    threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    return [threshold1, threshold2]

# Find the biggest contour with exactly 4 points
def biggestContour(contours):
    maxArea = 0
    biggest = np.array([])
    for c in contours:
        area = cv2.contourArea(c)
        if area > maxArea:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)  # Approximate the contour to a polygon
            if len(approx) == 4:  # Check if the contour has 4 points
                biggest = approx
                maxArea = area
    return biggest, maxArea

# Reorder the contour points for perspective transformation
def reorder(points):
    if points.shape[0] != 4:
        print("Error: Contour does not have exactly 4 points!")
        return points

    points = points.reshape((4, 2))  # Ensure the points are in 4x2 shape
    newPoints = np.zeros((4, 1), dtype=np.float32)

    add = points.sum(1)
    diff = np.diff(points, axis=1)

    newPoints[0] = points[np.argmin(add)]  # top-left
    newPoints[3] = points[np.argmax(add)]  # bottom-right
    newPoints[1] = points[np.argmin(diff)]  # top-right
    newPoints[2] = points[np.argmax(diff)]  # bottom-left

    return newPoints

# Draw rectangle around the biggest contour
def drawRectangle(img, biggest, thickness):
    cv2.line(img, tuple(biggest[0]), tuple(biggest[1]), (0, 255, 0), thickness)
    cv2.line(img, tuple(biggest[1]), tuple(biggest[2]), (0, 255, 0), thickness)
    cv2.line(img, tuple(biggest[2]), tuple(biggest[3]), (0, 255, 0), thickness)
    cv2.line(img, tuple(biggest[3]), tuple(biggest[0]), (0, 255, 0), thickness)
    return img
