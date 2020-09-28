import cv2 as cv
import numpy as np

# The video feed is read in as
# a VideoCapture object
#cap = cv.VideoCapture("videoplayback.mp4")

# ret = a boolean return value from
# getting the frame, first_frame = the
# first frame in the entire video sequence
first_frame = cv.imread("./input_images/TestSeq/Shift0.png")

# Converts frame to grayscale because we
# only need the luminance channel for
# detecting edges - less computationally
# expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

# Creates an image filled with zero
# intensities with the same dimensions
# as the frame
mask = np.zeros_like(first_frame)

# Sets image saturation to maximum
mask[..., 1] = 255

frame = cv.imread("./input_images/TestSeq/ShiftR5U5.png")
cv.imshow("input", frame)

gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
                                       None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)

magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

mask[..., 0] = angle * 180 / np.pi / 2

mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

cv.imshow("dense optical flow", rgb)

prev_gray = gray


cv.waitKey(0)
cv.destroyAllWindows()
