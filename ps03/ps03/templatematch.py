import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import rotate


img_rgb = cv2.imread('./input_images/test_images/rectangle_wall_noisy.png')

blurred = cv2.GaussianBlur(img_rgb, (5,5), 0)
denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21)
img_gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
#img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('./input_images/template.jpg',0)
w, h = template.shape[::-1]

for dig in range(180):
    templ = rotate(template, dig, mode='constant', reshape=False)
    print(dig)
    res = cv2.matchTemplate(img_gray,templ,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    print(loc)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    if len(loc[0]) != 0:
        cv2.imshow("", img_rgb)
        cv2.waitKey(0)
#cv2.imwrite('res.png',img_rgb)
