import cv2
import numpy as np
import copy

def show_img(str, img):
    cv2.imshow(str, img)
    cv2.waitKey(0)

def draw_lines(lines, img):
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

def draw_circles(circles, img):
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(img, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv2.circle(img, center, radius, (255, 0, 255), 3)

def get_vertical(lines):
    lshape = lines.shape
    lr = (lines.reshape(lshape[0], lshape[2])).T
    x = lr[0, lr[1, :] == 0]  # parallel to y axis
    minx = np.min(x)
    maxx = np.max(x)
    return (minx, maxx)

def get_center(circles, minx, maxx):
    cshape = circles.shape
    cr = circles.reshape(cshape[1], cshape[2])
    center = np.sort(cr[(cr[:, 0] > minx) * (cr[:, 0] < maxx), 1])[1]
    cr = cr.T
    centerxy = cr[0:2,cr[1,:]==center].reshape(2)
    return centerxy


tl = cv2.imread("input_images\\simple_tl.png")
tl_draw = copy.deepcopy(tl)
#show_img("sample tl", tl)

#should I convert it to grayscale???????
tl_gray = cv2.cvtColor(tl, cv2.COLOR_BGR2GRAY)
#show_img("gray scale", tl_gray)

#check how canny edge filter shows up)
edges = cv2.Canny(tl, 100, 200)
#show_img("edges of tl", edges)

#get all lines using Hough Transform
lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
#print all lines
draw_lines(lines, tl_draw)

#show_img("lines on tl", tl_draw)

#get x axis of the vertical lines
xmin, xmax = get_vertical(lines)

"""
So far, all tl images, have just two vertical lines - related to tl itself. 
The vertical lines have x=a kind of structure, thus its y component is always 0. 
so to identify the tl, first find the two vertical lines - which will give you approximate location for center x
then find the three circles in between those two x values, and identify the circle in between - > that will provide y.
"""

circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                          param1=50, param2=30,
                          minRadius=5, maxRadius=50)
if circles is not None:
    circles = np.uint16(np.around(circles))
    draw_circles(circles, tl_draw)
    print(circles)
    cshape = circles.shape
    cr = circles.reshape(cshape[1], cshape[2])
    cr = cr[(cr[:, 0] > xmin) * (cr[:, 0] < xmax), 0:2]
    for circle in cr:
        row = int(circle[0])
        col = int(circle[1])
        if tl[row, col, :][2] == tl[row,col,:][1] and tl[row,col,:][2] > 250: color = 'yellow'
        elif tl[row, col, :][2] > 250 and tl[row, col, :][1] < 200: color = 'red'
        elif tl[row, col, :][1] > 250 and tl[row, col, :][2] < 200: color = 'green'

    #show_img("lines and circles", tl_draw)
    center_tl = get_center(circles, xmin,xmax)





cv2.destroyAllWindows()