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

def draw_tl_center(image_in, center, state):
    img = image_in.copy()
    x,y=int(center[0]), int(center[1])
    text = "(({},{}),'{}')".format(x,y, state)
    xs, ys = image_in.shape[0], image_in.shape[1]
    orgx = x+50 if x+50+200<xs else x-200
    orgy = y
    cv2.putText(img, text, (orgx,orgy), cv2.FONT_HERSHEY_SIMPLEX, .5,(0,0,0))
    cv2.putText(img,"*",(x-8, y+9), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255), thickness=2)
    return img


tl = cv2.imread("input_images\\scene_tl_1.png")
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
                          param1=15, param2=20,
                          minRadius=5, maxRadius=40)
if circles is not None:
    circles = np.uint16(np.around(circles))
    draw_circles(circles, tl_draw)
    #show_img("lines and circles", tl_draw)
    cshape = circles.shape
    centers = circles.reshape(cshape[1], cshape[2])
    centers = centers[(centers[:, 0] > xmin) * (centers[:, 0] < xmax), :]
    sorted(centers, key=lambda x:x[1])
    print(centers)
    r = (centers[:,2]/4).astype(int)
    red = np.mean(tl[centers[0][1]-r[0]:centers[0][1]+r[0], centers[0][0]-r[0]:centers[0][0]+r[0], 2])
    yellow = np.mean(tl[centers[1][1] - r[1]:centers[1][1] + r[1], centers[1][0] - r[1]:centers[1][0] + r[1], 1:])
    green = np.mean(
         tl[centers[2][1] - r[2]:centers[2][1] + r[2], centers[2][0] - r[2]:centers[2][0] + r[2], 1])
    print("{}  {}  {}".format(red, yellow, green))
    #
    colors = ["red","yellow","green"]
    color = colors[np.argmax([red, yellow, green])]
    # c1 = tl_gray[centers[0][0],centers[0][1]]
    # c2 = tl_gray[centers[1][0], centers[1][1]]
    # c3 = tl_gray[centers[2][0], centers[2][1]]



    center_tl = get_center(circles, xmin,xmax)
    tl_draw = draw_tl_center(tl, (center_tl[0],center_tl[1]), color)
    show_img("lines and circles", tl_draw)


cv2.destroyAllWindows()