import cv2
import numpy as np
import copy
import math

def show_img(str, img):
    cv2.imshow(str, img)
    cv2.waitKey(0)

def line_angle(line):
    x1, y1 = line[0:2]
    x2, y2 = line[2:]

    angle = np.rad2deg(np.arctan2(y2-y1, x2-x1))

    #make it in multiplcation of 5
    angle = 5*(int(angle/5))
    return angle

def filter_angles(lines, angles):
    lines_filtered = [line for line in lines if line_angle(line) in angles]
    return lines_filtered


def mark_traffic_signs(image_in, signs_dict):
    img = image_in.copy()
    for sign_name, center in signs_dict.items():
        x, y = int(center[0]), int(center[1])
        text = "(({},{}),'{}')".format(x, y, sign_name)
        xs, ys = image_in.shape[0], image_in.shape[1]
        orgx = x + 50 if x + 50 + 200 < xs else x - 200
        orgy = y
        cv2.putText(img, text, (orgx, orgy), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
        cv2.putText(img, "*", (x - 8, y + 9), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
    return img

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

def get_midpt(line):
    x1,y1 = line[0],line[1]
    x2, y2 = line[2], line[3]
    return [(x1/2)+(x2/2),(y1/2)+(y2/2)]

def get_length(line):
    x1, y1 = line[0], line[1]
    x2, y2 = line[2], line[3]
    length = math.sqrt(abs(x1-x2)**2 + abs(y1-y2)**2)
    return length

def get_centers(lines):
    m = 0
    l1m = None
    l2m = None
    for l1 in lines:
        for l2 in lines:
            if tuple(l1) == tuple(l2):
                continue
            l1c = get_midpt(l1)
            l2c = get_midpt(l2)
            distance = get_length([l1c[0],l1c[1],l2c[0],l2c[1]])
            if m < distance:
                m = distance
                l1m = l1
                l2m = l2
    l1c = get_midpt(l1m)
    l2c = get_midpt(l2m)
    center = get_midpt([l1c[0],l1c[1],l2c[0],l2c[1]])
    return center

def get_square_centers(lines):
    # filter lines with +45 and -45 angles
    side_45 = filter_angles(lines, [45])
    side__45 = filter_angles(lines, [-45])

    # list of set of square lines
    squares = []
    for line1 in side_45:
        l1c = get_midpt(line1)
        for line2 in side__45:
            l2c = get_midpt(line2)
            """
            if both lines are almost same length and 
            either their mid point x values or mid point y values are approximately the same
            then they belong to the same set of square
            """
            if ((abs(get_length(line1) - get_length(line2)) < 3)
                    and ((abs(l1c[0] - l2c[0]) < 3) or (abs(l1c[1] - l2c[1]) < 3))):
                placed = False
                l1 = tuple(line1)
                l2 = tuple(line2)
                if len(squares) != 0:
                    for square in squares:
                        if l1 in square or l2 in square:
                            square.add(l1)
                            square.add(l2)
                            placed = True
                if not placed:
                    square = set({l1, l2})
                    squares.append(square)

    print(squares)
    # for square in squares:
    #     for line in square:
    #         #cv2.line(sign_draw, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
    #         x, y = line[0],line[1]
    #         cv2.putText(sign_draw, "*", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
    #                     thickness=2)
    x,y = (749.75, 349.75)
    cv2.putText(sign_draw, "*", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                thickness=2)
    show_img("lines", sign_draw)


    centers = []

    for square in squares:
        """
        get lines of same angle, get their mid points and the midpoint of these midpoints
        from the two groups, find mid-mid point
        """



        # lines = [np.array(line) for line in list(square)]
        # side45 = filter_angles(lines, [45])
        # side_45 = filter_angles(lines, [-45])
        #
        # c = np.array([get_centers(side45), get_centers(side_45)])
        #
        # center = np.mean(c, axis=1)
        # centers.append(center)
    return centers

def proximal_pts(p1, p2, threshold):
    if (abs(p1[0]-p2[0]) < threshold and abs(p1[1]-p2[1]) < threshold):
        return True
    else:
        return False

def get_diamonds(lines):
    l45 = filter_angles(lines, [45])
    l_45 = filter_angles(lines, [-45])

    diamonds = []

    for l1 in l45:
        common = []
        p1 = (l1[0], l1[1])
        p2 = (l1[2], l1[3])
        for l2 in l_45:
            p3 = (l2[0], l2[1])
            p4 = (l2[2], l2[3])
            for pin1 in [p1, p2]:
                for pin2 in [p3, p4]:
                    # print("{} {} {} {}".format(pin1, pin2, abs(pin1[0]-pin2[0]), abs(pin1[1]-pin2[1])))
                    if proximal_pts(pin1, pin2, 10):
                        common = [(pin1[0] + pin2[0]) / 2, (pin1[1] + pin2[1]) / 2]
                        placed = False
                        l1t = tuple(l1)
                        l2t = tuple(l2)
                        ct = tuple(common)
                        for diamond in diamonds:
                            if l1t in diamond["lines"] or l2t in diamond["lines"]:
                                diamond["lines"].add(l1t)
                                diamond["lines"].add(l2t)
                                diamond["common"].add(ct)
                                placed = True
                        if not placed:
                            diamond = {"lines": set([l1t, l2t]), "common": set({ct})}
                            diamonds.append(diamond)
    return diamonds


def draw_circles(circles, img):
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(img, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv2.circle(img, center, radius, (255, 0, 255), 3)


sign = cv2.imread("input_images\\scene_dne_1.png")
#sign = cv2.imread("input_images\\scene_all_signs.png")

sign_draw = copy.deepcopy(sign)
#show_img("sample tl", tl)

# gray_img = cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY)
# blured = cv2.GaussianBlur(gray_img, (5,5),2)
# edges_blur = cv2.Canny(blured, 5, 10)
# show_img("blurred edges", edges_blur)

#check how canny edge filter shows up)
edges = cv2.Canny(sign_draw, 100, 200)
#show_img("edges of tl", edges)

lines = cv2.HoughLinesP(edges, rho=1, theta = np.pi/36, threshold=20, minLineLength=5, maxLineGap=5)
lines = lines.reshape(lines.shape[0], lines.shape[2])

circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=15, param2=20,
                               minRadius=5, maxRadius=50)

for line in lines:
    cv2.line(sign_draw, (line[0], line[1]), (line[2], line[3]), (0,0,0), 2)

show_img("lines and circles", sign_draw)
# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     draw_circles(circles, sign_draw)
#     show_img("lines and circles", sign_draw)



"""for construction and warning
# for diamond in diamonds:
#     print("lines: {} \n common: {}".format(diamond["lines"], diamond["common"]))
#     for point in diamond["common"]:
#         cv2.putText(sign_draw, "*", (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
#                     thickness=2)
#     show_img("",sign_draw)
# diamonds = get_diamonds(lines)
#
# for diamond in diamonds:
#     centerx = np.mean([c[0] for c in diamond["common"]])
#     centery = np.mean([c[1] for c in diamond["common"]])
#     area = sign_draw[int(centery) - 5:int(centery) + 5, int(centerx) - 5:int(centerx) + 5]
#     red = np.mean(area[:, :, 2])
#     green = np.mean(area[:, :, 1])
#     print("{} {}".format(red, green))
#     if red > 200 and green > 200:
#         print("warning")
#         cv2.putText(sign_draw, "* warning", (int(centerx), int(centery)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
#                     thickness=2)
#         show_img("", sign_draw)


# centers = get_square_centers(lines)
#
# for center in centers:
#     area = sign_draw[int(center[0])-5:int(center[0])+5, int(center[1])-5:int(center[1]) +5]
#     red = np.mean(area[:,:,2])
#     green = np.mean(area[:,:,1])
#     cv2.putText(sign_draw, "*", (int(center[1]),int(center[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
#     show_img("sign",sign_draw)
#     print(red)
#     print(green)
#     if red > 200 and green > 200:
#         print("warning")
#
#
# print(centers)
# #print("{} {}".format(x,y))

#draw_tl_center(sign_draw, (x,y), "yield")
# show_img("warning", sign_draw)
"""
cv2.destroyAllWindows()