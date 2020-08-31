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

    if y1 - y2 == 0:
        return 0

    if x1 - x2 == 0:
        return 90

    angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))

    # make it in multiplcation of 5
    angle = 5 * (int(angle / 5))
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
    x, y = int(center[0]), int(center[1])
    text = "(({},{}),'{}')".format(x, y, state)
    xs, ys = image_in.shape[0], image_in.shape[1]
    orgx = x + 50 if x + 50 + 200 < xs else x - 200
    orgy = y
    cv2.putText(img, text, (orgx, orgy), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
    cv2.putText(img, "*", (x - 8, y + 9), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
    return img


def get_midpt(line):
    x1, y1 = line[0], line[1]
    x2, y2 = line[2], line[3]
    return [(x1 / 2) + (x2 / 2), (y1 / 2) + (y2 / 2)]


def get_length(line):
    x1, y1 = line[0], line[1]
    x2, y2 = line[2], line[3]
    length = math.sqrt(abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2)
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
            distance = get_length([l1c[0], l1c[1], l2c[0], l2c[1]])
            if m < distance:
                m = distance
                l1m = l1
                l2m = l2
    l1c = get_midpt(l1m)
    l2c = get_midpt(l2m)
    center = get_midpt([l1c[0], l1c[1], l2c[0], l2c[1]])
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
    x, y = (749.75, 349.75)
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
    if (abs(p1[0] - p2[0]) < threshold and abs(p1[1] - p2[1]) < threshold):
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


def pt_in_circle(c, r, p):
    if (c[0] - r < p[0] < c[0] + r) and (c[1] - r < p[1] < c[1] + r):
        return True
    return False


def get_lines_in_circles(lines, circles):
    linesIn = []
    for circle in circles:
        c = (circle[0], circle[1])
        r = circle[2]
        for line in lines:
            p1 = (line[0], line[1])
            p2 = (line[2], line[3])
            if pt_in_circle(c, r, p1) and pt_in_circle(c, r, p2):
                placed = False
                for l_pair in linesIn:
                    if tuple(circle) == l_pair["cir"]:
                        l_pair["l"].add(tuple(line))
                        placed = True
                if not placed:
                    linesIn.append({"cir": tuple(circle), "l": set({tuple(line)})})
    return linesIn


sign = cv2.imread("input_images\\scene_stp_1.png")
# sign = cv2.imread("input_images\\scene_wrng_1.png")

# sign = cv2.imread("input_images\\scene_all_signs.png")
sign = cv2.imread("input_images\\test_images\\stop_249_149_blank.png")
sign_draw = copy.deepcopy(sign)
# show_img("sample tl", tl)

# gray_img = cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY)
# blured = cv2.GaussianBlur(gray_img, (5,5),2)
# edges_blur = cv2.Canny(blured, 5, 10)
# show_img("blurred edges", edges_blur)

# check how canny edge filter shows up)
edges = cv2.Canny(sign_draw, 100, 200)
show_img("edges of tl", edges)

lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 36, threshold=20, minLineLength=5, maxLineGap=5)
lines = lines.reshape(lines.shape[0], lines.shape[2])

lengths = [5 * int(get_length(line) / 5) for line in lines]
print(sorted(lengths))

lines = np.array([lines[i] for i in range(len(lines)) if 25 <= lengths[i] <= 40])
print(len(lines))

i = 0
for line in lines:
    if 40 >= 5 * int(get_length(line) / 5) >= 25 or lengths[i] == 995:
        i += 1
        cv2.line(sign_draw, (line[0], line[1]), (line[2], line[3]), (0, 0, 0), 2)
show_img("", sign_draw)

linesS = filter_angles(lines, [0, 90])
linesA = filter_angles(lines, [45, -45])

linesV = filter_angles(lines, [90])
linesH = filter_angles(lines, [0])
linesP = filter_angles(lines, [45])
linesN = filter_angles(lines, [-45])

print(linesS)
print(linesA)

octagons = []

for l1 in linesS:
    p1 = (l1[0], l1[1])
    p2 = (l1[2], l1[3])
    for l2 in linesA:
        p3 = (l2[0], l2[1])
        p4 = (l2[2], l2[3])
        for pin1 in [p1, p2]:
            for pin2 in [p3, p4]:
                # print("{} {} {} {}".format(pin1, pin2, abs(pin1[0]-pin2[0]), abs(pin1[1]-pin2[1])))
                if proximal_pts(pin1, pin2, 10):
                    common = [int((pin1[0] + pin2[0]) / 2), int((pin1[1] + pin2[1]) / 2)]
                    placed = False
                    l1t = tuple(l1)
                    l2t = tuple(l2)
                    ct = tuple(common)
                    l1end = p2 if pin1 == p1 else p1
                    l2end = p4 if pin2 == p3 else p3
                    for octagon in octagons:
                        if l1t in octagon["lines"] or l2t in octagon["lines"]:
                            octagon["lines"].add(l1t)
                            octagon["lines"].add(l2t)
                            octagon["common"].add(ct)
                            octagon["common"].add(l1end)
                            octagon["common"].add(l2end)
                            placed = True
                    if not placed:
                        octagon = {"lines": set([l1t, l2t]), "common": set({ct, l1end, l2end})}
                        octagons.append(octagon)

fo = {"lines": set(), "common": set()}
for o in octagons:
    if len(o["lines"]) >= 3:
        fo["lines"] = fo["lines"].union(o["lines"])
        fo["common"] = fo["common"].union(o["common"])
print(len(fo["lines"]))

points = list(fo["common"])
pd = list([0 for i in range(len(fo["common"]))])

for i in range(len(points)):
    distances = [proximal_pts(points[i], pt, 15) for pt in points]
    distances[i] = False
    cx = points[i][0]
    cy = points[i][1]
    cx = np.mean([points[i][0] for i in range(len(points)) if distances[i]])
    for j in range(i + 1, len(distances)):
        if pd[j] == 0 and distances[j]:
            pd[j] = 1
print(pd)

centerx = int(np.mean([points[i][0] for i in range(len(points)) if pd[i] == 0]))
centery = int(np.mean([points[i][1] for i in range(len(points)) if pd[i] == 0]))

area = sign_draw[centery - 5:centery + 5, centerx - 5:centerx + 5]
red = np.mean(area[:, :, 2])
green = np.mean(area[:, :, 1])
blue = np.mean(area[:, :, 0])
if red > 200:
    print("finally!!")

cv2.putText(sign_draw, "*", (int(centerx), int(centery)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)

print(points)
for i in range(len(points)):
    if pd[i] == 0:
        cv2.putText(sign_draw, "*", (int(points[i][0]), int(points[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                    thickness=2)
show_img("", sign_draw)

"""
for each point, calculate the distance with each point in the array 
remove the points from array which have distance with the point < 15
"""

# for o in octagons:
#     print("lines: {} \n common: {}".format(o["lines"], o["common"]))
#     for point in o["common"]:
#         cv2.putText(sign_draw, "*", (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),
#                     thickness=2)
#     show_img("",sign_draw)

# oct2 = copy.deepcopy(octagons)


# final_octagons = []
# inter = [set({}) for i in range(len(octagons))]
#
# for i in range(len(octagons)):
#     for j in range(i+1, len(octagons)):
#         if octagons[i]["lines"].intersection(octagons[j]["lines"]) != set():
#             inter[i].add(j)
#
# oc = copy.deepcopy(octagons)
# print(oc)
# for i in range(len(inter)-1, -1, -1):
#     new_set = octagons[i]
#     dependents = list(inter[i])
#     dependents.sort()
#     for j in range(len(dependents)-1,-1,-1):
#         if dependents[j] >= len(octagons):
#             continue
#         old = octagons[dependents[j]]
#         new_set["lines"] = new_set["lines"].union(old["lines"])
#         new_set["common"] = new_set["common"].union(old["common"])
#         octagons.pop(j)
#     octagons[i] = new_set
#     final_octagons.append(new_set)
#
#
# for o in octagons:
#     print("lines: {} \n common: {}".format(o["lines"], o["common"]))
#     for point in o["common"]:
#         cv2.putText(sign_draw, "*", (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),
#                     thickness=2)
#     show_img("",sign_draw)


# for o1 in octagons:
#     for o2 in octagons:
#         if o1["lines"].intersection(o2["lines"]) != set():
#

# print(len(lines))
#
# i=0
# for line in lines:
#     if 40 >= 5*int(get_length(line)/5) >= 25:
#         i+=1
#         cv2.line(sign_draw, (line[0], line[1]), (line[2], line[3]), (0,0,0), 2)
# show_img("",sign_draw)


"""
# for line in lines:
#     cv2.line(sign_draw, (line[0], line[1]), (line[2], line[3]), (0,0,0), 2)

linesS = filter_angles(lines, [0,90])
linesA = filter_angles(lines, [45, -45])

linesV = filter_angles(lines, [90])
linesH = filter_angles(lines, [0])
linesP = filter_angles(lines, [45])
linesN = filter_angles(lines, [-45])

# print(linesV)
# print(linesH)
# print(linesP)
# print(linesN)

octagons = []

for lV in linesV:
    p1 = (lV[0], lV[1])
    p2 = (lV[2], lV[3])
    for lP in linesP:
        p3 = (lP[0], lP[1])
        p4 = (lP[2], lP[3])
        if abs(get_length(lV) - get_length(lP)) <10:
            for lH in linesH:
                p5 = (lH[0], lH[1])
                p6 = (lH[2], lH[3])
                if abs(get_length(lH) - get_length(lP)) < 10:
                    for pin1 in [p1, p2]:
                        for pin2 in [p3,p4]:
                            for pin3 in [p5,p6]:
                                cv2.line(sign_draw, pin1, pin2, (0,0,0), 2)
                                cv2.line(sign_draw, pin2, pin3, (0, 0, 0), 2)
                                show_img("", sign_draw)
                                if proximal_pts(pin1, pin2, 10) and abs(get_length(lV) - get_length(lP)) <10:
                                    print("I am V- P ")
                                    common = [int((pin1[0] + pin2[0]) / 2), int((pin1[1] + pin2[1]) / 2)]
                                    placed = False
                                    lVt = tuple(lV)
                                    lPt = tuple(lP)
                                    ct = tuple(common)
                                    for o in octagons:
                                        if lVt in o["lines"] or lPt in o["lines"]:
                                            o["lines"].add(lVt)
                                            o["lines"].add(lPt)
                                            o["common"].add(ct)
                                            placed = True
                                    if not placed:
                                        o = {"lines":set({lVt, lPt}), "common":set({ct})}
                                if proximal_pts(pin2, pin3, 10) and abs(get_length(lH) - get_length(lP)) <10:
                                    print("I am P - H")
                                    common = [int((pin2[0] + pin3[0]) / 2), int((pin2[1] + pin3[1]) / 2)]
                                    placed = False
                                    lHt = tuple(lH)
                                    lPt = tuple(lP)
                                    ct = tuple(common)
                                    for o in octagons:
                                        if lHt in o["lines"] or lPt in o["lines"]:
                                            o["lines"].add(lHt)
                                            o["lines"].add(lPt)
                                            o["common"].add(ct)
                                            placed = True
                                    if not placed:
                                        o = {"lines":set({lHt, lPt}), "common":set({ct})}
                                        octagons.append(o)

            #points of lH and lV will be the last ones.

"""

# for l1 in linesS:
#     p1 = (l1[0], l1[1])
#     p2 = (l1[2], l1[3])
#     for l2 in linesA:
#         p3 = (l2[0], l2[1])
#         p4 = (l2[2], l2[3])
#         for pin1 in [p1, p2]:
#             for pin2 in [p3, p4]:
#                 # print("{} {} {} {}".format(pin1, pin2, abs(pin1[0]-pin2[0]), abs(pin1[1]-pin2[1])))
#                 if proximal_pts(pin1, pin2, 10) and abs(get_length(l1)-get_length(l2)) < 10:
#                     common = [(pin1[0] + pin2[0]) / 2, (pin1[1] + pin2[1]) / 2]
#                     placed = False
#                     l1t = tuple(l1)
#                     l2t = tuple(l2)
#                     ct = tuple(common)
#                     for octagon in octagons:
#                         if l1t in octagon["lines"] or l2t in octagon["lines"]:
#                             octagon["lines"].add(l1t)
#                             octagon["lines"].add(l2t)
#                             octagon["common"].add(ct)
#                             placed = True
#                     if not placed:
#                         octagon = {"lines": set([l1t, l2t]), "common": set({ct})}
#                         octagons.append(octagon)
"""
print(len(octagons))

for o in octagons:
    print("lines: {} \n common: {}".format(o["lines"], o["common"]))
    for point in o["common"]:
        cv2.putText(sign_draw, "*", (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),
                    thickness=2)
    show_img("",sign_draw)

For Do Not Enter signs
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=15, param2=20,
                               minRadius=5, maxRadius=50)
if circles is None: exit() # should become return 0, 0

circles = np.uint16(np.around(circles))
cshape = circles.shape
centers = circles.reshape(cshape[1], cshape[2])


# will get circles with lines inside of it
linesIn = get_lines_in_circles(lines, centers)

for l_pair in linesIn:
    circle = np.array(l_pair["cir"])
    r = (circle[2] / 2).astype(int)
    red = np.mean(sign_draw[circle[1] - r:circle[1] + r,
                  circle[0] - r:circle[0] + r,
                  2])
    # check circle color
    if red > 200:
        c = (circle[0], circle[1])
        area = sign_draw[c[1]-5:c[1]+5, c[0]-5:c[0]+5]
        red = np.mean(area[:, :, 2])
        green = np.mean(area[:, :, 1])
        blue = np.mean(area[:, :, 0])
        if red > 200 and blue > 200 and green > 200:
            print("{} {}".format(int(c[0]), int(c[1])))
            cv2.putText(sign_draw, "*", (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        thickness=2)
            show_img("sign",sign_draw)


#show_img("lines and circles", sign_draw)
# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     draw_circles(circles, sign_draw)
#     show_img("lines and circles", sign_draw)

"""

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
