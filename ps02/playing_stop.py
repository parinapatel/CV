import cv2
import numpy as np
import math

def show_img(str, img):
    cv2.imshow(str, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_length(line):
    x1, y1 = line[0], line[1]
    x2, y2 = line[2], line[3]
    length = math.sqrt(abs(x1-x2)**2 + abs(y1-y2)**2)
    return length

def line_angle(line):
    x1, y1 = line[0],line[1]
    x2, y2 = line[2], line[3]

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


def proximal_pts(p1, p2, threshold):
    if (abs(p1[0]-p2[0]) < threshold and abs(p1[1]-p2[1]) < threshold):
        return True
    else:
        return False
    

img_in = cv2.imread("./input_images/test_images/stop_249_149_blank.png")

sign = img_in.copy()
sign_hsv = cv2.cvtColor(sign, cv2.COLOR_BGR2HSV)
lower = np.array([0, 43, 46])
upper = np.array([10, 255, 230])

sign_draw = cv2.inRange(sign_hsv, lower, upper)
show_img("",sign_draw)
edges = cv2.Canny(sign, 100, 200)
show_img("edges of tl", edges)


lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 36, threshold=20, minLineLength=5, maxLineGap=5)

lines = lines.reshape(lines.shape[0], lines.shape[2])

print(len(lines))

# show_img("", edges)

lengths = [5 * int(get_length(line) / 5) for line in lines]
# print(sorted(lengths))

lines = np.array([lines[i] for i in range(len(lines)) if 25 <= lengths[i] <= 40])
# print(len(lines))

i = 0
for line in lines:
    if 40 >= 5 * int(get_length(line) / 5) >= 25 or lengths[i] == 995:
        i += 1
        cv2.line(sign_draw, (line[0], line[1]), (line[2], line[3]), (0, 0, 0), 2)
# show_img("", sign_draw)
linesS = filter_angles(lines, [0, 90])
linesA = filter_angles(lines, [45, -45])
octagons = []
commons = []
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
                    common = [int((pin1[0] + pin2[0]) / 2) + 2, int((pin1[1] + pin2[1]) / 2) - 3]
                    placed = False
                    l1t = tuple(l1)
                    l2t = tuple(l2)
                    ct = tuple(common)
                    l1end = p2 if pin1 == p1 else p1
                    l2end = p4 if pin2 == p3 else p3

                    # check pin1 distance from l2end and pin2 distance from l1end => whatever matches the length take that one
                    l1_length = get_length(l1)
                    l2_length = get_length(l2)
                    l = max(l1_length, l2_length)
                    pin1l2 = get_length([pin1[0], pin1[1], l2end[0], l2end[1]])
                    pin2l1 = get_length([pin2[0], pin2[1], l1end[0], l1end[1]])
                    if abs(pin1l2-l) < abs(pin2l1-l):
                        ct = tuple(pin1)
                    else:
                        ct = tuple(pin2)
                    #print(ct)
                    commons.append(ct)
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
if len(octagons) == 0:
    print("no octagons")
for o in octagons:
    if len(o["lines"]) >= 3:
        fo["lines"] = fo["lines"].union(o["lines"])
        fo["common"] = fo["common"].union(o["common"])
print(len(fo["lines"]))
print(fo["common"])
for line in fo["lines"]:
    cv2.line(sign, (line[0], line[1]), (line[2], line[3]), (0,0,0), thickness=2)

show_img("", sign)
# cv2.destroyAllWindows()

"""through list of common points + l1end and l2end"""

points = list(fo["common"])
pd = list([0 for i in range(len(fo["common"]))])

for i in range(len(points)):
    distances = [proximal_pts(points[i], pt, 15) for pt in points]
    distances[i] = False
    for j in range(i + 1, len(distances)):
        if pd[j] == 0 and distances[j]:
            pd[j] = 1
# print(pd)

centerx = int(np.mean([points[i][0] for i in range(len(points)) if pd[i] == 0]))
centery = int(np.mean([points[i][1] for i in range(len(points)) if pd[i] == 0]))

"""through ct and lines"""
for l in fo["lines"]:
    p1 = tuple([l[0],l[1]])
    p2 = tuple([l[2], l[3]])
    for p in [p1,p2]:
        distances =[proximal_pts(c, p, 20) for c in commons]
        if (p in commons) or (sum(distances) >= 1):
            continue
        else:
            commons.append(p)
cx = int(np.mean([c[0] for c in commons]))
cy = int(np.mean([c[1] for c in commons]))

print(len(commons))

print("{} {}".format(cx, cy))

area = sign[centery - 5:centery + 5, centerx - 5:centerx + 5]
red = np.mean(area[:, :, 2])
green = np.mean(area[:, :, 1])
blue = np.mean(area[:, :, 0])
if red > 200:
    print("{} {}".format(centerx, centery))
else:
    print("did not find one :(")