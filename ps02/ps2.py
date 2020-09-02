"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2

import numpy as np

import math

def show_img(str, img):
    cv2.imshow(str, img)
    cv2.waitKey(0)


def get_length(line):
    x1, y1 = line[0], line[1]
    x2, y2 = line[2], line[3]
    length = math.sqrt(abs(x1-x2)**2 + abs(y1-y2)**2)
    return length


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
    if len(x) == 0:
        return None
    minx = np.min(x)
    maxx = np.max(x)
    return (minx, maxx)


def get_center(circles, minx, maxx):
    cshape = circles.shape
    cr = circles.reshape(cshape[1], cshape[2])
    center = np.sort(cr[(cr[:, 0] > minx) * (cr[:, 0] < maxx), 1])[1]
    cr = cr.T
    centerxy = cr[0:2, cr[1, :] == center].reshape(2)
    return centerxy


def get_light_color(img, centers):
    r = (centers[:, 2] / 4).astype(int)
    red = np.mean(img[centers[0][1] - r[0]:centers[0][1] + r[0],
                  centers[0][0] - r[0]:centers[0][0] + r[0],
                  2])
    yellow = np.mean(img[centers[1][1] - r[1]:centers[1][1] + r[1],
                     centers[1][0] - r[1]:centers[1][0] + r[1],
                     1:])
    green = np.mean(
        img[centers[2][1] - r[2]:centers[2][1] + r[2],
        centers[2][0] - r[2]:centers[2][0] + r[2],
        1])
    colors = ["red", "yellow", "green"]
    color = colors[np.argmax([red, yellow, green])]
    return color


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
                        common = [int((pin1[0] + pin2[0]) / 2) + 2, int((pin1[1] + pin2[1]) / 2) - 3]
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


def pt_in_circle(c, r, p):
    if (c[0] - r < p[0] < c[0] + r) and (c[1] -  r < p[1] < c[1] + r):
        return True
    return False


def get_lines_in_circles(lines, circles):
    linesIn = []
    for circle in circles:
        c = (circle[0], circle[1])
        r = circle[2]
        for line in lines:
            p1 = (line[0],line[1])
            p2 = (line[2], line[3])
            if pt_in_circle(c,r,p1) and pt_in_circle(c,r,p2):
                placed = False
                for l_pair in linesIn:
                    if tuple(circle) == l_pair["cir"]:
                        l_pair["l"].add(tuple(line))
                        placed = True
                if not placed:
                    linesIn.append({"cir": tuple(circle), "l": set({tuple(line)})})
    return linesIn


def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    """
       logic: So far, all tl images, have just two vertical lines - related to tl itself. 
       The vertical lines have x=a kind of structure, thus its y component is always 0. 
       so to identify the tl, first find the two vertical lines - which will give you approximate location for center x
       then find the three circles in between those two x values, and identify the circle in between - > that will provide y.
       """

    tl = img_in.copy()

    # edges of traffic light
    edges = cv2.Canny(tl, 100, 200)
    #show_img("edges of image", edges)

    # get all lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if (len(lines)) == 0:
        return (0, 0), ""


    # print all lines
    # draw_lines(lines, tl)
    # show_img("lines on tl", tl)
    # print(lines)
    # get x axis of the vertical lines
    check_circle = False
    tlc = []

    if get_vertical(lines) == None:
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=15, param2=20,
                                   minRadius=max(5, min(radii_range) - 5), maxRadius=min(50, max(radii_range) + 10))
        if circles is None: return (0,0),""

        cshape = circles.shape
        c_checks = circles.reshape(cshape[1], cshape[2])
        near = [0 for i in range(len(c_checks))]
        cx, r = 0, 0
        for i in range(len(c_checks)):
            c1 = c_checks[i]
            for c2 in c_checks:
                if tuple(c1) == tuple(c2):
                    continue
                if abs(c1[0]-c2[0]) <= 5:
                    near[i] += 1
            if near[i] >= 2:
                tlc.append(c_checks[i])
        if len(tlc) == 3:
            tlc = np.array(tlc)
            check_circle = True
            for c in tlc:
                cx += c[0]
                r = max(r, c[2])
            cx = int(cx/len(tlc))
            xmin, xmax = cx-r-10, cx+r+10
        else:
            return (0,0), ""

    #print("I was here {}".format(len(lines)))

    if not check_circle:
        xmin, xmax = get_vertical(lines)
    if xmin == xmax:
        if xmin < 100:
            xmin = 0
        else:
            xmax = tl.shape[1]

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=15, param2=20,
                               minRadius=max(5, min(radii_range) - 5), maxRadius=min(50, max(radii_range) + 10))

    if circles is None: return (0,0), ""

    circles = np.uint16(np.around(circles))
        #draw_circles(circles, tl)
        #show_img("lines and circles", tl)

    cshape = circles.shape
    centers = circles.reshape(cshape[1], cshape[2])

        # get the circles between the two vertical lines
    centers = centers[(centers[:, 0] > xmin) * (centers[:, 0] < xmax), :]
    if len(centers) < 3:
        return (0, 0), ""
    centers = centers[centers[:, 1].argsort()]

        # get the color which is on
    color = get_light_color(tl, centers)
    center_tl = get_center(circles, xmin, xmax)
    return (center_tl[0], center_tl[1]), color

    #     tl_draw = draw_tl_center(tl, (center_tl[0], center_tl[1]), color)
    #     show_img("lines and circles", tl_draw)
    #
    # cv2.destroyAllWindows()


def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    sign_draw = img_in.copy()
    edges = cv2.Canny(sign_draw, 100, 200)
    #show_img(sign_draw)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 36, threshold=20, minLineLength=5, maxLineGap=5)
    if (len(lines)) == 0:
        return 0, 0
    lines = lines.reshape(lines.shape[0], lines.shape[2])

    # filter lines with 0, +60 and -60 angles
    side_60 = filter_angles(lines, range(55,66,1))
    side__60 = filter_angles(lines, range(-55, -66,-1))

    triangle = []

    """
    find two lines at 60 and -60 angles, which are almost connected - they will be the angular lines
    put the three points found by these two lines in a set. Check the set of triangles to find the center.
    """
    for s60 in side_60:
        s60_1 = [s60[0], s60[1]]
        s60_2 = [s60[2], s60[3]]
        s60_low, s60_up = (s60_1, s60_2) if s60_1[1] > s60_2[1] else (s60_2, s60_1)
        for s_60 in side__60:
            s_60_1 = [s_60[0], s_60[1]]
            s_60_2 = [s_60[2], s_60[3]]
            s_60_low, s_60_up = (s_60_1, s_60_2) if s_60_1[1] > s_60_2[1] else (s_60_2, s_60_1)
            if (abs(s60_low[0] - s_60_low[0]) < 3 and abs(s60_low[1] - s_60_low[1]) < 3):
                mid_pt = [(s60_low[0] + s_60_low[0]) / 2, (s60_low[1] + s_60_low[1]) / 2]
                triangle.append([s60_up, mid_pt, s_60_up])

    if len(triangle) == 0:
      return 0, 0

    x = np.mean([np.mean([s[0] for s in tri]) for tri in triangle])
    y = np.mean([np.mean([s[1] for s in tri]) for tri in triangle])

    return int(x), int(y)


def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    sign_draw = img_in.copy()
    edges = cv2.Canny(sign_draw, 100, 200)
    #show_img("edges of tl", edges)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 36, threshold=20, minLineLength=5, maxLineGap=5)
    if (len(lines)) == 0:
        return 0, 0
    lines = lines.reshape(lines.shape[0], lines.shape[2])

    lengths = [5 * int(get_length(line) / 5) for line in lines]
    #print(sorted(lengths))

    lines = np.array([lines[i] for i in range(len(lines)) if 25 <= lengths[i] <= 40])
    #print(len(lines))

    i = 0
    for line in lines:
        if 40 >= 5 * int(get_length(line) / 5) >= 25 or lengths[i] == 995:
            i += 1
            cv2.line(sign_draw, (line[0], line[1]), (line[2], line[3]), (0, 0, 0), 2)
    #show_img("", sign_draw)

    linesS = filter_angles(lines, [0, 90])
    linesA = filter_angles(lines, [45, -45])
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
                        common = [int((pin1[0] + pin2[0]) / 2) + 2, int((pin1[1] + pin2[1]) / 2) - 3]
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
    if len(octagons) == 0:
        return 0, 0
    for o in octagons:
        if len(o["lines"]) >= 3:
            fo["lines"] = fo["lines"].union(o["lines"])
            fo["common"] = fo["common"].union(o["common"])
    #print(len(fo["lines"]))

    points = list(fo["common"])
    pd = list([0 for i in range(len(fo["common"]))])

    for i in range(len(points)):
        distances = [proximal_pts(points[i], pt, 15) for pt in points]
        distances[i] = False
        for j in range(i + 1, len(distances)):
            if pd[j] == 0 and distances[j]:
                pd[j] = 1
    #print(pd)

    centerx = int(np.mean([points[i][0] for i in range(len(points)) if pd[i] == 0]))
    centery = int(np.mean([points[i][1] for i in range(len(points)) if pd[i] == 0]))

    area = sign_draw[centery - 5:centery + 5, centerx - 5:centerx + 5]
    red = np.mean(area[:, :, 2])
    green = np.mean(area[:, :, 1])
    blue = np.mean(area[:, :, 0])
    if red > 200:
        return centerx, centery
    return 0,0


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    sign_draw = img_in.copy()

    edges = cv2.Canny(sign_draw, 100, 200)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 36, threshold=20, minLineLength=5, maxLineGap=5)
    if (len(lines)) == 0:
        return 0, 0
    lines = lines.reshape(lines.shape[0], lines.shape[2])

    diamonds = get_diamonds(lines)
    if len(diamonds) == 0:
        return 0, 0

    for diamond in diamonds:
        centerx = np.mean([c[0] for c in diamond["common"]])
        centery = np.mean([c[1] for c in diamond["common"]])
        area = sign_draw[int(centery) - 5:int(centery) + 5, int(centerx) - 5:int(centerx) + 5]
        red = np.mean(area[:, :, 2])
        green = np.mean(area[:, :, 1])
        #print("{} {}".format(red, green))
        if red > 200 and green > 200:
            return int(centerx), int(centery)

    return 0, 0


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    sign_draw = img_in.copy()

    edges = cv2.Canny(sign_draw, 100, 200)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 36, threshold=20, minLineLength=5, maxLineGap=5)
    if (len(lines)) == 0:
        return 0, 0
    lines = lines.reshape(lines.shape[0], lines.shape[2])

    diamonds = get_diamonds(lines)
    if len(diamonds) == 0:
        return 0, 0

    for diamond in diamonds:
        centerx = np.mean([c[0] for c in diamond["common"]])
        centery = np.mean([c[1] for c in diamond["common"]])
        area = sign_draw[int(centery) - 5:int(centery) + 5, int(centerx) - 5:int(centerx) + 5]
        red = np.mean(area[:, :, 2])
        green = np.mean(area[:, :, 1])
        #print("{} {}".format(red, green))
        if red > 200 and 100 < green < 200:
            return int(centerx), int(centery)

    return 0, 0


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    sign_draw = img_in.copy()
    edges = cv2.Canny(sign_draw, 100, 200)
    # show_img("edges of tl", edges)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 36, threshold=20, minLineLength=5, maxLineGap=5)
    if (len(lines)) == 0:
        return 0, 0
    lines = lines.reshape(lines.shape[0], lines.shape[2])
    # for line in lines:
    #     cv2.line(sign_draw, (line[0], line[1]), (line[2], line[3]), (0,0,0), 2)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=15, param2=20,
                               minRadius=5, maxRadius=50)
    if circles is None: return 0, 0  # should become return 0, 0

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
            area = sign_draw[c[1] - 5:c[1] + 5, c[0] - 5:c[0] + 5]
            red = np.mean(area[:, :, 2])
            green = np.mean(area[:, :, 1])
            blue = np.mean(area[:, :, 0])
            if red > 200 and blue > 200 and green > 200:
                return c[0], c[1]

    return 0, 0


def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    all_signs = {}
    coordinates = [traffic_light_detection(img_in, range(5,50,1))[0],
                   do_not_enter_sign_detection(img_in),
                   stop_sign_detection(img_in),
                   warning_sign_detection(img_in),
                   yield_sign_detection(img_in),
                   construction_sign_detection(img_in)]
    signs = ["traffic_light", "no_entry","stop","warning","yield","construction"]

    for i in range(len(signs)):
        if coordinates[i] != (0,0) :
            all_signs[signs[i]] = coordinates[i]
    return all_signs


def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    blurred = cv2.GaussianBlur(img_in, (5,5),0)
    denoised = cv2.fastNlMeansDenoisingColored(blurred,None,10,10,7,21)
    show_img("denoised", denoised)
    edges = cv2.Canny(denoised, 100, 200)
    show_img("edges", edges)
    signs = traffic_sign_detection(denoised)
    return signs


def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    blurred = cv2.GaussianBlur(img_in, (15, 15), 0)
    #show_img("blurred", blurred)
    denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21)
    #show_img("denoised", denoised)
    edges = cv2.Canny(denoised, 100, 200)
    show_img("edges", edges)
    signs = traffic_sign_detection(denoised)
    return signs
