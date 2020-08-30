"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2

import numpy as np

import math

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
    centerxy = cr[0:2, cr[1, :] == center].reshape(2)
    return centerxy


def get_light_color(img, centers):
    print(centers)
    r = (centers[:, 2] / 4).astype(int)
    red = np.mean(img[centers[0][1] - r[0]:centers[0][1] + r[0], centers[0][0] - r[0]:centers[0][0] + r[0], 2])
    yellow = np.mean(img[centers[1][1] - r[1]:centers[1][1] + r[1], centers[1][0] - r[1]:centers[1][0] + r[1], 1:])
    green = np.mean(
        img[centers[2][1] - r[2]:centers[2][1] + r[2], centers[2][0] - r[2]:centers[2][0] + r[2], 1])
    print("{}  {}  {}".format(red, yellow, green))
    #
    colors = ["red", "yellow", "green"]
    color = colors[np.argmax([red, yellow, green])]
    return color


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


def get_midpt(line):
    print(line)
    x1,y1 = line[0],line[1]
    x2, y2 = line[2], line[3]
    return [(x1+x2)/2,(y1+y2)/2]

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

    centers = []

    for square in squares:
        """
        get lines of same angle, get their mid points and the midpoint of these midpoints
        from the two groups, find mid-mid point
        """
        # lines = [np.array(line) for line in list(square)]
        # side45 = filter_angles(lines, [45])
        # side_45 = filter_angles(lines, [-45])
        # # c = np.array([get_centers(side45), get_centers(side_45)])
        # center = np.mean(c, axis=1)
        # centers.append(center)

        d = []
        for i in range(4):
            d.append(np.mean([c[i] for c in list(square)]))
        return ((d[0]+d[2])/2,(d[1]+d[3])/2)
    #return centers

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
    # show_img("edges of image", edges)

    # get all lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    # print all lines
    # draw_lines(lines, tl_draw)
    # show_img("lines on tl", tl_draw)

    # get x axis of the vertical lines
    xmin, xmax = get_vertical(lines)
    print(radii_range)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=15, param2=20,
                               minRadius=max(5, min(radii_range) - 5), maxRadius=min(50, max(radii_range) + 10))

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # draw_circles(circles, tl)
        # show_img("lines and circles", tl)

        cshape = circles.shape
        centers = circles.reshape(cshape[1], cshape[2])

        # get the circles between the two vertical lines
        centers = centers[(centers[:, 0] > xmin) * (centers[:, 0] < xmax), :]
        centers = centers[centers[:, 1].argsort()]

        # get the color which is on
        color = get_light_color(tl, centers)

        center_tl = get_center(circles, xmin, xmax)
        return (center_tl[0], center_tl[1]), color
    else:
        return (0, 0), ""
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

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 36, threshold=20, minLineLength=5, maxLineGap=5)
    lines = lines.reshape(lines.shape[0], lines.shape[2])

    # filter lines with 0, +60 and -60 angles
    side_60 = filter_angles(lines, [60])
    side__60 = filter_angles(lines, [-60])

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
    raise NotImplementedError


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    sign_draw = img_in.copy()
    # show_img("sample tl", tl)

    # gray_img = cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY)
    # blured = cv2.GaussianBlur(gray_img, (5,5),2)
    # edges_blur = cv2.Canny(blured, 5, 10)
    # show_img("blurred edges", edges_blur)

    # check how canny edge filter shows up)
    edges = cv2.Canny(sign_draw, 100, 200)
    # show_img("edges of tl", edges)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 36, threshold=20, minLineLength=5, maxLineGap=5)
    lines = lines.reshape(lines.shape[0], lines.shape[2])

    centers = get_square_centers(lines)

    for center in centers:
        area = sign_draw[int(center[0]) - 5:int(center[0]) + 5, int(center[1]) - 5:int(center[1]) + 5]
        red = np.mean(area[:, :, 2])
        green = np.mean(area[:, :, 1])

        if red > 200 and green > 200:
            return int(center[0]), int(center[1])
    return int(center[0]), int(center[1])


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    raise NotImplementedError


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
