"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
from math import pi as PI
import math

import cv2

import numpy as np


# helping functions
def aproxeq(a,b,thresh):
    if a-thresh<=b<=a+thresh:
        return True
    return False

def calc_dist(p1, p2):
    return math.sqrt((p1[0]-p2[0]) ** 2 + (p1[1]-p2[1]) ** 2)

def is_close(p1, p2, dist_thresh):
    return abs(p1[0]-p2[0]) < dist_thresh and abs(p1[1]-p2[1]) < dist_thresh

def line_connected(l1, l2, dist_thresh):
    return is_close(l1[0], l2[0], dist_thresh) or \
           is_close(l1[0], l2[1], dist_thresh) or \
           is_close(l1[1], l2[0], dist_thresh) or \
           is_close(l1[1], l2[1], dist_thresh)

def line_len(line):
    return math.sqrt((line[0][0] - line[1][0]) ** 2 + (line[0][1] - line[1][1]) ** 2)

def line_angle(line):
    x1,y1 = line[0]
    x2,y2 = line[1]
    a = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
    r = int(5 * round(float(a)/5))
    return r

def dist_to_line(line, p):
    x0, y0 = p
    x1, y1 = line[0]
    x2, y2 = line[1]

    return abs( (y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1 ) / line_len(line)

def line_center(line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    return (x1+x2)/2, (y1+y2)/2


def filter_lines(lines, angles=None, min_len=0, max_len=200):
    lines = [l for l in lines if min_len <= line_len(l) <= max_len]
    if angles is not None:
        lines = [l for l in lines if line_angle(l) in angles]
    return lines


def to_gray(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray_image)
    # cv2.waitKey()
    return gray_image

def preprocess(img):
    gray = to_gray(img)
    blured = cv2.GaussianBlur(gray, (5, 5), 2)
    return blured


def coords_to_points(coords):
    return ((coords[0], coords[1]), (coords[2], coords[3]))


def find_edges(img, minGrad=5, maxGrad=10):
    edges = cv2.Canny(img, minGrad, maxGrad)
    # cv2.imshow('edges', edges)
    # cv2.waitKey()
    return edges


# def cut_lines(lines, min_cut_len=200, approx_thresh=5, min_line_len=15):
#     big_lines = [line for line in lines if line_len(line)>=min_cut_len]
#     points = []
#     for l in lines:
#         points.extend(l)
#
#     new_lines = []
#
#     for bl in big_lines:
#         cut_points = [p for p in points if dist_to_line(bl, p) <= approx_thresh and p not in bl]
#         sort_func = lambda p: calc_dist(p, bl[0])
#         cut_points.sort(key=sort_func)
#
#         cut_points.append(bl[1])
#         prev_point = bl[0]
#         for p in cut_points:
#             new_line = (prev_point, p)
#             if line_len(new_line) > min_line_len:
#                 new_lines.append(new_line)
#             prev_point = p
#
#     return new_lines


def find_lines(img):
    gray_blur_img = preprocess(img)
    edges = find_edges(gray_blur_img)
    lines_coords = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=PI / 36,
        threshold=20,
        minLineLength=5,
        maxLineGap=5)
    if lines_coords is None:
        return []
    lines_coords = lines_coords[0]

    lines = map(coords_to_points, lines_coords)

    # lines.extend(cut_lines(lines))

    return lines


# def find_polygons(img, n_sides=4, prox_thresh=5):
#     poly_complete = lambda poly: is_close(poly[0][0], poly[-1][1], prox_thresh)
#
#     raw_lines = find_lines(img)
#     lines = set(raw_lines)
#     polygons = []
#     while len(lines) > 0:
#         candidate = [lines.pop()]
#         changed = True
#
#         while changed:
#             next_lines = set(lines)
#             if poly_complete(candidate):
#                 for l in candidate:
#                     lines.remove(l)
#                 polygons.append(candidate)
#                 break
#
#             changed = False
#
#             # check if end can be connected
#             final_point = candidate[-1][1]
#             for next_line in next_lines:
#                 if is_close(final_point, next_line[0], prox_thresh):
#                     candidate.append(next_line)                     # add line to polygon
#                     next_lines.remove(next_line)                         # don't use this line anymore
#                     changed = True
#                     break
#                 if is_close(final_point, next_line[1], prox_thresh):
#                     candidate.append((next_line[1], next_line[0]))  # change line direction and add to polygon
#                     next_lines.remove(next_line)  # don't use this line anymore
#                     changed = True
#                     break
#             # check if beginning can be connected
#             first_point = candidate[0][0]
#             for next_line in next_lines:
#                 if is_close(first_point, next_line[0], prox_thresh):
#                     candidate = [(next_line[1], next_line[0])] + candidate  # change line direction and add to polygon
#                     next_lines.remove(next_line)  # don't use this line anymore
#                     changed = True
#                     break
#                 if is_close(first_point, next_line[1], prox_thresh):
#                     candidate = [next_line] + candidate
#                     next_lines.remove(next_line)  # don't use this line anymore
#                     changed = True
#                     break
#         # try to complete
#         if len(candidate) == n_sides-1:
#             p1 = candidate[0][0]
#             p2 = candidate[-1][1]
#             for l in raw_lines:
#                 if dist_to_line(l,p1) < prox_thresh and dist_to_line(l,p2) < prox_thresh:
#                     for l in candidate:
#                         lines.remove(l)
#                     candidate.append((p2, p1))
#                     polygons.append(candidate)
#                     break
#
#     # filter polygons with required number of sides
#     if n_sides is not None:
#         polygons = [poly for poly in polygons if len(poly) == n_sides]
#
#     return polygons


# def collapse_lines(lines):
#     polyline = []
#     polyline.append(lines[0][0])
#     for i in range(len(lines)-1):
#         inter_point = ( (lines[i][1][0] + lines[i+1][0][0]) / 2,
#                         (lines[i][1][1] + lines[i+1][0][1]) / 2 )
#         polyline.append(inter_point)
#     return polyline


def find_circles(img, minRadius, maxRadius, thresh=20, hough_bound=15):
    gray_blur_img = preprocess(img)

    # edges = find_edges(gray_img)
    circles = cv2.HoughCircles(
        gray_blur_img,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=max(15, 2 * minRadius - 10),
        param1=hough_bound,
        param2=thresh,
        minRadius=minRadius,
        maxRadius=maxRadius
    )
    if circles is None:
        return []
    circles = circles[0]

    circles_tuples = []
    for circle in circles:
        circles_tuples.append( (circle[0], circle[1], circle[2]) )

    return circles_tuples

def traffic_light_detection_wrapper(img_in, radii_range, thresh=20, hough_bound=15):
    img = img_in.copy()

    minRadius = min(radii_range)
    maxRadius = max(radii_range)
    circles = find_circles(img, max(3, minRadius - 5), maxRadius + 5, thresh, hough_bound)

    lights = []
    for circle in circles:
        other_circles = set(circles)
        other_circles.remove(circle)
        other_circles = {c for c in other_circles if circle[2] - 5 <= c[2] <= circle[2] + 5}
        if len(other_circles) < 2:
            continue
        other_circles = {c for c in other_circles if circle[0] - 5 <= c[0] <= circle[0] + 5}
        if len(other_circles) < 2:
            continue
        if len(other_circles) == 2:
            lights.append(circle)
            lights.extend(other_circles)
            break

    if len(lights) == 0:
        return (0, 0), 'green'

    lights.sort(key=lambda cir: cir[1])
    lights = [(int(l[0]), int(l[1]), int(l[2])) for l in lights]

    R = lights[0][2]

    red = np.mean(img_in[lights[0][1] - R / 2:lights[0][1] + R / 2, lights[0][0] - R / 2:lights[0][0] + R / 2, 2])
    yellow = np.mean(img_in[lights[1][1] - R / 2:lights[1][1] + R / 2, lights[1][0] - R / 2:lights[1][0] + R / 2, 1:])
    green = np.mean(img_in[lights[2][1] - R / 2:lights[2][1] + R / 2, lights[2][0] - R / 2:lights[2][0] + R / 2, 1])

    color = ['red', 'yellow', 'green'][np.argmax([red, yellow, green])]

    return (lights[1][0], lights[1][1]), color


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
    return traffic_light_detection_wrapper(img_in, radii_range, thresh=20, hough_bound=15)


def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    img = img_in.copy()
    lines = find_lines(img)
    lines = filter_lines(lines, angles=[0, 60, -60], min_len=10, max_len=300)

    side1_lines = filter_lines(lines, angles=[60])
    side2_lines = filter_lines(lines, angles=[-60])
    horiz_lines = filter_lines(lines, angles=[0])

    prox_thresh = 3

    tris = []

    for s1 in side1_lines:
        s1_low_p, s1_up_p = (s1[0], s1[1]) if s1[0][1] > s1[1][1] else (s1[1], s1[0])
        for s2 in side2_lines:
            s2_low_p, s2_up_p = (s2[0], s2[1]) if s2[0][1] > s2[1][1] else (s2[1], s2[0])
            if is_close(s1_low_p, s2_low_p, prox_thresh):
                tris.append([s1_up_p, ((s1_low_p[0]+s2_low_p[0])/2, (s1_low_p[1]+s2_low_p[1])/2), s2_up_p])

    if len(tris) == 0:
        return 0, 0

    center_x = np.mean([s1_low_p[0] for _, s1_low_p, _ in tris])
    center_y = np.mean([(s1_low_p[1] + s1_up_p[1] + s2_up_p[1]) / 3 for s1_up_p, s1_low_p, s2_up_p in tris])

    return int(center_x), int(center_y)

def stop_sign_detection_wrapper(img_in, prox_thresh=15, size_thresh=30):
    img = img_in.copy()
    lines = find_lines(img)
    lines = filter_lines(lines, angles=[0, 45, -45, 90, -90], min_len=5, max_len=400)

    angle1_lines = filter_lines(lines, angles=[45])
    angle2_lines = filter_lines(lines, angles=[-45])
    horiz_lines = filter_lines(lines, angles=[0])
    vert_lines = filter_lines(lines, angles=[90, -90])

    angle_sets = []
    for a1 in angle1_lines:
        a1_c = line_center(a1)
        for a2 in angle2_lines:
            a2_c = line_center(a2)

            if aproxeq(line_len(a1), line_len(a2), size_thresh) and \
                    (aproxeq(a1_c[0], a2_c[0], prox_thresh) or aproxeq(a1_c[1], a2_c[1], prox_thresh)) and \
                    not line_connected(a1, a2, prox_thresh):
                in_set = False
                for s in angle_sets:
                    if a1 in s or a2 in s:
                        s.add(a1)
                        s.add(a2)
                        in_set = True
                if not in_set:
                    angle_sets.append(set([a1, a2]))

    angle_sets = [s for s in angle_sets if len(s) > 2]

    for s in angle_sets:
        avglen = np.mean([line_len(l) for l in s])
        points = []
        for l in s:
            points.append(l[0])
            points.append(l[1])
        for hl in horiz_lines:
            if not aproxeq(line_len(hl), avglen, size_thresh):
                continue
            for p1 in points:
                if is_close(hl[0], p1, prox_thresh):
                    for p2 in points:
                        if is_close(hl[1], p2, prox_thresh):
                            s.add(hl)
        for vl in vert_lines:
            if not aproxeq(line_len(vl), avglen, size_thresh):
                continue
            for p1 in points:
                if is_close(vl[0], p1, prox_thresh):
                    for p2 in points:
                        if is_close(vl[1], p2, prox_thresh):
                            s.add(vl)

    # find biggest set
    maxlen = 0
    max_s = None
    for s in angle_sets:
        if len(s) > maxlen:
            maxlen = len(s)
            max_s = s

    if max_s is None:
        return 0, 0

    lines = list(max_s)
    angle1_lines = filter_lines(lines, angles=[45])
    angle2_lines = filter_lines(lines, angles=[-45])
    horiz_lines = filter_lines(lines, angles=[0])
    vert_lines = filter_lines(lines, angles=[90, -90])
    groups = [angle1_lines, angle2_lines, horiz_lines, vert_lines]

    def calc_center(lines):
        m = 0
        l1_m = None
        l2_m = None
        for l1 in lines:
            for l2 in lines:
                if l1 == l2:
                    continue
                dist = calc_dist(line_center(l1), line_center(l2))
                if m < dist:
                    m = dist
                    l1_m = l1
                    l2_m = l2
        if l1_m is None:
            return None
        lc1 = line_center(l1_m)
        lc2 = line_center(l2_m)
        return (lc1[0] + lc2[0]) / 2, (lc1[1] + lc2[1]) / 2

    centers = [calc_center(g) for g in groups]
    centers = [c for c in centers if c is not None]

    center = np.mean(centers, axis=0)

    return int(center[0]), int(center[1])

def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    return stop_sign_detection_wrapper(img_in)


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    img = img_in.copy()
    lines = find_lines(img)
    lines = filter_lines(lines, angles=[45, -45], min_len=10, max_len=300)

    angle1_lines = filter_lines(lines, angles=[45])
    angle2_lines = filter_lines(lines, angles=[-45])

    prox_thresh = 10
    size_thresh = 10

    angle_sets = []
    for a1 in angle1_lines:
        a1_c = line_center(a1)
        for a2 in angle2_lines:
            a2_c = line_center(a2)

            if aproxeq(line_len(a1), line_len(a2), size_thresh) and \
                    (aproxeq(a1_c[0], a2_c[0], prox_thresh) or aproxeq(a1_c[1], a2_c[1], prox_thresh)) and \
                    line_connected(a1, a2, prox_thresh):
                in_set = False
                for s in angle_sets:
                    if a1 in s or a2 in s:
                        s.add(a1)
                        s.add(a2)
                        in_set = True
                if not in_set:
                    angle_sets.append(set([a1, a2]))

    angle_sets = [s for s in angle_sets if len(s) > 2]

    centers = []

    for s in angle_sets:
        lines = list(s)
        angle1_lines = filter_lines(lines, angles=[45])
        angle2_lines = filter_lines(lines, angles=[-45])
        groups = [angle1_lines, angle2_lines]

        def calc_center(lines):
            m = 0
            l1_m = None
            l2_m = None
            for l1 in lines:
                for l2 in lines:
                    if l1 == l2:
                        continue
                    dist = calc_dist(line_center(l1), line_center(l2))
                    if m < dist:
                        m = dist
                        l1_m = l1
                        l2_m = l2
            if l1_m is None:
                return None
            lc1 = line_center(l1_m)
            lc2 = line_center(l2_m)
            return (lc1[0] + lc2[0]) / 2, (lc1[1] + lc2[1]) / 2

        group_centers = [calc_center(g) for g in groups]
        group_centers = [c for c in group_centers if c is not None]

        center = np.mean(group_centers, axis=0)
        centers.append((int(center[0]), int(center[1])))

    for center in centers:
        center_area = img_in[center[1]-5:center[1]+5, center[0]-5:center[0]+5, :]
        r = np.mean(center_area[:,:,2])
        g = np.mean(center_area[:,:,1])
        b = np.mean(center_area[:,:,0])

        if r > 200 and g > 200:
            return int(center[0]), int(center[1])

    return 0, 0


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    img = img_in.copy()
    lines = find_lines(img)
    lines = filter_lines(lines, angles=[45, -45], min_len=10, max_len=300)

    angle1_lines = filter_lines(lines, angles=[45])
    angle2_lines = filter_lines(lines, angles=[-45])

    prox_thresh = 10
    size_thresh = 10

    angle_sets = []
    for a1 in angle1_lines:
        a1_c = line_center(a1)
        for a2 in angle2_lines:
            a2_c = line_center(a2)

            if aproxeq(line_len(a1), line_len(a2), size_thresh) and \
                    (aproxeq(a1_c[0], a2_c[0], prox_thresh) or aproxeq(a1_c[1], a2_c[1], prox_thresh)) and \
                    line_connected(a1, a2, prox_thresh):
                in_set = False
                for s in angle_sets:
                    if a1 in s or a2 in s:
                        s.add(a1)
                        s.add(a2)
                        in_set = True
                if not in_set:
                    angle_sets.append(set([a1, a2]))

    angle_sets = [s for s in angle_sets if len(s) > 2]

    centers = []

    for s in angle_sets:
        lines = list(s)
        angle1_lines = filter_lines(lines, angles=[45])
        angle2_lines = filter_lines(lines, angles=[-45])
        groups = [angle1_lines, angle2_lines]

        def calc_center(lines):
            m = 0
            l1_m = None
            l2_m = None
            for l1 in lines:
                for l2 in lines:
                    if l1 == l2:
                        continue
                    dist = calc_dist(line_center(l1), line_center(l2))
                    if m < dist:
                        m = dist
                        l1_m = l1
                        l2_m = l2
            if l1_m is None:
                return None
            lc1 = line_center(l1_m)
            lc2 = line_center(l2_m)
            return (lc1[0] + lc2[0]) / 2, (lc1[1] + lc2[1]) / 2

        group_centers = [calc_center(g) for g in groups]
        group_centers = [c for c in group_centers if c is not None]

        center = np.mean(group_centers, axis=0)
        centers.append((int(center[0]), int(center[1])))

    for center in centers:
        center_area = img_in[center[1]-5:center[1]+5, center[0]-5:center[0]+5, :]
        r = np.mean(center_area[:,:,2])
        g = np.mean(center_area[:,:,1])
        b = np.mean(center_area[:,:,0])

        if r > 200 and 100 < g < 200:
            return int(center[0]), int(center[1])

    return 0, 0


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    img = img_in.copy()

    minRadius = 10
    maxRadius = 50
    circles = find_circles(img, max(3, minRadius - 5), maxRadius + 5)

    for circle in circles:
        r = int(circle[2])
        center_x = int(circle[0])
        center_y = int(circle[1])

        brick_h = r/3

        center_area = img_in[center_y-brick_h/2+2:center_y+brick_h/2-2, center_x-brick_h:center_x+brick_h,:]
        mcolors = np.mean(center_area, axis=(0,1))

        if mcolors[0] > 200 and mcolors[1] > 200 and mcolors[2] > 200:
            area_above_center = img_in[center_y-brick_h:center_y-brick_h/2-2, center_x-brick_h:center_x+brick_h,:]
            mcolors = np.mean(area_above_center, axis=(0, 1))
            if mcolors[0] < 100 and mcolors[1] < 100 and mcolors[2] > 200:
                return center_x, center_y

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
    tl, _ = traffic_light_detection(img_in, range(10, 30, 1))
    no_entry = do_not_enter_sign_detection(img_in)
    stop = stop_sign_detection(img_in)
    warning = warning_sign_detection(img_in)
    yld = yield_sign_detection(img_in)
    constr = construction_sign_detection(img_in)

    signs = {}

    if tl[0] > 0 and tl[1] > 0:
        signs['traffic_light'] = tl
    if no_entry[0] > 0 and no_entry[1] > 0:
        signs['no_entry'] = no_entry
    if stop[0] > 0 and stop[1] > 0:
        signs['stop'] = stop
    if warning[0] > 0 and warning[1] > 0:
        signs['warning'] = warning
    if yld[0] > 0 and yld[1] > 0:
        signs['yield'] = yld
    if constr[0] > 0 and constr[1] > 0:
        signs['construction'] = constr

    return signs


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
    img = img_in.copy()

    img = cv2.bilateralFilter(img, 25,100,100)
    img = cv2.bilateralFilter(img, 19, 50, 50)

    # cv2.imshow('noise', img)
    # cv2.waitKey()

    tl, _ = traffic_light_detection_wrapper(img, range(10, 30, 1), thresh=15, hough_bound=7)
    # print("tl", tl)
    no_entry = do_not_enter_sign_detection(img)
    # print("no_entry", no_entry)
    warning = warning_sign_detection(img)
    # print("warning", warning)
    yld = yield_sign_detection(img)
    # print("yield", yld)
    constr = construction_sign_detection(img)
    # print("constr", constr)
    # stop=(0,0)
    stop = stop_sign_detection_wrapper(img, prox_thresh=10, size_thresh=10)
    # print("stop", stop)


    signs = {}

    if tl[0] > 0 and tl[1] > 0:
        signs['traffic_light'] = tl
    if no_entry[0] > 0 and no_entry[1] > 0:
        signs['no_entry'] = no_entry
    if stop[0] > 0 and stop[1] > 0:
        signs['stop'] = stop
    if warning[0] > 0 and warning[1] > 0:
        signs['warning'] = warning
    if yld[0] > 0 and yld[1] > 0:
        signs['yield'] = yld
    if constr[0] > 0 and constr[1] > 0:
        signs['construction'] = constr

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
    return traffic_sign_detection_noisy(img_in)