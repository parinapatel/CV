"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2

import numpy as np

import math


def magnitude(v1, v2):
    x = v2[0] - v1[0]
    y = v2[1] - v1[1]
    return math.sqrt((x ** 2 + y ** 2))


def isCloseColour(colour, expectedColour, tolerance):
    return (np.isclose(colour[0], expectedColour[0], atol=tolerance) and
            np.isclose(colour[1], expectedColour[1], atol=tolerance) and
            np.isclose(colour[2], expectedColour[2], atol=tolerance))


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

    # remove circles not in the traffic light vertical
    def filterCircles(circles, r):
        ranges = {}

        for i in circles[0, :]:
            if (i[0] in ranges):
                ranges[j].append(i)
            else:

                for j in range(i[0] - r, i[0] + r, 1):
                    if (j in ranges):
                        ranges[j].append(i)
                    else:
                        ranges[j] = []
                        ranges[j].append(i)

        for key in ranges:
            if len(ranges[key]) == 3:
                return ranges[key]

        return []

    img = cv2.medianBlur(img_in, 5)
    gray_img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 0.75, 10, param1=20, param2=10,
                               minRadius=radii_range[0], maxRadius=radii_range[0] + len(radii_range))
    # print circles
    result = ()
    x = 0
    y = 0
    traffic_light_colour = ''
    tf_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        tf_circles = filterCircles(circles, 5)

        if tf_circles != []:
            for i in tf_circles:
                # get colour
                colour = img_in[i[1]][i[0]]
                if (colour[1] == 255 and colour[2] != 255):
                    traffic_light_colour = "green"

                elif (colour[1] == 255 and colour[2] == 255):
                    traffic_light_colour = "yellow"

                elif (colour[2] == 255):
                    traffic_light_colour = "red"

                # print i
                # print img_in[i[1]][i[0]]
                # draw the outer circle
                # draw the center of the circle
                # cv2.circle(img_in,(i[0],i[1]),i[2],(0,255,0),2)
                # cv2.circle(img_in,(i[0],i[1]),2,(0,0,255),3)

                y = np.uint16(np.median([s[1] for s in tf_circles]))
                x = np.uint16(np.median([s[0] for s in tf_circles]))
                result = (x, y)
                # cv2.circle(img_in,(result[0],result[1]),2,(0,0,0),3)
        # print traffic_light_colour

    return (result, traffic_light_colour)


def yield_sign_detection(img_in):
    # works for non white background ----
    # img = cv2.medianBlur(img_in,11)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray,25,100,apertureSize = 3)
    # lines = cv2.HoughLines(edges,4,3*(np.pi/180),90)

    # for autograder
    # img = cv2.medianBlur(img_in,11)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray,25,100,apertureSize = 3)
    # lines = cv2.HoughLines(edges,2.25,1.5*(np.pi/180),45)

    img = cv2.medianBlur(img_in, 11)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 450, 500, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, 3 * (np.pi / 180), 40)

    result = ()
    # print lines
    angle_map = {"30": [], "90": [], "150": []}
    # angle_map = {90: [(274.0, 1.5707964)], 150: [(-62.0, 2.6179938)], 30: [(434.0, 0.52359879)]}
    if lines is not None:

        for rho, theta in lines[0]:
            t = (theta * (180 / np.pi))
            if (np.isclose(30, t, atol=1e-1)):
                angle_map["30"].append((rho, theta))
            elif (np.isclose(90, t, atol=1e-1)):
                angle_map["90"].append((rho, theta))
            elif (np.isclose(150, t, atol=1e-1)):
                angle_map["150"].append((rho, theta))
            # print (theta*(180/np.pi))

            # a,b = np.cos(theta), np.sin(theta)
            # x0,y0 = a*rho, b*rho
            # x1,y1= int(x0 + 1000*(-b)),int(y0 + 1000*(a))
            # x2,y2 = int(x0 - 1000*(-b)),int(y0 - 1000*(a))

            # cv2.line(img_in,(x1,y1),(x2,y2),(0,255,0),2)

            # loop through all lines with 30,90,150 deg and find vertices
        # raise Exception(angle_map)
        for deg30 in angle_map["30"]:
            for deg90 in angle_map["90"]:
                for deg150 in angle_map["150"]:
                    # raise Exception("%s %s %s!" % (deg30,deg90,deg150))
                    # get vertices of lines
                    a = [[np.cos(deg30[1]), np.sin(deg30[1])], [np.cos(deg90[1]), np.sin(deg90[1])]]
                    b = [deg30[0], deg90[0]]
                    deg30deg90 = np.linalg.solve(a, b)

                    a = [[np.cos(deg30[1]), np.sin(deg30[1])], [np.cos(deg150[1]), np.sin(deg150[1])]]
                    b = [deg30[0], deg150[0]]
                    deg30deg150 = np.linalg.solve(a, b)

                    a = [[np.cos(deg90[1]), np.sin(deg90[1])], [np.cos(deg150[1]), np.sin(deg150[1])]]
                    b = [deg90[0], deg150[0]]
                    deg90deg150 = np.linalg.solve(a, b)

                    verticies = (deg30deg90, deg30deg150, deg90deg150)
                    # print verticies
                    # find centroid
                    centroid = np.uint16(((verticies[0][0] + verticies[1][0] + verticies[2][0]) / 3,
                                          (verticies[0][1] + verticies[1][1] + verticies[2][1]) / 3))

                    redspot = (centroid[0], np.uint16((deg90deg150[1] + deg30deg90[1]) / 2) + 5)
                    # print centroid
                    # check if white, then probably yield sign
                    # Also check if right orientation ("tip" is pointing down
                    # print np.array_equal(img_in[centroid[1],centroid[0]],[255,255,255])
                    # check if pixel right below 90deg line is red
                    # raise Exception("%s %s" % (angle_map, centroid))
                    # print magnitude(deg30deg90,deg30deg150)
                    # print img_in[centroid[1],centroid[0]]
                    if (isCloseColour(img_in[centroid[1], centroid[0]], [255, 255, 255], 10) and
                            np.isclose(magnitude(deg30deg90, deg30deg150), 83, atol=5)):
                        # and deg30deg150[1] > centroid[1]
                        # and np.array_equal(img_in[redspot[1],redspot[0]],[0,0,255])):
                        result = (centroid[0], centroid[1])
                        cv2.circle(img_in, (result[0], result[1]), 2, (0, 0, 255), 3)

    # print result
    # cv2.circle(img_in,(result[0],result[1]),2,(0,0,255),3)
    return result


def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """

    # Works for white/autograder
    # img = cv2.medianBlur(img_in,11)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray,25,100,apertureSize = 3)
    # lines = cv2.HoughLines(edges,1.1,3*(np.pi/180),13)

    # part 3
    # img = cv2.medianBlur(img_in,11)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray,300,350,apertureSize = 3)
    # lines = cv2.HoughLines(edges,1.08,15*(np.pi/180),25)

    img = cv2.medianBlur(img_in, 11)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 300, 350, apertureSize=3)
    lines = cv2.HoughLines(edges, 1.08, 15 * (np.pi / 180), 25)

    result = ()
    # print lines
    angle_map = {"45": [], "135": []}
    # angle_map = {90: [(274.0, 1.5707964)], 150: [(-62.0, 2.6179938)], 30: [(434.0, 0.52359879)]}
    if lines is not None:

        for rho, theta in lines[0]:
            t = (theta * (180 / np.pi))
            if (np.isclose(45, t, atol=1e-1)):
                angle_map["45"].append((rho, theta))
            elif (np.isclose(135, t, atol=1e-1)):
                angle_map["135"].append((rho, theta))
            # print (theta*(180/np.pi))

            # a,b = np.cos(theta), np.sin(theta)
            # x0,y0 = a*rho, b*rho
            # x1,y1= int(x0 + 1000*(-b)),int(y0 + 1000*(a))
            # x2,y2 = int(x0 - 1000*(-b)),int(y0 - 1000*(a))

            # cv2.line(img_in,(x1,y1),(x2,y2),(0,255,0),2)

        # raise Exception(angle_map)
        for line1 in angle_map["45"]:
            for line2 in angle_map["135"]:
                for line3 in angle_map["45"]:
                    for line4 in angle_map["135"]:
                        if (line1 == line3 or line2 == line4):
                            continue

                        # raise Exception("%s %s %s!" % (deg30,deg90,deg150))
                        # get vertices of lines
                        a = [[np.cos(line1[1]), np.sin(line1[1])], [np.cos(line2[1]), np.sin(line2[1])]]
                        b = [line1[0], line2[0]]
                        line12 = np.linalg.solve(a, b)
                        # cv2.circle(img_in,(line12[0],line12[1]),2,(0,0,0),3)

                        a = [[np.cos(line1[1]), np.sin(line1[1])], [np.cos(line4[1]), np.sin(line4[1])]]
                        b = [line1[0], line4[0]]
                        line14 = np.linalg.solve(a, b)
                        # cv2.circle(img_in,(line14[0],line14[1]),2,(0,0,0),3)

                        a = [[np.cos(line2[1]), np.sin(line2[1])], [np.cos(line3[1]), np.sin(line3[1])]]
                        b = [line2[0], line3[0]]
                        line23 = np.linalg.solve(a, b)
                        # cv2.circle(img_in,(line23[0],line23[1]),2,(0,0,0),3)

                        a = [[np.cos(line3[1]), np.sin(line3[1])], [np.cos(line4[1]), np.sin(line4[1])]]
                        b = [line3[0], line4[0]]
                        line34 = np.linalg.solve(a, b)
                        # cv2.circle(img_in,(line34[0],line34[1]),2,(0,0,0),3)

                        verticies = (line12, line14, line23, line34)
                        # print verticies
                        # find centroid
                        centroid = np.uint16(
                            ((verticies[0][0] + verticies[1][0] + verticies[2][0] + verticies[3][0]) / 4,
                             (verticies[0][1] + verticies[1][1] + verticies[2][1] + verticies[3][1]) / 4))

                        # redspot = (centroid[0],np.uint16((deg90deg150[1]+deg30deg90[1])/2)+5)
                        # print centroid

                        # print np.array_equal(img_in[centroid[1],centroid[0]],[255,255,255])

                        # raise Exception("%s %s" % (angle_map, centroid))
                        # print img_in[centroid[1],centroid[0]]
                        # (img[centroid[1],centroid[0],2] >= 204 and img_in[centroid[1],centroid[0],2] <= 255)
                        if (np.isclose(magnitude(line12, line34), 140, atol=5)):
                            if (isCloseColour(img[centroid[1], centroid[0]], [0, 0, 204], 15) or
                                    isCloseColour(img[centroid[1], centroid[0]], [255, 255, 255], 15)):
                                result = (centroid[0], centroid[1])
                                # cv2.circle(img_in,(result[0],result[1]),2,(0,0,0),3)

    # cv2.circle(img_in,(result[0],result[1]),2,(0,0,0),3)
    return result


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    # for auto grader
    # img = cv2.medianBlur(img_in,11)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray,100,125,apertureSize = 3)
    # lines = cv2.HoughLines(edges,0.9,3*(np.pi/180),25)

    img = cv2.medianBlur(img_in, 11)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 125, apertureSize=3)
    lines = cv2.HoughLines(edges, 0.9, 3 * (np.pi / 180), 35)

    result = ()
    # print lines
    angle_map = {"45": [], "135": []}
    # angle_map = {90: [(274.0, 1.5707964)], 150: [(-62.0, 2.6179938)], 30: [(434.0, 0.52359879)]}
    if lines is not None:

        for rho, theta in lines[0]:
            t = (theta * (180 / np.pi))
            if (np.isclose(45, t, atol=1e-1)):
                angle_map["45"].append((rho, theta))
            elif (np.isclose(135, t, atol=1e-1)):
                angle_map["135"].append((rho, theta))
            # print (theta*(180/np.pi))

            # a,b = np.cos(theta), np.sin(theta)
            # x0,y0 = a*rho, b*rho
            # x1,y1= int(x0 + 1000*(-b)),int(y0 + 1000*(a))
            # x2,y2 = int(x0 - 1000*(-b)),int(y0 - 1000*(a))

            # cv2.line(img_in,(x1,y1),(x2,y2),(0,255,0),2)

        # raise Exception(angle_map)
        for line1 in angle_map["45"]:
            for line2 in angle_map["135"]:
                for line3 in angle_map["45"]:
                    for line4 in angle_map["135"]:
                        if (line1 == line3 or line2 == line4):
                            continue

                        # raise Exception("%s %s %s!" % (deg30,deg90,deg150))
                        # get vertices of lines
                        a = [[np.cos(line1[1]), np.sin(line1[1])], [np.cos(line2[1]), np.sin(line2[1])]]
                        b = [line1[0], line2[0]]
                        line12 = np.linalg.solve(a, b)
                        # cv2.circle(img_in,(line12[0],line12[1]),2,(0,0,0),3)

                        a = [[np.cos(line1[1]), np.sin(line1[1])], [np.cos(line4[1]), np.sin(line4[1])]]
                        b = [line1[0], line4[0]]
                        line14 = np.linalg.solve(a, b)
                        # cv2.circle(img_in,(line14[0],line14[1]),2,(0,0,0),3)

                        a = [[np.cos(line2[1]), np.sin(line2[1])], [np.cos(line3[1]), np.sin(line3[1])]]
                        b = [line2[0], line3[0]]
                        line23 = np.linalg.solve(a, b)
                        # cv2.circle(img_in,(line23[0],line23[1]),2,(0,0,0),3)

                        a = [[np.cos(line3[1]), np.sin(line3[1])], [np.cos(line4[1]), np.sin(line4[1])]]
                        b = [line3[0], line4[0]]
                        line34 = np.linalg.solve(a, b)
                        # cv2.circle(img_in,(line34[0],line34[1]),2,(0,0,0),3)

                        verticies = (line12, line14, line23, line34)
                        # find centroid
                        centroid = np.uint16(
                            ((verticies[0][0] + verticies[1][0] + verticies[2][0] + verticies[3][0]) / 4,
                             (verticies[0][1] + verticies[1][1] + verticies[2][1] + verticies[3][1]) / 4))

                        # redspot = (centroid[0],np.uint16((deg90deg150[1]+deg30deg90[1])/2)+5)
                        # print centroid
                        # check if white, then probably yield sign
                        # Also check if right orientation ("tip" is pointing down
                        # print np.array_equal(img_in[centroid[1],centroid[0]],[255,255,255])
                        # check if pixel right below 90deg line is red
                        # raise Exception("%s %s" % (angle_map, centroid))
                        # print img_in[centroid[1],centroid[0]]
                        if (centroid[0] in range(0, img_in.shape[1], 1) and centroid[1] in range(0, img_in.shape[0],
                                                                                                 1)):
                            if (isCloseColour(img[centroid[1], centroid[0]], [0, 255, 255], 15) and
                                    np.isclose(100, magnitude(line12, line34), atol=5)):
                                result = (centroid[0], centroid[1])
                                # cv2.circle(img_in,(result[0],result[1]),2,(0,0,0),3)

    # cv2.circle(img_in,(result[0],result[1]),2,(0,0,0),3)
    return result


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    # Params for tester
    # img = cv2.medianBlur(img_in,11)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray,400,450,apertureSize = 3)
    # lines = cv2.HoughLines(edges,1,3*(np.pi/180),15)
    img = cv2.medianBlur(img_in, 11)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 40, apertureSize=3)
    lines = cv2.HoughLines(edges, 0.5, 1 * (np.pi / 180), 35)

    result = ()
    # print lines
    angle_map = {"45": [], "135": []}
    if lines is not None:

        for rho, theta in lines[0]:
            t = (theta * (180 / np.pi))
            if (np.isclose(45, t, atol=1e-1)):
                angle_map["45"].append((rho, theta))
            elif (np.isclose(135, t, atol=1e-1)):
                angle_map["135"].append((rho, theta))
            # print (theta*(180/np.pi))

            # a,b = np.cos(theta), np.sin(theta)
            # x0,y0 = a*rho, b*rho
            # x1,y1= int(x0 + 1000*(-b)),int(y0 + 1000*(a))
            # x2,y2 = int(x0 - 1000*(-b)),int(y0 - 1000*(a))

            # cv2.line(img_in,(x1,y1),(x2,y2),(0,255,0),2)

        # raise Exception(angle_map)
        for line1 in angle_map["45"]:
            for line2 in angle_map["135"]:
                for line3 in angle_map["45"]:
                    for line4 in angle_map["135"]:
                        if (line1 == line3 or line2 == line4):
                            continue

                        # raise Exception("%s %s %s!" % (deg30,deg90,deg150))
                        # get vertices of lines
                        a = [[np.cos(line1[1]), np.sin(line1[1])], [np.cos(line2[1]), np.sin(line2[1])]]
                        b = [line1[0], line2[0]]
                        line12 = np.linalg.solve(a, b)
                        # cv2.circle(img_in,(line12[0],line12[1]),2,(0,0,0),3)

                        a = [[np.cos(line1[1]), np.sin(line1[1])], [np.cos(line4[1]), np.sin(line4[1])]]
                        b = [line1[0], line4[0]]
                        line14 = np.linalg.solve(a, b)
                        # cv2.circle(img_in,(line14[0],line14[1]),2,(0,0,0),3)

                        a = [[np.cos(line2[1]), np.sin(line2[1])], [np.cos(line3[1]), np.sin(line3[1])]]
                        b = [line2[0], line3[0]]
                        line23 = np.linalg.solve(a, b)
                        # cv2.circle(img_in,(line23[0],line23[1]),2,(0,0,0),3)

                        a = [[np.cos(line3[1]), np.sin(line3[1])], [np.cos(line4[1]), np.sin(line4[1])]]
                        b = [line3[0], line4[0]]
                        line34 = np.linalg.solve(a, b)
                        # cv2.circle(img_in,(line34[0],line34[1]),2,(0,0,0),3)

                        verticies = (line12, line14, line23, line34)
                        # print magnitude(line12,line34)
                        # print magnitude(line14,line23)
                        # print verticies
                        # find centroid
                        centroid = np.uint16(
                            ((verticies[0][0] + verticies[1][0] + verticies[2][0] + verticies[3][0]) / 4,
                             (verticies[0][1] + verticies[1][1] + verticies[2][1] + verticies[3][1]) / 4))

                        # redspot = (centroid[0],np.uint16((deg90deg150[1]+deg30deg90[1])/2)+5)
                        # print centroid
                        # check if white, then probably yield sign
                        # Also check if right orientation ("tip" is pointing down
                        # print np.array_equal(img_in[centroid[1],centroid[0]],[255,255,255])
                        # check if pixel right below 90deg line is red
                        # raise Exception("%s %s" % (angle_map, centroid))
                        # print img_in[centroid[1],centroid[0]]
                        if (centroid[0] in range(0, img_in.shape[1], 1) and centroid[1] in range(0, img_in.shape[0],
                                                                                                 1)):
                            if (isCloseColour(img[centroid[1], centroid[0]], [0, 128, 255], 15) and
                                    np.isclose(100, magnitude(line12, line34), atol=5)):
                                result = (centroid[0], centroid[1])
                                # cv2.circle(img_in,(result[0],result[1]),2,(0,0,0),3)

    # cv2.circle(img_in,(result[0],result[1]),2,(0,0,0),3)
    return result


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    img = cv2.medianBlur(img_in, 11)
    gray_img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 0.1, 100, param1=10, param2=13, minRadius=26,
                               maxRadius=40)
    # print circles
    result = ()
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # print i
            colour = img[i[1], i[0]]
            # print colour
            if (isCloseColour(colour, [255, 255, 255], 15)):
                result = i
                # print result
                # cv2.circle(img_in,(i[0],i[1]),2,(0,0,255),3)

            # cv2.circle(img_in,(i[0],i[1]),i[2],(0,255,0),2)
            # #draw the center of the circle
            # cv2.circle(img_in,(i[0],i[1]),2,(0,0,255),3)
    # print result
    return (result[0], result[1])
    # return (result[0],result[1])


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
    functions = [do_not_enter_sign_detection,
                 stop_sign_detection,
                 construction_sign_detection,
                 warning_sign_detection,
                 yield_sign_detection
                 ]
    sign_labels = ['no_entry', 'stop', 'const', 'warn', 'yield']
    results = {}
    for fn, label in zip(functions, sign_labels):
        point = fn(img_in)
        if (point != ()):
            results[label] = point

    tl = traffic_light_detection(img_in, range(10, 30, 1))
    if (tl[0] != ()):
        results["traffic_light"] = tl[0]

    return results


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
    functions = [do_not_enter_sign_detection,
                 stop_sign_detection,
                 construction_sign_detection,
                 warning_sign_detection,
                 yield_sign_detection
                 ]
    sign_labels = ['no_entry', 'stop', 'const', 'warn', 'yield']
    results = {}
    for fn, label in zip(functions, sign_labels):
        point = fn(img_in)
        if (point != ()):
            results[label] = point

    tl = noisy_traffic_light_detection(img_in)
    if (tl != ()):
        results["traffic_light"] = tl

    return results


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


def noisy_traffic_light_detection(img_in):
    img = cv2.medianBlur(img_in, 3)
    gray_img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray_img, cv2.cv.CV_HOUGH_GRADIENT, 0.01, 10, param1=18, param2=19, minRadius=13,
                               maxRadius=20)
    result = ()
    x = 0
    y = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # print img[i[1]][i[0]]
            if (isCloseColour(img[i[1]][i[0]], [0, 128, 128], 15) or
                    isCloseColour(img[i[1]][i[0]], [0, 255, 255], 15)):
                result = (i[0], i[1])
                # cv2.circle(img_in,(result[0],result[1]),2,(0,0,0),3)

    return result