"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
from scipy.ndimage import rotate


def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """

    (x0, y0) = p0
    (x1, y1) = p1
    dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    return dist

    raise NotImplementedError


def is_accept(maxtemp, temp, threshold):
    k = 0
    while len(temp) > 0 and k < len(temp):
        if euclidean_distance(maxtemp, temp[k]) < threshold: return False
        k += 1
    return True


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    result = []
    img_temp = np.copy(image)
    # print(image.shape)
    height, width = img_temp.shape[0], img_temp.shape[1]
    tl = (0, 0)
    bl = (0, height - 1)
    tr = (width - 1, 0)
    br = (width - 1, height - 1)
    result.append(tl)
    result.append(bl)
    result.append(tr)
    result.append(br)

    return result

    raise NotImplementedError


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """

    img_temp = np.copy(image)

    findmax = 0

    img_harr = np.copy(image)
    gray = cv2.cvtColor(img_harr, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, 5, 7, 0.01)
    dst = cv2.dilate(dst, None)
    img_harr[dst > 0.02 * dst.max()] = [0, 0, 255]

    for dig in range(36):
        templ = rotate(template, 5 * dig, mode='constant', reshape=False)
        temp_res = cv2.matchTemplate(img_temp, templ, cv2.TM_CCOEFF_NORMED)
        if temp_res.max() > findmax:
            findmax = temp_res.max()
            result = temp_res
            # print(15*dig)

    # result = abs(result1-result2)
    # print(result.shape)
    # cv2.imshow('img_temp_gray', result)
    # print(template.shape)
    threshold = 5
    h, w, _ = template.shape
    pos, temp = [], []
    while len(temp) < 4:
        (minval, maxval, minloc, maxloc) = cv2.minMaxLoc(result)
        result[maxloc[1], maxloc[0]] = 0
        maxtemp = (maxloc[0] + w // 2, maxloc[1] + h // 2)
        # print(maxtemp,is_accept(maxtemp,temp,threshold))
        if is_accept(maxtemp, temp, threshold) and img_harr[maxtemp[1], maxtemp[0], 2] > 250:
            temp.append(maxtemp)
    temp = sorted(temp, key=lambda x: x[0])
    if temp[0][1] <= temp[1][1]:
        pos.append(temp[0])
        pos.append(temp[1])
    else:
        pos.append(temp[1])
        pos.append(temp[0])
    if temp[2][1] <= temp[3][1]:
        pos.append(temp[2])
        pos.append(temp[3])
    else:
        pos.append(temp[3])
        pos.append(temp[2])
    # print(temp)
    return pos
    raise NotImplementedError


def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    img_temp = np.copy(image)
    cv2.line(img_temp, (markers[0][0], markers[0][1]), (markers[1][0], markers[1][1]), color=(0, 50, 255),
             thickness=thickness)
    cv2.line(img_temp, (markers[1][0], markers[1][1]), (markers[3][0], markers[3][1]), color=(0, 50, 255),
             thickness=thickness)
    cv2.line(img_temp, (markers[3][0], markers[3][1]), (markers[2][0], markers[2][1]), color=(0, 50, 255),
             thickness=thickness)
    cv2.line(img_temp, (markers[0][0], markers[0][1]), (markers[2][0], markers[2][1]), color=(0, 50, 255),
             thickness=thickness)

    return img_temp

    raise NotImplementedError


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """

    tempA = np.copy(imageA)
    h, w, _ = tempA.shape
    tempB = np.copy(imageB)
    H, W, _ = tempB.shape
    src_bi = np.zeros((3, h * w), np.int32)
    src_bi[2, :] = 1
    for x in range(w):
        src_bi[0, x * h: (x + 1) * h] = x
        src_bi[1, x * h: (x + 1) * h] = np.arange(h)
    dst_bi = np.dot(homography, src_bi)
    dst_bi[:, :] = dst_bi[:, :] / dst_bi[2, :]
    srcx = np.array(src_bi[0, :])
    srcy = np.array(src_bi[1, :])
    dstX = np.array(dst_bi[0, :])
    dstY = np.array(dst_bi[1, :])

    dstX = np.clip(dstX, 0, W - 1)
    dstY = np.clip(dstY, 0, H - 1)

    dstX = dstX.astype(int)
    dstY = dstY.astype(int)
    tempB[dstY, dstX, :] = tempA[srcy, srcx, :]
    return tempB
    raise NotImplementedError


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    comb = []
    for i in range(4):
        (x1, y1) = src_points[i]
        (x2, y2) = dst_points[i]
        comb.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        comb.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
    comb = np.asarray(comb)
    U, Sigma, Vt = np.linalg.svd(comb)
    L = Vt[-1, :] / Vt[-1, -1]
    res = L.reshape(3, 3)
    return res

    raise NotImplementedError


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None
    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    raise NotImplementedError


def frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()
        # print(frame)

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None
