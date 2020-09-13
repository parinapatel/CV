"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np

def show_img(str, img):
    cv2.imshow(str, img)
    cv2.waitKey(0)

def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """

    return np.linalg.norm(np.array(p0) - np.array(p1))


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

    h, w = image.shape[0], image.shape[1]

    img_corners = [(0, 0),
                   (0, h-1),
                   (w-1, 0),
                   (w - 1, h - 1)]

    return img_corners


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

    blurred = cv2.GaussianBlur(image, (5,5), 0)
    denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    dst = cv2.cornerHarris(gray, 10, 3, 0.04)

    xy = np.where(dst > 0.1 * np.max(dst))
    locations = np.array([(xy[1][i], xy[0][i]) for i in range(len(xy[0]))], dtype=np.float32)

    # h, w = image.shape[0], image.shape[1]
    #
    # img_corners = [(0, 0),
    #                (h - 1, 0),
    #                (0, w - 1),
    #                (h - 1, w - 1)]
    # for corner in img_corners:
    #     locations = [location for location in locations if euclidean_distance(location, corner) > 10]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    _, _, centers = cv2.kmeans(locations, 4, None, criteria, 100, flags)

    markers = []
    for center in centers:
        c = tuple([np.uint(center[0]), np.uint(center[1])])
        markers.append(c)


    markers.sort(key=lambda x: x[0])
    left = markers[:2]
    right = markers[2:]
    left.sort(key=lambda x: x[1])
    right.sort(key=lambda x: x[1])

    p1, p2, p3, p4 = left[0], left[1], right[1], right[0]

    markers = []
    markers.append(p1)
    markers.append(p2)
    markers.append(p4)
    markers.append(p3)
    print(markers)
    return markers


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

    corners = markers
    markers = []
    for marker in corners:
        markers.append(tuple([marker[0], marker[1]]))

    markers.sort(key = lambda x:x[0])
    left = markers[:2]
    right = markers[2:]
    left.sort(key=lambda x:x[1])
    right.sort(key=lambda x:x[1])

    p1, p2, p3, p4 = left[0], left[1], right[1], right[0]

    cv2.line(image, p1, p2, (0,0,0), thickness)
    cv2.line(image, p2, p3, (0, 0, 0), thickness)
    cv2.line(image, p3, p4, (0, 0, 0), thickness)
    cv2.line(image, p4, p1, (0, 0, 0), thickness)

    return image


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

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None
