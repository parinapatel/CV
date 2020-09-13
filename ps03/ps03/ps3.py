"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
from scipy import ndimage

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
    img = image.copy()
    corner_img = image.copy()

    blurred = cv2.GaussianBlur(image, (5,5), 0)
    denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    dst = cv2.cornerHarris(gray, 5, 7, 0.01)
    corner_img[dst > 0.02 * dst.max()] = [0, 0, 255]

    if template is not None:

        #find the max heatmap for the template
        max_resemblance = 0
        max_heatmap = None
        for degree in range(0, 180, 5):
            rotated_template = ndimage.interpolation.rotate(template, degree, mode='constant', reshape=False)
            heatmap = cv2.matchTemplate(img, rotated_template, cv2.TM_CCOEFF_NORMED)
            if heatmap.max() > max_resemblance:
                max_resemblance = heatmap.max()
                max_heatmap = heatmap

        #compare if the heatmap values coincide with the harris corner detection
        markers = []
        while len(markers) < 4:
            _, _, _, max_loc = cv2.minMaxLoc(max_heatmap)
            max_heatmap[max_loc[1], max_loc[0]] = 0
            center_pt = (max_loc[0] + template.shape[0]//2, max_loc[1] + template.shape[1]//2)
            check_distance = True
            for pt in markers:
                if euclidean_distance(pt, center_pt) < 5:
                    check_distance = False
            if check_distance and corner_img[center_pt[1], center_pt[0],2] > 250:
                markers.append(center_pt)

    else:

        xy = np.where(dst > 0.1 * np.max(dst))
        locations = np.array([(xy[1][i], xy[0][i]) for i in range(len(xy[0]))], dtype=np.float32)

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

    copyA = imageA.copy()
    copyB = imageB.copy()

    h, w = copyA.shape[:2]
    H, W = copyB.shape[:2]

    # Make an array of all source points in homogenous coordinates
    A = np.zeros((3, h * w), np.uint32)
    for x in range(w):
        subArr = np.array([np.array([x for i in range(h)]), np.arange(h), np.ones(h)])
        A[:, x * h:(x + 1) * h] = subArr

    # Dot product with homography and convert to homogenous coordinates
    B = np.dot(homography, A)
    for i in range(3):
        B[i, :] /= B[2, :]

    # make the destination values to be in the range for an image
    B = B.astype(int)
    B[0, :] = np.clip(B[0, :], 0, W - 1)
    B[1, :] = np.clip(B[1, :], 0, H - 1)

    # put the source in the destination
    copyB[B[1, :], B[0, :], :] = copyA[A[1, :], A[0, :], :]

    return copyB


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

    src = []
    dst = []
    for i in range(len(src_points)):
        s1, s2 = src_points[i][0], src_points[i][1]
        d1, d2 = dst_points[i][0], dst_points[i][1]
        src.extend([[s1,s2,1, 0,0,0, -d1*s1, -d1*s2],[0,0,0, s1,s2,1, -d2*s1, -d2*s2]])
        dst.extend([[d1],[d2]])

    H, _, _, _ = np.linalg.lstsq(np.array(src),np.array(dst), rcond=None)
    H = np.append(H,1)
    H = H.reshape(3,3)

    # markers = []
    # for pt in src_points:
    #     new_pt = np.dot(H, [[pt[0]],[pt[1]],[1]])
    #     markers.append((int(new_pt[0]),int(new_pt[1])))
    #
    # print("source: {}\n dst:{}\n markers:{}".format(src_points, dst_points, markers))

    return H



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
