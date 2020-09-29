"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2

# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """

    # return cv2.Sobel(image, -1, 1, 0, 0.125, 3, cv2.BORDER_DEFAULT)
    sobelx = cv2.Sobel(image, ddepth=-1, dx=1, dy=0, scale=1.0 / 8, ksize=3)
    return sobelx


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """

    sobelx = cv2.Sobel(image, ddepth=-1, dx=0, dy=1, scale=1.0 / 8, ksize=3)
    return sobelx
    # x = cv2.Sobel(image, -1, 0, 1, 0.125, 3, cv2.BORDER_DEFAULT)


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """
    if k_type is None:
        k_type = "uniform"

    if k_type == "uniform":
        k = np.ones((k_size, k_size), np.float32)/(k_size**2)
    elif k_type == "gaussian":
        k = cv2.getGaussianKernel((k_size**2), sigma, ktype = cv2.CV_32F).reshape((k_size, k_size))
    else:
        print("kernel type is not defined properly. Please define kernel type")
        return

    It = cv2.subtract(img_a, img_b).astype(np.float64)
    Ix = gradient_x(img_a)
    Iy = gradient_y(img_a)

    IxIx = cv2.filter2D(Ix*Ix, -1, k)
    IxIy = cv2.filter2D(Ix*Iy, -1, k)
    IyIy = cv2.filter2D(Iy*Iy, -1, k)
    IxIt = cv2.filter2D(Ix*It, -1, k)
    IyIt = cv2.filter2D(Iy*It, -1, k)

    noise = 0.001*np.random.rand(1)[0]
    detA = IxIx * IyIy - IxIy * IxIy
    detA[detA == 0] = noise

    U = -(IyIt*IxIy - IyIy*IxIt)/detA
    V = -(IxIt*IxIy - IxIx*IyIt)/detA

    return U, V


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    param = 0.4    # for 0.4 the kernel turns out to be gaussian like according to the paper
    kernel_1d = np.array([0.25-param/2, 0.25, param, 0.25, 0.25-param/2])
    kernel = np.outer(kernel_1d, kernel_1d)

    filtered = cv2.filter2D(image, -1, kernel)

    reduced = filtered[::2,::2]

    return reduced



def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """

    pyramid = []
    pyramid.append(image)

    reduction = image
    for level in range(1, levels):
        reduced_img = reduce_image(reduction)
        pyramid.append(reduced_img)
        reduction = reduced_img

    return pyramid



def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """

    output_image = normalize_and_scale(img_list[0])

    rows, cols = output_image.shape
    for i in range(1,len(img_list)):
        out = normalize_and_scale(img_list[i])
        rs, cs = out.shape
        padding = np.zeros((rows-rs, cs))
        out = np.append(out, padding).reshape(rows, cs)
        output_image = np.append(output_image, out, axis=1)

    return output_image


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    r, c = image.shape
    expanded = np.zeros((r*2, c*2))
    expanded[::2,::2] = image

    kernel_1d = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625])
    kernel = np.outer(kernel_1d, kernel_1d)

    out = 4.0*cv2.filter2D(expanded, -1, kernel)

    return out




def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    l_pyr = [0]*len(g_pyr)
    l_pyr[-1] = g_pyr[-1]

    for i in range(len(g_pyr)-2,-1,-1):
        expanded = expand_image(g_pyr[i+1])
        r, c = g_pyr[i].shape
        expanded = expanded[:r,:c]
        l_pyr[i] = g_pyr[i] - expanded

    return l_pyr


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """

    M, N = image.shape
    X, Y = np.meshgrid(range(N), range(M))

    X = X + U
    Y = Y + V

    X, Y = X.astype('float32'), Y.astype('float32')

    warped = cv2.remap(src=image, map1=X, map2=Y, interpolation=interpolation, borderMode=border_mode)

    return warped


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """

    raise NotImplementedError
