import cv2
import ps3
import numpy as np
import os

IMG_DIR = "input_images"
VID_DIR = "input_videos"
OUT_DIR = "./"
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)


input_images = ['ps3-3-a_base.jpg', 'ps3-3-b_base.jpg', 'ps3-3-c_base.jpg']
output_images = ['ps3-3-a-1.png', 'ps3-3-a-2.png', 'ps3-3-a-3.png']

# Advertisement image
advert = cv2.imread(os.path.join(IMG_DIR, "img-3-a-1.png"))
src_points = ps3.get_corners_list(advert)

# Optional template image
template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))

for img_in, img_out in zip(input_images, output_images):
    print("Input image: {}".format(img_in))

    # Open image and identify the four marker positions
    scene = cv2.imread(os.path.join(IMG_DIR, img_in))

    markers = ps3.find_markers(scene, template)

    homography = ps3.find_four_point_transform(src_points, markers)

    imageA = advert.copy()
    imageB = scene.copy()
    copyA = imageA.copy()
    copyB = imageB.copy()

    h,w = copyA.shape[:2]
    H,W = copyB.shape[:2]

    #Make an array of all source points in homogenous coordinates
    A = np.zeros((3, h*w), np.uint32)
    for x in range(w):
        subArr = np.array([np.array([x for i in range(h)]),np.arange(h),np.ones(h)])
        A[:,x*h:(x+1)*h] = subArr

    #Dot product with homography and convert to homogenous coordinates
    B = np.dot(homography, A)
    for i in range(3):
        B[i, :] /= B[2, :]

    B = B.astype(int)
    #make the destination values to be in the range for an image
    B[0, :] = np.clip(B[0, :], 0, W-1)
    B[1, :] = np.clip(B[1, :], 0, H - 1)

    #put the source in the destination
    copyB[B[1,:],B[0,:],:] = copyA[A[1,:],A[0,:],:]







    #first row of src is x, second row is y and third row is 1





