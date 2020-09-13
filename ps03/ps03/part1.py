import cv2
import numpy as np

def show_img(str, img):
    return
    cv2.imshow(str, img)
    cv2.waitKey(0)

def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))

def proximal_pts(circles, location, threshold):
    for circle in circles:
        p1 = tuple([circle[0], circle[1]])
        p2 = tuple([location[0], location[1]])
        if (abs(p1[0]-p2[0]) < threshold and abs(p1[1]-p2[1]) < threshold):
            return True
    return False

def draw_circles(circles, img):
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(img, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv2.circle(img, center, radius, (255, 0, 255), 3)







"""
img = cv2.imread("./input_images/sim_clear_scene.jpg")
img = cv2.imread("./input_images/ps3-2-a_base.jpg")
img = cv2.imread("./input_images/test_images/simple_rectangle_noisy_gaussian.png")
blurred = cv2.GaussianBlur(img, (5,5), 0)
blurred = cv2.medianBlur(blurred,5)
denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21)
gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

show_img("gray", gray)



dst = cv2.cornerHarris(gray,2,3,0.04)
xy = np.where(dst > 0.2*np.max(dst))
locations = [(xy[1][i], xy[0][i]) for i in range(len(xy[0]))]


img[dst>0.01*dst.max()]=[0,0,255]
show_img("corners", img)


h, w = img.shape[0], img.shape[1]

img_corners = [(0,0),
               (h-1, 0),
               (0, w-1),
               (h-1, w-1)]
for corner in img_corners:
    locations = [location for location in locations if get_distance(location, corner) > 10]

locations = np.asarray(locations)
locations = np.float32(locations)
    # yy=loc[::-1]
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # criteria = (cv2.TERM_CRITERIA_EPS, 0, 1)
flags = cv2.KMEANS_PP_CENTERS

ret, label, centers = cv2.kmeans(locations, 4, None, criteria, 100, flags)
A = locations[label.ravel()==0]
B = locations[label.ravel()==1]
C = locations[label.ravel()==2]
D = locations[label.ravel()==3]
A = np.uint(A)
B = np.uint(B)
C = np.uint(C)
D = np.uint(D)
centers=np.uint(centers)
print(centers)
verbose = True
if verbose:
    cv2.circle(img,(centers[0][0],centers[0][1]),2,(255,0,0),2)
    cv2.circle(img,(centers[1][0],centers[1][1]),2,(0,255,0),2)
    cv2.circle(img,(centers[2][0],centers[2][1]),2,(0,255,255),2)
    cv2.circle(img,(centers[3][0],centers[3][1]),2,(0,255,0),2)
    for i in A:
        img[i[1]][i[0]] = [255,0,0]
    for i in B:
        img[i[1]][i[0]] = [0,255,0]

    for i in C:
        img[i[1]][i[0]] = [0,255,255]

    for i in D:
        img[i[1]][i[0]] = [111,111,111]

    cv2.imshow('res by groups',img)
    cv2.waitKey(0)

    possible_corners=[]

    for center in centers:
        possible_corners.append((center[0],center[1]))

print(centers)
#find the locations which are nearby of each other with 15 radius
#
# medians = []
#
# for location in locations:
#     count = 0
#     distance = [get_distance(location, l1)<8 for l1 in locations]
#     print(distance)
#     count = sum(distance)
#     print(count)
#
#     if count >= 4:
#         medians.append(location)
#
# print(medians)



# edges = cv2.Canny(gray, 100, 200)
# show_img("edges", edges)
#
# circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
#                                    param1=15, param2=20,
#                                    minRadius=5, maxRadius=50)
#
# if circles is None:
#     print("no circles found :(")
# else:
#     draw_circles(np.uint16(np.around(circles)), img)
#     show_img("circles", img)
#
#     cshape = circles.shape
#     circles = circles.reshape(cshape[1], cshape[2])
#
#     #points which are inside the circles
#     locations = [location for location in locations if proximal_pts(circles, location, 5)]
#
#     #circles which has points inside
#     print(circles)
#     circles = [circle for circle in circles if proximal_pts(locations, circle, 5)]
#     print(circles)


#result is dilated for marking the corners, not important
#dst = cv2.dilate(dst,None)

# print(dst.shape)
#
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,0,255]
# show_img("corners", img)
"""
