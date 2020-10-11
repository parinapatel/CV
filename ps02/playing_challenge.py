import cv2
import numpy as np
import util as u

def show_img(str, img):
    cv2.imshow(str, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def angle_in_thr(val, angles, thr):
    for angle in angles:
        if abs(val-angle) < thr:
            return True
    return False

#img_dne = ["do_not_enter","dne_2", "dne_3","dne_4jpg"]
img_dne = ["dne_5"]

img_yield = ["yield_sign.jpg","yield_sign_ss.jpg","yield_3.png"]
#img_yield = ["yield_3.png"]

img_construction = ["construction_2.jpg", "construction_3.png"]
img_construction = ["construction_3.png"]
img_warning =["warning.PNG", "warning_2.jpg", "warning_3.jpeg","warning_4.jpg","warning_5.jpg"]
img_warning =["warning_2.jpg","warning_5.jpg"]

"""CONSTRUCTION SIGN"""

"""
for img in img_construction:
    img_in = cv2.imread("./input_images/{}".format(img))
    blurred = cv2.GaussianBlur(img_in, (5, 5), 0)
    denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21)
    show_img("denoised", denoised)

    img_hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 43, 46])  # example value
    upper = np.array([10, 255, 255])  # example value
    mask = cv2.inRange(img_hsv, lower, upper)
    show_img("mask", mask)
    #     mask=generate_mask(img_in, [0,128,255])
    edges = cv2.Canny(mask, 100, 200)
    show_img("edges of mask", edges)

    #get relevant lines
    lines = cv2.HoughLines(edges, 1, np.pi / 36, 100)
    lines = lines.reshape(lines.shape[0], lines.shape[2])
    a45 = round(np.pi / 4, 3)
    a135 = round(np.pi - (np.pi / 4), 3)
    lines_1 = np.array(
        [[round(line[0], 3), round(line[1], 3)] for line in lines if angle_in_thr(line[1], [a45, a135], 0.05)])
    print(lines_1)


    #create a blank image with lines in it
    blank_image = np.zeros(img_in.shape, np.uint8)
    u.draw_lines_re(lines_1, blank_image)
    show_img("lines", blank_image)
    edges_blank = cv2.Canny(blank_image, 100,200)
    show_img("edges of blank", edges_blank)
    lines = cv2.HoughLinesP(edges_blank, rho=1, theta=np.pi / 36, threshold=20, minLineLength=5, maxLineGap=5)
    lines = lines.reshape(lines.shape[0], lines.shape[2])

    # for line in lines:
    #     cv2.line(blank_image, (line[0],line[1]), (line[2], line[3]), (255, 255, 255), 2)
    #     #cv2.putText(blank_image, "*", (line[0],line[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        #cv2.putText(blank_image, "*", (line[2], line[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
    show_img("lets see", blank_image)
    # edges = cv2.Canny(denoised, 100, 200)
    # show_img("edges of denoised", edges)

    center = u.find_construction_sign(blank_image, img_in)
    print(center)
    img = img_in.copy()
    x, y = int(center[0]), int(center[1])
    text = "(({},{}),'{}')".format(y, x, "construction")
    xs, ys = img_in.shape[0], img_in.shape[1]
    # orgx = x+50 if x+50+200<xs else x-10
    orgx = x
    orgy = y + 50

    cv2.putText(img_in, text, (orgx, orgy), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), thickness=2)
    cv2.putText(img_in, "*", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
    show_img("final", img_in)
    cv2.imwrite("construction_challenge.png", img_in)

"""
"""WARNING SIGN"""
"""
for img in img_warning:
    img_in = cv2.imread("./input_images/{}".format(img))
    blurred = cv2.GaussianBlur(img_in, (5, 5), 0)
    denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21)
    show_img("denoised", denoised)

    img_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower = np.array([20, 40, 60])  # example value
    upper = np.array([40, 255, 255])  # example value
    mask = cv2.inRange(img_hsv, lower, upper)
    show_img("mask", mask)
    #     mask=generate_mask(img_in, [0,128,255])
    edges = cv2.Canny(mask, 100, 200)
    show_img("edges of mask", edges)

    #get relevant lines
    lines = cv2.HoughLines(edges, 1, np.pi / 72, 100)
    lines = lines.reshape(lines.shape[0], lines.shape[2])
    a45 = round(np.pi / 4, 3)
    a135 = round(np.pi - (np.pi / 4), 3)
    lines_1 = np.array(
        [[round(line[0], 3), round(line[1], 3)] for line in lines if angle_in_thr(line[1], [a45, a135], 0.05)])
    print(lines_1)


    #create a blank image with lines in it
    blank_image = np.zeros(img_in.shape, np.uint8)
    u.draw_lines_re(lines_1, blank_image)
    #show_img("lines", blank_image)
    blurredb = cv2.GaussianBlur(blank_image, (15, 15), 0)
    show_img("edges of blank", blurredb)
    edges_blank = cv2.Canny(blurredb, 100,200)
    edges_blank = cv2.GaussianBlur(edges_blank, (5, 5), 0)
    show_img("edges of blank", edges_blank)
    edges_blank = cv2.Canny(blank_image, 100, 200)

    lines = cv2.HoughLinesP(edges_blank, rho=1, theta=np.pi / 36, threshold=20, minLineLength=5, maxLineGap=5)
    lines = lines.reshape(lines.shape[0], lines.shape[2])

    # for line in lines:
    #     cv2.line(blank_image, (line[0],line[1]), (line[2], line[3]), (255, 255, 255), 2)
    #     #cv2.putText(blank_image, "*", (line[0],line[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        #cv2.putText(blank_image, "*", (line[2], line[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
    #show_img("lets see", blank_image)
    # edges = cv2.Canny(denoised, 100, 200)
    # show_img("edges of denoised", edges)

    center = u.find_warning_sign(blank_image, img_in)
    print(center)
    cv2.putText(img_in, "*", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
    show_img("final", img_in)

    # lines = cv2.HoughLines(edges, 1, np.pi / 72, 100)
    # #u.draw_lines(lines, img_in)
    # lines = lines.reshape(lines.shape[0], lines.shape[2])
    # a45 = round(np.pi/4, 3)
    # a135 = round(np.pi-(np.pi/4),3)
    # lines_1 = np.array([[round(line[0], 3), round(line[1],3)] for line in lines if angle_in_thr(line[1], [a45, a135], 0.05)])
    # print(lines_1)
    # u.draw_lines_re(lines_1, img_in)
    # show_img("lines", img_in)
    #
    # # edges = cv2.Canny(denoised, 100, 200)
    # # show_img("edges of denoised", edges)
    #
    # center = u.construction_sign_detection(mask, img_in)
    # print(center)
    # cv2.putText(img_in, "*", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
    # show_img("final", img_in)
"""

"""DO NOT ENTER SIGN"""

for img in img_dne:
    img_in = cv2.imread("./input_images/{}.png".format(img))
    blurred = cv2.GaussianBlur(img_in, (5, 5), 0)
    denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21)

    img_hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    lower = np.array([156, 43, 200])  # example value
    upper = np.array([180, 255, 255])  # example value
    mask0 = cv2.inRange(img_hsv, lower, upper)
    lower = np.array([0, 43, 200])  # example value
    upper = np.array([10, 100, 255])  # example value
    mask1 = cv2.inRange(img_hsv, lower, upper)
    mask = mask0 + mask1
    edges = cv2.Canny(mask, 100, 200)

    show_img("real one",img_in)
    show_img("edges", edges)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 2000, param1=80, param2=30, minRadius=50,
                               maxRadius=130)

    circles = np.uint16(np.around(circles))
    cshape = circles.shape
    centers = circles.reshape(cshape[1], cshape[2])

    if len(centers) > 0:
        x = centers[0][0]
        y = centers[0][1]
        center = (x, y)
        cv2.putText(img_in, "*", center, cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0), thickness=2)
        show_img("final", img_in)


"""YIELD SIGN"""
"""
for img in img_yield:
    img_in = cv2.imread("./input_images/{}".format(img))
    blurred = cv2.GaussianBlur(img_in, (15, 15), 0)
    #show_img("blurred", blurred)
    denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21)
    #denoised = blurred
    show_img("", denoised)
    img_hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)

    lower = np.array([156, 43, 46])  # example value
    upper = np.array([180, 255, 255])  # example value
    mask = cv2.inRange(img_hsv, lower, upper)
    edges = cv2.Canny(mask, 300, 1200)
    show_img("edges", edges)

    center = u.yield_sign_detection(denoised, img_in)
    print(center)
    cv2.putText(img_in, "*", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
    show_img("final", img_in)
    #
        #
        # cv2.putText(img_in, "*", (x - 8, y + 9), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
        #
        # show_img("final", img_in)
"""