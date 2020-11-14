import os
import cv2
import numpy as np


debug = False


#reference: assignment 3

def video_frame_generator(filename):
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            yield frame
        else:
            break
    video.release()
    yield None


def mp4_video_writer(filename, frame_size, fps=20):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    filename = filename.replace('.mp4', '.avi')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


def create_binary_images(video_path, k, sigma, start_frame, end_frame, k_size, threshold, folder):
    image_gen = video_frame_generator(video_path)

    img = image_gen.__next__()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, k, sigma)
    h, w = img.shape

    if debug:
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder,"binary"))
            os.makedirs(os.path.join(folder,"real"))
        # outb = mp4_video_writer(os.path.join(folder, "binary.mp4"), (w, h), 40)
        # outr = mp4_video_writer(os.path.join(folder, "real.mp4"), (w, h), 40)

    num = 0

    binaries = []
    reals = []
    K = np.ones(k_size, dtype=np.uint8)

    while img is not None:
        current_img = image_gen.__next__()
        if current_img is None:
            break
        current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        current_img = cv2.GaussianBlur(current_img, k, sigma)

        if start_frame <= num <= end_frame:
            binary = np.abs(cv2.subtract(current_img, img))
            binary = binary.astype(np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, K)

            binaries.append(binary)
            reals.append(current_img)

            if debug:
                cv2.imwrite(os.path.join(folder, "binary", str(num)+".png"), binary)
                cv2.imwrite(os.path.join(folder, "real", str(num) + ".png"), current_img)

                # outb.write(binary)
                # outr.write(current_img)

        img = current_img
        num += 1

    # outb.release()
    # outr.release()

    return binaries, reals


def create_mhi(binaries, T):
    mhi_t = np.zeros(binaries[0].shape, dtype=np.float)
    for binary in binaries:
        p1 = T*(binary == 1)
        p2 = np.subtract(mhi_t, np.ones(mhi_t.shape))
        mhi_t = p1 + np.clip(p2, 0, 255)*(binary == 0)
    mhi = mhi_t.astype(np.uint8)
    return mhi


def create_hu_moments(img):
    pq = [(2,0), (0,2), (1,1), (1,2),(2,1), (2,2), (3,0), (0,3)]

    mu00 = img.sum()
    h, w = img.shape
    mu10 = np.sum(np.arange(w)*img)
    mu01 = np.sum(np.atleast_2d(np.arange(h)).T * img)
    xbar = mu10/mu00
    ybar = mu01/mu00

    mu = np.zeros(len(pq))
    v = np.zeros(len(pq))

    for i,(p,q) in enumerate(pq):
        mupq = np.sum((np.arange(w) - xbar) *
                      np.atleast_2d((np.arange(h) - ybar)).T *
                      img)
        mu[i] = mupq
        denom = mu00 ** (1 + (p+q)/2)
        v[i] = mupq/denom

    return mu, v


def get_features(video_path, start_frame, end_frame, Th, T):
    mus_mhi = []
    vs_mhi = []

    for i in range(len(start_frame)):
        binaries, real = create_binary_images(video_path, k=(5,)*2, sigma=0,
                                          start_frame=start_frame[i], end_frame=end_frame[i],
                                          k_size=(3,)*2, threshold=Th[i], folder="test_"+str(i))
        MHI = create_mhi(binaries, T[i])

        cv2.normalize(MHI, MHI, 0.0, 255.0, cv2.NORM_MINMAX)
        MEI = (255 * MHI > 0).astype(np.uint8)

        mu_mhi, v_mhi = create_hu_moments(MHI)

        mus_mhi.append(mu_mhi)
        vs_mhi.append(v_mhi)

    mus_mhi = np.array(mus_mhi).astype(np.float32)
    vs_mhi = np.array(vs_mhi).astype(np.float32)

    return mus_mhi, vs_mhi


def create_video_output(test_video_path, output_name, actions, predictions, start_frame, end_frame, save_frames):
    output_path = "./output/predicted-{}.avi".format(output_name)

    image_gen = video_frame_generator(test_video_path)
    img = image_gen.__next__()
    h, w, d = img.shape
    point = (int(w/7)-5, int(h/5)+20)

    out = mp4_video_writer(output_path, (w, h), fps=40)

    frame_num = 1
    i = 0
    start = start_frame[i]
    end = end_frame[i]

    while img is not None:

        cv2.putText(img, "(predicted: {})".format(actions[predictions[i]]), point, cv2.FONT_HERSHEY_DUPLEX, 2.5, (255,0,0), 1)
        out.write(img)

        if frame_num in save_frames:
            cv2.imwrite("output-{}.png".format(str(frame_num)), img)

        if frame_num == end and len(start_frame)!=i+1:
            i += 1
            start = start_frame[i]
            end = end_frame[i]
        frame_num += 1
        img = image_gen.__next__()

    out.release()

    return output_path
