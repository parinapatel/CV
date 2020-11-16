import os
import cv2
import numpy as np
from joblib import delayed, Parallel

debug = False


# reference: assignment 3
def video_frame_generator(filename):
    video = cv2.VideoCapture(filename)
    i = 0
    while video.isOpened():
        ret, frame = video.read()
        i += 1
        if ret:
            yield frame
        else:
            break
    video.release()
    yield None


def mp4_video_writer(filename, frame_size, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    filename = filename.replace('.mp4', '.avi')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


def create_mei_mhi(prev_frame, image_gen, current_frame, k, sigma, start_frame, end_frame, k_size, threshold, tau):
    # if image from previous sequence is not given take the image from this one
    if prev_frame is None:
        img = image_gen.__next__()
    else:
        img = prev_frame

    # preprocess each frame - gray scale -> blur -> morph (dilation)
    K = np.ones(k_size, dtype=np.int32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, k, sigma)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, K)

    # parameters
    h, w = img.shape
    num = current_frame

    # initialize mei mhi and reals
    mei = np.zeros(img.shape, dtype=np.int32)
    mhi = np.zeros(img.shape, dtype=np.int32)
    reals = []

    for frame_num in range(start_frame, end_frame + 1, 1):
        # print(num)
        current_img = image_gen.__next__()

        if current_img is None:
            break
        reals.append(current_img)

        # preprocess current frame
        current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        current_img = cv2.GaussianBlur(current_img, k, sigma)
        current_img = cv2.morphologyEx(current_img, cv2.MORPH_OPEN, K)

        # create diff mei image
        binary_mei = np.zeros(img.shape, dtype=np.int32)
        binary = np.abs(np.subtract(current_img, img))
        binary_mei[binary > threshold] = 1

        # get mhi image
        mhi = create_mhi_frame(binary_mei, mhi, tau)

        mei += binary_mei

        img = current_img
        num += 1

    mei[mei > 1] = 1

    return mei, mhi, reals


def validator_video_size(file):
    import sys
    j = 0
    image_gen = video_frame_generator(file)
    while image_gen.__next__() is not None:
        j += 1

    return sys.maxsize if j == 0 else j - 1


def hu_from_file(file, action, num, T, Th, j=0):
    hus_mhi = []
    labels = []

    cap = cv2.VideoCapture(file)

    length = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), validator_video_size(file))
    # print(length)
    cap.release()

    image_gen = video_frame_generator(file)

    splits = int((length - 1) / T)
    sub_seq = []
    for n in range(splits):
        start = int(1 + T * n)
        end = int(start + T)
        sub_seq.append([start, end - 1])

    if debug:
        print("Path: {}, Length: {}, Last sub-seq: {}".format(file, length, sub_seq[-1]))

    prev_frame = image_gen.__next__()

    for start, end in sub_seq:

        # get mei mhi and real images from the sequence
        mei, mhi, real = create_mei_mhi(prev_frame, image_gen, start, k=(13, 13), sigma=1,
                                        start_frame=start, end_frame=end,
                                        k_size=(9, 9), threshold=Th, tau=T)

        # last real image of the sequence will be the first previous frame for the next sequence
        prev_frame = real[-1]

        # if mei and mhi does not have any significance discard them
        if np.sum(mei) == 0 or np.sum(mhi) == 0:
            continue

        hu_mhi = create_hu_moments(mhi)

        hus_mhi.append(hu_mhi)
        labels.append(num)

    return hus_mhi, labels


def build_entire_dataset(actions, th, t, split_percent):
    Xtrain = []
    ytrain = []
    Xtest = []
    ytest = []

    for i, action in enumerate(actions):
        act = action[0]
        num = action[1]
        print("Creating dataset for " + act)

        files_path = [os.path.join("./Data MHI/" + str(act), x) for x in os.listdir("./Data MHI/" + str(act)) if
                      x.endswith(".avi")]

        total = len(files_path)
        indices = np.random.permutation(total)
        training_idx, test_idx = indices[:int(split_percent * total)], indices[int(split_percent * total):]
        # for j, file in enumerate(files_path):
        # if ".avi" in file:
        # abs_path = os.path.join("./Data MHI/"+str(act)+"/",file)

        return_data = Parallel(n_jobs=4)(delayed(hu_from_file)(file, act, num, t[i], th[i]) for file in files_path)
        # hu_mhi, labels = hu_from_file(abs_path, act, num, T[i], Th[i], j)
        j = 0
        for data in return_data:
            if j in training_idx:
                Xtrain.extend(data[0])
                ytrain.extend(data[1])
            else:
                Xtest.extend(data[0])
                ytest.extend(data[1])
            j += 1
        # if j in training_idx:
        #     Xtrain.extend(hu_mhi)
        #     ytrain.extend(labels)
        # if j in test_idx:
        #     Xtest.extend(hu_mhi)
        #     ytest.extend(labels)

    Xtrain = np.array(Xtrain).astype(np.float32)
    ytrain = np.array(ytrain).astype(np.int)
    Xtest = np.array(Xtest).astype(np.float32)
    ytest = np.array(ytest).astype(np.int)

    return Xtrain, ytrain, Xtest, ytest


def create_mhi_frame(mei, mhi, T):
    p1 = T * (mei == 1)
    p2 = np.subtract(mhi, np.ones(mhi.shape))
    mhi = p1 + np.clip(p2, 0, 255) * (mei == 0)

    return mhi


def create_hu_moments(img):
    pq = np.array([[2, 0], [0, 2], [1, 1], [1, 2], [2, 1], [2, 2], [3, 0], [0, 3]])
    p=pq[:,0]
    q = pq[:,1]

    mu00 = np.float32(np.sum(img))
    h, w = img.shape
    mu10 = np.float32(np.sum(np.arange(w)*np.sum(img, axis=0)))
    mu01 = np.float32(np.sum(np.arange(h)*np.sum(img, axis=1)))
    if mu00 == 0 or mu00 == np.nan:
        mu00 = 1

    xbar = mu10 / mu00
    ybar = mu01 / mu00

    xp = (np.power(np.arange(w) - xbar, p[:, np.newaxis], dtype=np.float32))*img[:, np.newaxis]
    yq = np.power(np.arange(h) - ybar, q[:, np.newaxis], dtype=np.float32).T
    mu = np.sum(np.sum(xp*yq[:,:,np.newaxis], axis=0), axis=1)

    denom = mu00 ** (1+np.divide(np.sum(pq, axis=1, dtype=np.float32),2.))
    v = mu/denom

    hus = np.array([
        # 1
        v[0] + v[1],
        # 2
        np.square(v[0] - v[1]) + (4 * np.square(v[2])),
        # 3
        np.square(v[6] - (3 * v[3])) + np.square((3 * v[4]) - v[7]),
        # 4
        np.square(v[6] + v[3]) + np.square(v[4] + v[7]),
        # 5
        ((v[6] - (3 * v[3])) * (v[6] + v[3]) * (
                (np.square(v[6] + v[3])) - (3 * (np.square(v[4] + v[7])))
        )) + ((3 * v[4] - v[7]) * (v[4] + v[7])) * (
            (3 * (np.square(v[6] + v[3])) - (np.square(v[4] + v[7])))
        ),
        # 6
        ((v[0] - v[1]) * (np.square(v[6] + v[3]) - np.square(v[4] + v[7]))) +
        4 * v[2] * (v[6] + v[3]) * (v[4] + v[7])
        # 7
        # (((3*mu[4]) - mu[7])*(mu[4] + mu[7])*(
        #     (3*(np.square(mu[6] + mu[3]))) - np.square(mu[4] + mu[7])
        # )) - ((mu[6] - (3*mu[3]))*(mu[4] + mu[7])*(
        #     (3*(np.square(mu[6] + mu[3]))) - np.square(mu[4] + mu[7])
        # ))

        # (((3*mu[4]) - mu[7])*(mu[6] + mu[3])*(
        #     np.square(mu[6] + mu[3]) - (3*(np.square(mu[4] + mu[7])))
        # )) - ((-mu[6] + (3*mu[3]))*(mu[4] + mu[6])*(
        #     (3*(np.square(mu[6] + mu[3]))) - np.square(mu[4] + mu[7])
        # ))
    ])

    hu_moments = np.concatenate((hus, mu, v))
    return hu_moments


def create_binary_images(video_path, k, sigma, start_frame, end_frame, k_size, threshold, folder):
    image_gen = video_frame_generator(video_path)

    img = image_gen.__next__()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, k, sigma)
    h, w = img.shape

    if debug:
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, "binary"))
            os.makedirs(os.path.join(folder, "real"))
        # outb = mp4_video_writer(os.path.join(folder, "binary.mp4"), (w, h), 40)
        # outr = mp4_video_writer(os.path.join(folder, "real.mp4"), (w, h), 40)

    num = 0

    binaries = []
    reals = []
    K = np.ones(k_size, dtype=np.int32)

    while img is not None:
        current_img = image_gen.__next__()
        if current_img is None:
            break
        current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        current_img = cv2.GaussianBlur(current_img, k, sigma)
        current_img = cv2.morphologyEx(current_img, cv2.MORPH_OPEN, K)

        if start_frame <= num <= end_frame:
            binary = np.abs(cv2.subtract(current_img, img))
            binary = binary.astype(np.uint8)

            binaries.append(binary)
            reals.append(current_img)

            if debug:
                cv2.imwrite(os.path.join(folder, "binary", str(num) + ".png"), binary)
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
        p1 = T * (binary == 1)
        p2 = np.subtract(mhi_t, np.ones(mhi_t.shape))
        mhi_t = p1 + np.clip(p2, 0, 255) * (binary == 0)
    mhi = mhi_t.astype(np.uint8)
    return mhi


def get_features(video_path, start_frame, end_frame, Th, T):
    mus_mhi = []
    vs_mhi = []

    for i in range(len(start_frame)):
        binaries, real = create_binary_images(video_path, k=(5,) * 2, sigma=0,
                                              start_frame=start_frame[i], end_frame=end_frame[i],
                                              k_size=(3,) * 2, threshold=Th[i], folder="test_" + str(i))
        MHI = create_mhi(binaries, T[i])

        cv2.normalize(MHI, MHI, 0.0, 255.0, cv2.NORM_MINMAX)
        MEI = (255 * MHI > 0).astype(np.uint8)

        hu_moments = create_hu_moments(MHI)

        # mus_mhi.append(mu_mhi)
        # vs_mhi.append(v_mhi)

    # mus_mhi = np.array(mus_mhi).astype(np.float32)
    # vs_mhi = np.array(vs_mhi).astype(np.float32)
    hu_moments = np.array(hu_moments).astype(np.float32)

    return hu_moments


def create_video_output(test_video_path, output_name, actions, predictions, start_frame, end_frame, save_frames):
    output_path = "./output/predicted-{}.avi".format(output_name)

    image_gen = video_frame_generator(test_video_path)
    img = image_gen.__next__()
    h, w, d = img.shape
    point = (int(w / 7) - 5, int(h / 5) + 20)

    out = mp4_video_writer(output_path, (w, h), fps=30)

    frame_num = 1
    i = 0
    start = start_frame[i]
    end = end_frame[i]

    while img is not None:

        cv2.putText(img, "(predicted: {})".format(actions[predictions[i]]), point, cv2.FONT_HERSHEY_DUPLEX, 2.5,
                    (255, 0, 0), 1)
        out.write(img)

        if frame_num in save_frames:
            cv2.imwrite("output-{}.png".format(str(frame_num)), img)

        if frame_num == end and len(start_frame) != i + 1:
            i += 1
            start = start_frame[i]
            end = end_frame[i]
        frame_num += 1
        img = image_gen.__next__()

    out.release()

    return output_path
