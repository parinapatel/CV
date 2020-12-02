import os
import cv2
import numpy as np

DEBUG = False
SAVE_MHI = False
IN_DIR = "input_videos"
OUT_DIR = "output"


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


def create_mei_mhi(prev_frame, image_gen, current_frame, k, sigma, start_frame, end_frame, k_size, threshold, tau,
                   save=False, folder=None):
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
    if save:
        meis = []
        mhis = []

    for frame_num in range(start_frame, end_frame + 1, 1):
        # print(num)
        current_img = image_gen.__next__()

        if current_img is None:
            save = False
            break
        reals.append(current_img)

        # preprocess current frame
        current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        current_img = cv2.GaussianBlur(current_img, k, sigma)
        current_img = cv2.morphologyEx(current_img, cv2.MORPH_OPEN, K)

        # create diff mei image
        binary_mei = np.zeros(img.shape, dtype=np.int32)
        binary = np.abs(cv2.subtract(current_img, img))
        # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, K)
        binary_mei[binary > threshold] = 1

        if save:
            meis.append(np.clip(binary_mei * 255, 0, 255))

        # get mhi image
        mhi = create_mhi_frame(binary_mei, mhi, tau)
        if save:
            mhis.append(mhi * 255)

        mei += binary_mei

        img = current_img
        num += 1

    mei[mei > 1] = 1

    if save:
        file_real = "real_{}_{}.png".format(start_frame, end_frame)
        file_mei = "mei_{}_{}.png".format(start_frame, end_frame)
        file_mei1 = "mei_agg_{}_{}.png".format(start_frame, end_frame)
        file_mhi = "mhi_{}_{}.png".format(start_frame, end_frame)
        gray = []
        for real in reals:
            gray.append(cv2.cvtColor(real, cv2.COLOR_BGR2GRAY))

        # combined = cv2.vconcat([cv2.hconcat(gray), cv2.hconcat(meis), cv2.hconcat(mhis)])
        cv2.imwrite(os.path.join(folder, file_real), cv2.hconcat(gray))
        cv2.imwrite(os.path.join(folder, file_mei), cv2.hconcat(meis))
        cv2.imwrite(os.path.join(folder, file_mei1), mei * 255)
        cv2.imwrite(os.path.join(folder, file_mhi), cv2.hconcat(mhis))

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

    save_frames = False
    folder = ""
    # if "person01" in file:
    #     save_frames = True
    #     folder = os.path.join(OUT_DIR, "{}".format(file.split("\\")[-1][:-4]))
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)

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

    if DEBUG:
        print("Path: {}, Length: {}, Last sub-seq: {}".format(file, length, sub_seq[-1]))

    prev_frame = image_gen.__next__()

    for start, end in sub_seq:

        # get mei mhi and real images from the sequence
        mei, mhi, real = create_mei_mhi(prev_frame, image_gen, start, k=(5, 5), sigma=1,
                                        start_frame=start, end_frame=end,
                                        k_size=(3, 3), threshold=Th, tau=T, save=save_frames, folder=folder)

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
        if DEBUG:
            print("Validation set for {}".format(action))
            for idx in test_idx:
                print(files_path[int(idx)])

        j = 0
        for file in files_path:
            data = hu_from_file(file, act, num, t[i], th[i])

            if j in training_idx:
                Xtrain.extend(data[0])
                ytrain.extend(data[1])
            else:
                Xtest.extend(data[0])
                ytest.extend(data[1])
            j += 1

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
    p = pq[:, 0]
    q = pq[:, 1]

    mu00 = np.float32(np.sum(img))
    h, w = img.shape
    mu10 = np.float32(np.sum(np.arange(w) * np.sum(img, axis=0)))
    mu01 = np.float32(np.sum(np.arange(h) * np.sum(img, axis=1)))
    if mu00 == 0 or mu00 == np.nan:
        mu00 = 1

    xbar = mu10 / mu00
    ybar = mu01 / mu00

    xp = (np.power(np.arange(w) - xbar, p[:, np.newaxis], dtype=np.float32)) * img[:, np.newaxis]
    yq = np.power(np.arange(h) - ybar, q[:, np.newaxis], dtype=np.float32).T
    mu = np.sum(np.sum(xp * yq[:, :, np.newaxis], axis=0), axis=1)

    denom = mu00 ** (1 + np.divide(np.sum(pq, axis=1, dtype=np.float32), 2.))
    v = mu / denom

    hu1 = v[0] + v[1]
    hu2 = np.square(v[0] - v[1]) + (4 * np.square(v[2]))
    hu3 = np.square(v[6] - (3 * v[3])) + np.square((3 * v[4]) - v[7])
    hu4 = np.square(v[6] + v[3]) + np.square(v[4] + v[7])
    hu5 = ((v[6] - (3 * v[3])) * (v[6] + v[3]) * ((np.square(v[6] + v[3])) - (3 * (np.square(v[4] + v[7]))))) + (
            (3 * v[4] - v[7]) * (v[4] + v[7])) * (3 * (np.square(v[6] + v[3])) - (np.square(v[4] + v[7])))
    hu6 = ((v[0] - v[1]) * (np.square(v[6] + v[3]) - np.square(v[4] + v[7]))) + 4 * v[2] * (v[6] + v[3]) * (v[4] + v[7])
    # hu7 = (((3 * v[4]) - v[7]) * (v[4] + v[7]) * ((3 * (np.square(v[6] + v[3]))) - np.square(v[4] + v[7]))) - ((v[6] - (3 * v[3])) * (v[4] + v[7]) * ((3 * (np.square(v[6] + v[3]))) - np.square(v[4] + v[7])))

    hus = np.array([hu1, hu2, hu3, hu4, hu5, hu6])

    hu_moments = np.concatenate((hus, mu, v))
    return hu_moments


def create_video_output(test_video_path, output_name, actions, predictions, save_frames):
    output_path = "./output/predicted-{}.avi".format(output_name)

    image_gen = video_frame_generator(test_video_path)
    img = image_gen.__next__()
    h, w, d = img.shape
    point = (int(w / 7) - 5, int(h / 5))

    out = mp4_video_writer(output_path, (w, h), fps=30)

    frame_num = 1
    i = 0

    while img is not None:

        cv2.putText(img, "[{}]".format(actions[predictions[i]]), point, cv2.FONT_HERSHEY_DUPLEX, 0.75,
                    (255, 0, 0), 1)
        out.write(img)

        if frame_num in save_frames:
            cv2.imwrite("output-{}.png".format(str(frame_num)), img)

        img = image_gen.__next__()

    out.release()

    return output_path


"""
name: get_feat_pred(prev_frame, frames, T, knn_cv)
parameters:     prev_frame - starting frame for background subtraction
                frames - all frames within maximum frame length for each action
                T - tau as well as num of frames needed to predict each action
                knn_cv - model for prediction
algorithm:

at each frame - create mhi and mei_agg
check if it is at any tau
    predict for hu moments at that frame
    if prediction is the same as tau key then return tau key
    else continue for next frame
"""


def get_feat_pred(prev_frame, frames, T, knn_cv, counter=0, g=-1):
    # if image from previous sequence is not given take the last image of this series
    if prev_frame is None:
        img = frames[-1]
    else:
        img = prev_frame
    k_size = (3, 3)
    k = (5, 5)
    sigma = 1
    # preprocess each frame - gray scale -> blur -> morph (dilation)
    K = np.ones(k_size, dtype=np.int32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, k, sigma)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, K)

    # parameters
    h, w = img.shape

    # initialize mei mhi and reals
    mei = np.zeros(img.shape, dtype=np.int32)
    mhi = np.zeros(img.shape, dtype=np.int32)

    for i, frame in enumerate(frames):
        # at each frame - create mhi and mei_agg
        current_img = frame

        if current_img is None:
            break

        # preprocess current frame
        current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        current_img = cv2.GaussianBlur(current_img, k, sigma)
        current_img = cv2.morphologyEx(current_img, cv2.MORPH_OPEN, K)

        # create diff mei image
        binary_mei = np.zeros(img.shape, dtype=np.int32)
        binary = np.abs(cv2.subtract(current_img, img))
        binary_mei[binary > 14] = 1

        # get mhi image
        tau = 7
        if counter == -1:
            tau = T[0]
        mhi = create_mhi_frame(binary_mei, mhi, tau)

        if DEBUG:
            if counter % 50 == 0 or counter == -1:
                cv2.imwrite(os.path.join(OUT_DIR, "mhi_{}.png".format(counter)), mhi * 255)
                cv2.imwrite(os.path.join(OUT_DIR, "mei_{}.png".format(counter)), mei * 255)

            if g != -1:
                cv2.imwrite(os.path.join(OUT_DIR, "mhi_{}.png".format(g)), mhi * 255)
                cv2.imwrite(os.path.join(OUT_DIR, "mei_{}.png".format(g)), binary_mei * 255)


        if counter != -1:
            counter += 1
        if g != -1:
            g += 1

        mei += binary_mei

        img = current_img

        # check if it is at any tau
        if i in T:
            if np.sum(mei) == 0 or np.sum(mhi) == 0:
                continue
            x = create_hu_moments(mhi)
            predict = knn_cv.predict([x])

            # if prediction is the same as tau  index + 1 then return tau index + 1
            if predict == T.index(i) + 1:
                mei[mei > 1] = 1
                return mei, mhi, T.index(i) + 1, counter
            if counter == -1:
                return mei, mhi, predict, counter

    mei[mei > 1] = 1

    return mei, mhi, None, counter


def create_video_predictions(test_video_path, Th, T, knn_cv):
    predictions = []
    frames = []
    cap = cv2.VideoCapture(test_video_path)
    counter = 0

    if DEBUG:
        length = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), validator_video_size(test_video_path))
        print("length: {}".format(length))
        cap.release()

    image_gen = video_frame_generator(test_video_path)

    maxT = max(T)

    prev_frame = image_gen.__next__()

    for i in range(maxT):
        frames.append(image_gen.__next__())

    video_complete = False
    while len(frames) > 0:
        # get mhi and mei and possible prediction
        mei, mhi, prediction, counter = get_feat_pred(prev_frame, frames, T, knn_cv, counter=counter)

        # if mhi is non zero
        if np.sum(mhi) > 0:
            # if possible prediction is not None
            if prediction is not None:
                # append possible prediction to label and get value of tau
                num_frames = T[prediction - 1]
                # put the action label in the frames
                for i in range(num_frames):
                    predictions.append(prediction)
                #  remove the frames to be processed
                prev_frame = frames[num_frames - 1]
                frames = frames[num_frames:]

            # else:
            else:
                # get hu moments of mhi
                x = create_hu_moments(mhi)
                # predict X
                prediction = knn_cv.predict([x])[0]
                # append it to list of predictions so far
                # predictions.append(prediction)
                # if total number of predictions is less than 11
                # or 8/10 of last 10 predictions is the same as predicted then there is no need to do anything
                # else get max bincount prediction and append it
                predictions.append(prediction)
                arr = predictions[-10:]
                if not (len(predictions) <= 11 or np.sum(np.array(arr) == prediction) > 8):
                    prediction = np.bincount(np.array(arr)).argmax()
                for frame in frames[:-1]:
                    predictions.append(prediction)
                prev_frame = frames[-1]
                frames = []
        # else:
        #     there was no prediction for this frame
        #     start with the next frame
        else:
            predictions.append(0)
            prev_frame = frames[0]
            frames = frames[1:]
        while len(frames) <= maxT:
            frame = image_gen.__next__()
            if frame is None:
                pred = predictions[-1]
                for fr in frames:
                    predictions.append(pred)
                video_complete = True
                break
            frames.append(frame)
        if video_complete == True:
            break

    return predictions


def get_predictions(test_video_path, Th, T, knn_cv):
    predictions = []
    frames = []
    cap = cv2.VideoCapture(test_video_path)

    length = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), validator_video_size(test_video_path))
    cap.release()

    if DEBUG:
        print("length: {}".format(length))

    image_gen = video_frame_generator(test_video_path)

    maxT = max(T)

    prev_frame = image_gen.__next__()

    for i in range(length):
        frames.append(image_gen.__next__())

    preds = []
    for t in T:
        g = 0
        act_pred = []
        loops = int(length / t)
        prev_loop_frame = prev_frame
        for loop in range(loops):
            mei, mhi, predict, _ = get_feat_pred(prev_loop_frame, frames[loop:t + loop + 1], [t], knn_cv, counter=-1,
                                                 g=g)
            g += t
            prev_loop_frame = frames[t + loop]
            if isinstance(predict, list) or isinstance(predict, np.ndarray):
                predict = predict[0]
            if predict is None:
                predict = 0
            act_pred.extend([predict] * t)
        if len(act_pred) != length:
            act_pred.extend([predict] * (length - len(act_pred)))
        preds.append(act_pred)

    preds = np.array(preds, dtype=np.int32)

    unique, counts = np.unique(preds, return_counts=True)
    c = dict(zip(unique, counts))
    print(c)

    total = 0
    for count in counts:
        total += count

    actions = [0] * 7
    for k in c.keys():
        actions[k] = float(c[k]) / total
        print("k: {}, v: {}".format(k, float(c[k]) / total))

    act = np.argmax(np.array(actions))

    for i in range(length):
        # predictions.append(int(np.bincount(preds[:,i]).argmax()))
        predictions.append(act)
    return predictions
