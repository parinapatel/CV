import cv2
import numpy as np
import os
import matplotlib.pyplot
import util
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

IN_DIR = "input_videos"
OUT_DIR = "output"

debug = True
# Build dataset using MHIs and MEIs

actions = [('boxing', 1), ('handclapping', 2), ('handwaving', 3), ('jogging', 4), ('running', 5), ('walking', 6)]

frames = {'boxing': [(0, 36), (36, 72), (72, 108)],
          'handclapping': [(0, 27), (27, 54), (54, 81)],
          'handwaving': [(0, 48), (48, 96), (96, 144)],
          'jogging': [(15, 70), (145, 200), (245, 300)],
          'running': [(15, 37), (114, 137), (192, 216)],
          'walking': [(18, 88), (242, 320), (441, 511)]}

# Th needs to be chosen for each action
#Th = [1, 8, 10, 40, 5, 5]
Th = [14,14,14,14,14,14]

# T needs to be chosen and differs with actions
#T = [50, 30, 50, 50, 30, 10]
T = [7,13,17,21,11,35]

# create all the action MHI, MEI, and labels
def build_dataset(actions, frames):
    MHIs = []
    MEIs = []
    mus_mhi = []  # central moments
    vs_mhi = []  # scale invariant moments
    mus_mei = []  # central moments
    vs_mei = []  # scale invariant moments

    labels = []

    for i, action in enumerate(actions):
        act = action[0]
        num = action[1]
        frames_act = frames[act]
        for j, t in enumerate(frames_act):
            start_t = t[0]
            end_t = t[1]
            duration = end_t - start_t
            file = 'person15_' + str(act) + '_d1_uncomp.avi'
            path = os.path.join(IN_DIR, file)

            binaries, real = util.create_binary_images(path, k=(5,) * 2, sigma=0,
                                                       start_frame=start_t, end_frame=end_t,
                                                       k_size=(3,) * 2, threshold=Th[i], folder=act + "_" + str(j))

            MHI = util.create_mhi(binaries, T[i])

            cv2.normalize(MHI, MHI, 0.0, 255.0, cv2.NORM_MINMAX)
            MEI = (255 * MHI > 0).astype(np.uint8)

            mu_mhi, v_mhi = util.create_hu_moments(MHI)
            mu_mei, v_mei = util.create_hu_moments(MEI)

            MHIs.append(MHI)
            MEIs.append(MEI)

            mus_mhi.append(mu_mhi)
            vs_mhi.append(v_mhi)
            mus_mei.append(mu_mei)
            vs_mei.append(v_mei)

            labels.append(num)

    labels = np.array(labels).astype(np.int)
    mus_mhi = np.array(mus_mhi).astype(np.float32)
    vs_mhi = np.array(vs_mhi).astype(np.float32)
    mus_mei = np.array(mus_mei).astype(np.float32)
    vs_mei = np.array(vs_mei).astype(np.float32)

    return labels, MHIs, MEIs, mus_mhi, vs_mhi, mus_mei, vs_mei


def build_entire_dataset(actions):
    MHIs = []
    MEIs = []
    mus_mhi = []  # central moments
    vs_mhi = []  # scale invariant moments
    mus_mei = []  # central moments
    vs_mei = []  # scale invariant moments
    hus_mhi = []
    hus_mei = []

    labels = []

    for i, action in enumerate(actions):
        act = action[0]
        num = action[1]

        files_path = [x for x in os.listdir("./Data MHI/"+str(act))]
        c = 0
        for j,file in enumerate(files_path):

            if ".avi" in file and c<20:
                c += 1
                cap = cv2.VideoCapture(os.path.join("./Data MHI/"+str(act)+"/",file))
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                print(os.path.join("./Data MHI/" + str(act) + "/", file))
                # start = [1, int(length/3)+1, int(2*length/3)+1]
                # end = [int(length/3), int(2*length/3), length-1]
                for iter in range(3):
                    binaries, real = util.create_binary_images(os.path.join("./Data MHI/"+str(act)+"/",file), k=(13,13), sigma=1,
                                                               start_frame=1, end_frame=length-1,
                                                               k_size=(9,9), threshold=Th[i], folder=act + "_" + str(j))

                    MHI = util.create_mhi(binaries, T[i])

                    cv2.normalize(MHI, MHI, 0.0, 255.0, cv2.NORM_MINMAX)
                    MEI = (255 * MHI > 0).astype(np.uint8)

                    # mu_mhi, v_mhi, hu_mhi = util.create_hu_moments(MHI)
                    # mu_mei, v_mei, hu_mei = util.create_hu_moments(MEI)
                    hu_mhi = util.create_hu_moments(MHI)
                    hu_mei = util.create_hu_moments(MEI)
                    MHIs.append(MHI)
                    MEIs.append(MEI)

                    # mus_mhi.append(mu_mhi)
                    # vs_mhi.append(v_mhi)
                    hus_mhi.append(hu_mhi)
                    # mus_mei.append(mu_mei)
                    # vs_mei.append(v_mei)
                    hus_mei.append(hu_mei)

                    labels.append(num)

    labels = np.array(labels).astype(np.int)
    # mus_mhi = np.array(mus_mhi).astype(np.float32)
    # vs_mhi = np.array(vs_mhi).astype(np.float32)
    hus_mhi = np.array(hus_mhi).astype(np.float32)
    # mus_mei = np.array(mus_mei).astype(np.float32)
    # vs_mei = np.array(vs_mei).astype(np.float32)
    hus_mei = np.array(hus_mei).astype(np.float32)

    return labels, MHIs, MEIs, hus_mhi, hus_mei


"""
1. Build Dataset
2. Save it as numpy 
3. Load from numpy
4. create KNN
5. train model
    -> Here I am just using scale invariant features - v
6. create features from test video
7. test model
8. create output video
9. create confusion matrix

"""
# data_loaded = build_dataset(actions, frames)

data_folder = "Data NPY e1"
fields = ["labels","MHI","MEI", "hus_mhi", "hus_mei"]

# 1. build dataset
# data = build_entire_dataset(actions)
# if debug:
#     print("1. Build Dataset - Done")
#
# # 2. Save it as numpy
# if not os.path.exists(data_folder):
#     os.makedirs(data_folder)
# for i, filename in enumerate(fields):
#     np.save(os.path.join("./"+data_folder+"/"+filename+".npy"), data[i])
# if debug:
#     print("2. Save it as numpy - Done")

#3. Load from numpy
data_loaded = []
for field in fields:
    np_data = np.load(os.path.join(data_folder ,field+".npy"))
    data_loaded.append(np_data)

if debug:
    print("3. Load Dataset from Numpy - Done")

# 4. create KNN
# reference https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a
knn_cv = KNeighborsClassifier()

knn_cv.fit(data_loaded[3],data_loaded[0])
if debug:
    print("4.  create KNN - Done")

# 5. train model with cv of 5
print(data_loaded[0].shape)
cv_scores = cross_val_score(knn_cv, data_loaded[3],data_loaded[0], cv=10)
#print each cv score (accuracy) and average them
if debug:
    print("5. train model with cv of 5 - Done")
    print(cv_scores)
    print("cv_scores mean:{}".format(np.mean(cv_scores)))

# 6. create features from test video
#start_frame = [203, 380, 520, 809, 1028, 1228]
start_frame = [1]
end_frame = [108]
#end_frame = [273, 447, 600, 900, 1062, 1271]
# # Th needs to be chosen for each action
# Th = [5, 55, 50, 50, 70, 60]
# # T needs to be chosen and differs with actions
# T = [40, 60, 60, 60, 60, 60]

test_video_path = os.path.join(IN_DIR, "person15_boxing_d1_uncomp.avi")
v = util.get_features(test_video_path, start_frame, end_frame, Th, T)
if debug:
    print("6. create features from test video - Done")
    # print(v)

# 7. test model
predictions = knn_cv.predict(v.reshape(1,-1))
if debug:
    print("7. test model - Done")
    print(predictions)

# 8. create output video
frame_ids = []
actions = {1: "boxing",
           2: "handclapping",
           3: "handwaving",
           4: " jogging",
           5: "running",
           6: "walking"}
output_path = util.create_video_output(test_video_path, "boxing", actions, predictions, start_frame, end_frame, frame_ids)

if debug:
    print("8. create output video - Done")
    print("Output video is created at {}".format(output_path))
