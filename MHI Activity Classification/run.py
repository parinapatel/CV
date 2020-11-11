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

# Build dataset using MHIs and MEIs

"""
key 
name
videos
    Key
    filename
    frames list
"""

info = [
    ("boxing", "  ", [])
]
actions = [('boxing', 1), ('handclapping', 2), ('handwaving', 3), ('jogging', 4), ('running', 5), ('walking', 6)]

frames = {'boxing': [(0, 36), (36, 72), (72, 108)],
          'handclapping': [(0, 27), (27, 54), (54, 81)],
          'handwaving': [(0, 48), (48, 96), (96, 144)],
          'jogging': [(15, 70), (145, 200), (245, 300)],
          'running': [(15, 37), (114, 137), (192, 216)],
          'walking': [(18, 88), (242, 320), (441, 511)]}

# Th needs to be chosen for each action
Th = [1, 8, 10, 40, 5, 5]

# T needs to be chosen and differs with actions
T = [50, 30, 50, 50, 30, 10]

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

labels, MHIs, MEIs, mus_mhi, vs_mhi, mus_mei, vs_mei = build_dataset(actions, frames)

knn_cv = KNeighborsClassifier(n_neighbors=3)

# knn_cv.fit(vs_mhi, labels)
#train model with cv of 5
print(labels.shape)
cv_scores = cross_val_score(knn_cv, vs_mhi, labels, cv=3)
#print each cv score (accuracy) and average them
print(cv_scores)
print("cv_scores mean:{}".format(np.mean(cv_scores)))
