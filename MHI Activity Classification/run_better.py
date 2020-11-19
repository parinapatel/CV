import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import util
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from datetime import datetime
print(datetime.now())
IN_DIR = "input_videos"
OUT_DIR = "output"

debug = True
# Build dataset using MHIs and MEIs

actions = [('boxing', 1), ('handclapping', 2), ('handwaving', 3), ('jogging', 4), ('running', 5), ('walking', 6)]
# Th needs to be chosen for each action
Th = [14,14,14,14,14,14]
# T needs to be chosen and differs with actions
# T = [7,13,17,21,11,35]
split_percent = 0.8
T = [7,14,14,21,14,21]

data_folder = "Data NPY cv2"
fields = ["Xtrain", "ytrain", "Xtest", "ytest"]

# 1. build dataset
data = util.build_entire_dataset(actions, Th, T, split_percent)
if debug:
    print("1. Build Dataset - Done")

# 2. Save it as numpy
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
for i, filename in enumerate(fields):
    np.save(os.path.join("./"+data_folder+"/"+filename+".npy"), data[i])
if debug:
    print("2. Save it as numpy - Done")

#3. Load from numpy
data_loaded = []
for field in fields:
    np_data = np.load(os.path.join(data_folder, field+".npy"))
    data_loaded.append(np_data)

if debug:
    print("3. Load Dataset from Numpy - Done")

# 4. create KNN
# reference https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a
Xtrain, ytrain, Xtest, ytest = data_loaded[0], data_loaded[1], data_loaded[2], data_loaded[3]
knn_cv = KNeighborsClassifier()
knn_cv.fit(Xtrain, ytrain)
if debug:
    print("4.  create KNN - Done")

ypredict = knn_cv.predict(Xtest)
accuracy = 100*(np.sum([1 if ytest[i] == ypredict[i] else 0 for i in range(len(ytest))]))/(ypredict.shape[0])
print("Accuracy: {}".format(accuracy))

#create confusion matrix
cnf_mtx = confusion_matrix(ytest, ypredict, normalize='true')
print("Confusion Matrix: {}".format(cnf_mtx))

# create confusion matrix plot
disp = plot_confusion_matrix(knn_cv, Xtest, ytest, cmap=plt.cm.Blues, normalize='true')
disp.ax_.set_title("confusion matrix")
plt.savefig("confusion matrix.png", bbox_inches="tight")

# 5. create features from test video
#start_frame = [203, 380, 520, 809, 1028, 1228]
# start_frame = [1]
# end_frame = [108]
# end_frame = [273, 447, 600, 900, 1062, 1271]
# # Th needs to be chosen for each action
# Th = [5, 55, 50, 50, 70, 60]
# # T needs to be chosen and differs with actions
# T = [40, 60, 60, 60, 60, 60]
for act, i in actions:
    print("Predicting for {}".format(act))
    test_video_path = os.path.join("./Data MHI/" + str(act), "person01_"+str(act)+"_d1_uncomp.avi")
    # test_video_path = os.path.join(IN_DIR, "Test1.mp4")

    # predictions = util.create_video_predictions(test_video_path, Th, T, knn_cv)
    predictions = util.get_predictions(test_video_path, Th, T, knn_cv)
    print(np.bincount(np.array(predictions)))
    print(set(predictions))

    action = ["None","boxing","handclapping","handwaving","jogging","running","walking"]
    output_path = util.create_video_output(test_video_path, "pred", action, predictions, [])

# v = util.get_features(test_video_path, start_frame, end_frame, Th, T)
# if debug:
#     print("6. create features from test video - Done")
#     # print(v)
#
# # 7. test model
# predictions = knn_cv.predict(v.reshape(1,-1))
# if debug:
#     print("7. test model - Done")
#     print(predictions)
#
# # 8. create output video
# frame_ids = []
# actions = {1: "boxing",
#            2: "handclapping",
#            3: "handwaving",
#            4: " jogging",
#            5: "running",
#            6: "walking"}
# output_path = util.create_video_output(test_video_path, "boxing", actions, predictions, start_frame, end_frame, frame_ids)
#
# if debug:
#     print("8. create output video - Done")
#     print("Output video is created at {}".format(output_path))
print(datetime.now())
