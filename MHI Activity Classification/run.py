import numpy as np
import os
import matplotlib.pyplot as plt
import util
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from datetime import datetime
from pathlib import Path

try:
    from joblib import dump as dump
    from joblib import load as load
except ImportError:
    from pickle import load as load
    from pickle import dump as dump

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t"%(message)s"', level=logging.DEBUG)


class MHI_train(object):
    def __init__(self, dataset_dir="", log_level=logging.INFO, **kwargs):
        self.actions = [
            ('boxing', 1),
            ('handclapping', 2),
            ('handwaving', 3),
            ('jogging', 4),
            ('running', 5),
            ('walking', 6)
        ]
        # Th needs to be chosen for each action
        self.Th = [14, 14, 14, 14, 14, 14]
        # T needs to be chosen and differs with actions
        self.T = [7, 13, 17, 21, 11, 35]
        self.split_percent = 0.8
        # self.data_folder = "Data NPY cv2"
        self.fields = ["Xtrain", "ytrain", "Xtest", "ytest"]
        self.dataset_dir = dataset_dir or "training_dataset"

    def create_dataset(self, ):
        """
        Build dataset using MHIs and MEIs and saves to directory
        Returns: None

        """
        logging.debug("Building dataset.")
        data = util.build_entire_dataset(self.actions, self.Th, self.T, self.split_percent)
        logging.debug("Build dataset done.")
        if not Path(self.dataset_dir).exists():
            os.makedirs(self.dataset_dir)
        for i, filename in enumerate(self.fields):
            logging.debug("Writing {}th data to file {}".format
                          (i, Path(self.dataset_dir, filename + ".npy")))
            np.save(Path(".").joinpath(self.dataset_dir, filename + ".npy"), data[i])
        logging.info("Save it as Numpy done")

    def __load_dataset(self) -> list:
        data_loaded = []
        logging.info("loading data from {}".format(self.dataset_dir))
        for field in self.fields:
            logging.debug("loading data field {} from {}".format(field, Path(self.dataset_dir, field + ".npy")))
            data_loaded.append(np.load(Path(self.dataset_dir, field + ".npy")))
        logging.info("loading data done.")
        return data_loaded

    # reference https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a
    def generate_KNN(self) -> KNeighborsClassifier:
        """

        Args:
            dataset_dir:

        Returns: Knn classifier , Classifier implementing the k-nearest neighbors vote.


        """
        data = self.__load_dataset()
        Xtrain, ytrain, Xtest, ytest = data[0], data[1], data[2], data[3]
        logging.info("Creating KNN from {}".format(self.dataset_dir))
        knn_cv = KNeighborsClassifier(n_neighbors=5)
        logging.debug("Fit the model using Xtrain as training data and ytrain as target values")
        knn_cv.fit(Xtrain, ytrain)
        logging.info("Training KNN Done.")
        return knn_cv

    def predict_test_accuracy(self, knn_classifer: KNeighborsClassifier) -> np.ndarray:
        logging.info("Validating Trained model accuracy.")
        data = self.__load_dataset()
        Xtest, ytest = data[2], data[3]
        del data
        ypredict = knn_classifer.predict(Xtest)
        accuracy = 100 * (np.sum([1 if ytest[i] == ypredict[i] else 0 for i in range(len(ytest))])) / (
            ypredict.shape[0])
        logging.warning("Model training Accuracy: {} %".format(accuracy))
        return ypredict

    def confusion_matrix(self, ypredict: np.ndarray, knn_classifier: KNeighborsClassifier) -> None:
        logging.info("Validating Trained model accuracy.")
        data = self.__load_dataset()
        Xtest, ytest = data[2], data[3]
        del data
        logging.debug("create confusion matrix")
        cnf_mtx = confusion_matrix(ytest, ypredict, normalize="true")
        logging.info("Confusion Matrix: {}".format(cnf_mtx))
        logging.debug("Plotting confusion matrix.")
        disp = plot_confusion_matrix(knn_classifier, Xtest, ytest, cmap=plt.cm.Blues, normalize='true')
        disp.ax_.set_title("confusion matrix")
        plt.savefig("confusion matrix.png", bbox_inches="tight")

    @staticmethod
    def store_classifier(knn_classifier: KNeighborsClassifier, output_pickle_file: str):
        logging.info("Dumping Classifer at {}".format(output_pickle_file))
        with open(output_pickle_file, 'wb') as file:
            dump(knn_classifier, file)
        logging.debug("Dumped Classifier.")

    @staticmethod
    def load_classifier(input_pickle_file: str) -> KNeighborsClassifier:
        logging.info("Loading Classifer from {}".format(input_pickle_file))
        if input_pickle_file == "" or Path(input_pickle_file).is_file():
            logging.error("Model file is not present.")
        else:
            try:
                with open(input_pickle_file, 'rb') as file:
                    data = load(file)
                logging.info("model Loaded")
                return data
            except Exception as e:
                logging.exception("model corrupted.")


class video_predictor(object):
    def __init__(self, model_location=""):
        if model_location == "" or not Path(model_location).is_file():
            logging.error("Model file is not present.")
        else:
            try:
                with open(model_location, 'rb') as f:
                    self.classifier = load(f)
            except Exception as e:
                logging.exception("model corrupted.",stack_info=True)
                exit(1)
        # Th needs to be chosen for each action
        self.Th = [14, 14, 14, 14, 14, 14]
        # T needs to be chosen and differs with actions
        self.T = [7, 13, 17, 21, 11, 35]

        self.actions = [
            ('boxing', 1),
            ('handclapping', 2),
            ('handwaving', 3),
            ('jogging', 4),
            ('running', 5),
            ('walking', 6)
        ]

    def predict(self, test_video_path,output_file_name):

        if output_file_name.endswith(".avi"):
            output_file_name = output_file_name[:-4]

        predictions = util.get_predictions(test_video_path, self.Th, self.T, self.classifier)
        logging.info(set(predictions))
        pred_action = ["None", "boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
        util.create_video_output(test_video_path, output_file_name, pred_action, predictions, [])


if __name__ == '__main__':
    pikle_file = "knn_trained_model.pkl"
    train_model = False
    input_file='./Data MHI/boxing/person12_boxing_d1_uncomp.avi'
    output_name = "parin_test"

    if train_model:
        mhi = MHI_train(dataset_dir="Training Dataset")
        mhi.create_dataset()
        knn = mhi.generate_KNN()
        ypredict = mhi.predict_test_accuracy(knn)
        mhi.confusion_matrix(ypredict, knn)
        mhi.store_classifier(knn, pikle_file)

    v_predictor = video_predictor(pikle_file)
    v_predictor.predict(input_file,output_file_name=output_name)

