"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os
import math

from helper_classes import WeakClassifier, VJ_Classifier


# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   (tuple): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """

    images_files = [f for f in os.listdir(folder) if f.endswith(".png")]

    X = []
    y = []

    size = tuple(size)

    for file in images_files:
        image = cv2.imread(os.path.join(folder, file))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, size)
        resized_flat = resized.flatten()
        X.append(resized_flat)
        label = file.split('.')[0][-2:]
        y.append(int(label))
    X = np.array(X)
    y = np.array(y)

    return X, y


def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """

    total = X.shape[0]
    N = int(p*total)
    index = np.random.permutation(total)
    Xtrain = X[index[:N], :]
    Xtest = X[index[N:], :]
    ytrain = y[index[:N]]
    ytest = y[index[N:]]

    return Xtrain, ytrain, Xtest, ytest


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """

    mean_face = np.mean(x, axis=0)
    return mean_face


def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """
    # mu = get_mean_face(X)
    # diff = X - mu
    # covariance = np.cov(diff.T, bias=True)
    # covariance = covariance * X.shape[0]
    # eigenVal , eigenVec = np.linalg.eigh(covariance)
    # index = eigenVal.argsort()[::-1]
    # eigenVal_sorted = eigenVal[index]
    # eigenVec_sorted = eigenVec[:, index]
    # eigenVal_pca = eigenVal_sorted[:k]
    # eigenVec_pca = eigenVec_sorted[:,:k]

    mu = get_mean_face(X)
    diff = X - np.array(mu, ndmin=2)
    sigma = np.dot(diff.T, diff)
    eigenVal, eigenVec = np.linalg.eigh(sigma)
    eigenVec = eigenVec.T[::-1]
    eigenVal_pca = eigenVal[::-1][:k]
    eigenVec_pca = eigenVec[:k].T

    return eigenVec_pca, eigenVal_pca


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        for j in range(self.num_iterations):
            self.weights /= np.sum(self.weights)
            weakC = WeakClassifier(self.Xtrain, self.ytrain, self.weights, self.eps)
            weakC.train()
            weakC_results = [weakC.predict(x) for x in self.Xtrain]
            eps = np.sum([self.weights[i] if self.ytrain[i] != weakC_results[i] else 0 for i in range(len(self.ytrain))])
            alpha = 0.5*math.log((1.-eps)/eps)

            self.weakClassifiers.append(weakC)
            self.alphas.append(alpha)
            if eps >= self.eps:
                for i in range(self.num_obs):
                    self.weights[i] *= math.exp((-1)*self.ytrain[i]*self.alphas[j]*weakC.predict(self.Xtrain[i]))
            else:
                break

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        predictions = self.predict(self.Xtrain)
        correct = np.sum([1 if self.ytrain[i]==predictions[i] else 0 for i in range(len(self.ytrain))])
        incorrect = np.sum([1 if self.ytrain[i]!=predictions[i] else 0 for i in range(len(self.ytrain))])

        return correct, incorrect

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        predictions = [np.sign(np.sum(
            [self.alphas[j]*self.weakClassifiers[j].predict(X[i]) for j in range(len(self.alphas))]
        )) for i in range(X.shape[0])]
        predictions = np.array(predictions)
        return predictions

class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)
        r, c = self.position
        h, w = self.size
        divide = int(h/2)
        img[r:r+divide, c:c+w] = 255
        img[r+divide:r+h, c:c+w] = 126

        return img

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)
        r, c = self.position
        h, w = self.size
        divide = int(w/2)
        img[r:r + h, c:c + divide] = 255
        img[r:r + h, c + divide:c + w] = 126

        return img

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)
        r, c = self.position
        h, w = self.size
        divide = int(h/3)
        img[r:r + divide, c:c + w] = 255
        img[r + divide:r + 2*divide, c:c + w] = 126
        img[r + 2*divide:r + h, c:c + w] = 255

        return img

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)
        r, c = self.position
        h, w = self.size
        divide = int(w/3)
        img[r:r + h, c:c + divide] = 255
        img[r:r + h, c + divide:c + 2*divide] = 126
        img[r:r + h, c + 2*divide:c + w] = 255

        return img

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)
        r, c = self.position
        h, w = self.size
        dh = int(h/2)
        dw = int(w/2)

        img[r:r+dh, c:c+dw] = 126
        img[r:r+dh, c+dw:c+w] = 255
        img[r+dh:r+h, c:c+dw] = 255
        img[r+dh:r+h, c+dw:c+w] = 126

        return img

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """

        def get_area(pt1, pt2):
            r1, c1 = pt1
            r2, c2 = pt2
            h = r2 - r1
            w = c2 - c1
            # pt1, pt3 (next line) pt4, pt2
            pt3 = (r1, c1 + w)
            pt4 = (r1 + h, c1)

            area = ii[pt1] + ii[pt2] - ii[pt3] - ii[pt4]

            return area

        ii = ii.astype(np.float32)
        h, w = self.size
        r, c = self.position

        if self.feat_type == (2, 1):
            dh = int(h / 2)
            A = get_area((r - 1, c - 1), (r + dh - 1, c + w - 1))
            B = get_area((r + dh - 1, c - 1), (r + h - 1, c + w - 1))
            return A - B
        if self.feat_type == (1, 2):
            dw = int(w / 2)
            A = get_area((r - 1, c - 1), (r + h - 1, c + dw - 1))
            B = get_area((r - 1, c + dw - 1), (r + h - 1, c + w - 1))
            return A - B
        if self.feat_type == (3, 1):
            dh = int(h / 3)
            A = get_area((r - 1, c - 1), (r + dh - 1, c + w - 1))
            B = get_area((r + dh - 1, c - 1), (r + 2 * dh - 1, c + w - 1))
            C = get_area((r + 2 * dh - 1, c - 1), (r + h - 1, c + w - 1))
            return A + C - B
        if self.feat_type == (1, 3):
            dw = int(w / 3)
            A = get_area((r - 1, c - 1), (r + h - 1, c + dw - 1))
            B = get_area((r - 1, c + dw - 1), (r + h - 1, c + 2 * dw - 1))
            C = get_area((r - 1, c + 2 * dw - 1), (r + h - 1, c + w - 1))
            return A + C - B
        if self.feat_type == (2, 2):
            dh = int(h / 2)
            dw = int(w / 2)
            A = get_area((r - 1, c - 1), (r + dh - 1, c + dw - 1))
            B = get_area((r - 1, c + dw - 1), (r + dh - 1, c + w - 1))
            C = get_area((r + dh - 1, c - 1), (r + h - 1, c + dw - 1))
            D = get_area((r + dh - 1, c + dw - 1), (r + h - 1, c + w - 1))
            return B + C - A - D

        return 0


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """

    outputs = []
    for image in images:
        output = np.cumsum(np.cumsum(image, axis=0), axis=1)
        outputs.append(output)
    return outputs


class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.items():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei-1, sizej-1]))
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):

        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print(" -- compute all scores --")
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        weights = np.hstack((weights_pos, weights_neg))

        print(" -- select classifiers --")
        for i in range(num_classifiers):

            # TODO: Complete the Viola Jones algorithm

            raise NotImplementedError

    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)

        scores = np.zeros((len(ii), len(self.haarFeatures)))

        # Populate the score location for each classifier 'clf' in
        # self.classifiers.

        # Obtain the Haar feature id from clf.feature

        # Use this id to select the respective feature object from
        # self.haarFeatures

        # Add the score value to score[x, feature id] calling the feature's
        # evaluate function. 'x' is each image in 'ii'

        result = []

        # Append the results for each row in 'scores'. This value is obtained
        # using the equation for the strong classifier H(x).

        for x in scores:
            # TODO
            raise NotImplementedError

        return result

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """

        raise NotImplementedError
