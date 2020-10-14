"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter
        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([init_x, init_y, 0., 0.])  # state

        # state is basically x, y, vx and vy
        # x = x + dt*vx , y = y + dt*vy, vx = vx, vy = vy
        self.D = [[1., 0., 1., 0.],
                  [0., 1., 1., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]]

        # we can only measure the x and y axis
        self.M = [[1., 0., 0., 0.],
                  [0., 1., 0., 0.]]

        self.Kalman = np.mat(np.zeros((4, 2)))
        self.St = np.mat(np.zeros((4, 4)))
        self.Zt = np.mat([init_x, init_y])

        self.Q = Q
        self.R = R
        # raise NotImplementedError

    def predict(self):
        self.state = np.dot(self.state, self.D)
        self.St = np.dot(self.D, np.dot(self.St, np.transpose(self.D))) + self.Q

    # raise NotImplementedError

    def correct(self, meas_x, meas_y):
        self.Zt = np.array([meas_x, meas_y])

        self.Kalman = np.dot(np.dot(self.St, np.transpose(self.M)),
                             np.linalg.inv(self.M * self.St * np.transpose(self.M) + self.R))

        self.state = self.state + np.dot(self.Zt - np.dot(self.state, np.transpose(self.M)), np.transpose(self.Kalman))
        self.St = self.St - np.dot(np.dot(self.Kalman, self.M), self.St)
        # raise NotImplementedError

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)
        # print(self.state[1])
        return self.state[0,0], self.state[0,1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = self.get_gray_scale(template)
        self.frame = frame
        max_x, max_y, _ = self.frame.shape
        self.particles = np.array([np.random.choice(max_x, self.num_particles, True),
                                   np.random.choice(max_y, self.num_particles, True)]).T  # Initialize your particles array. Read the docstring.
        self.weights = np.ones(self.num_particles)/self.num_particles  # Initialize your weights array. Read the docstring.
        # Initialize any other components you may need when designing your filter.


    def get_gray_scale(self, image):
        B = image[:,:,0]
        G = image[:,:,1]
        R = image[:,:,2]

        return 0.12*B + 0.58*G + 0.3*R

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        x,y = template.shape
        diff = np.subtract(template, frame_cutout, dtype=np.float32)
        mse = np.sum(diff**2) / float(x*y)

        similarity = np.exp(-mse / (2*( (self.sigma_exp)**2) ))

        return similarity


    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.

        Returns:
            numpy.array: particles data structure.
        """
        indices = np.random.choice(self.num_particles, len(self.particles), p=self.weights)
        new_particles = np.array([self.particles[i] for i in indices])

        return new_particles


    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        self.particles = self.particles + np.random.normal(0, self.sigma_dyn, self.particles.shape)

        image = self.get_gray_scale(frame)
        th, tw = np.shape(self.template)
        ih, iw = np.shape(image)
        centers = np.array([np.clip((self.particles[:, 0]-tw/2).astype(int), 0, iw - tw - 1),
                            np.clip((self.particles[:, 1]-th/2).astype(int), 0, ih - th - 1)])

        frame_cutouts = [image[centers[1,i]:centers[1,i] + th, centers[0,i]:centers[0,i] + tw] for i in range(self.num_particles)]

        self.weights = np.array([self.get_error_metric(self.template, frame_cutout) for frame_cutout in frame_cutouts])
        self.weights /= np.sum(self.weights)

        self.particles = self.resample_particles()



    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Complete the rest of the code as instructed.

        # circle for each particle
        for i in range(self.num_particles):
            cv2.circle(frame_in, (int(self.particles[i,0]), int(self.particles[i,1])), 1, (255,0,0), thickness=1)

        # rectangle around the tracking window
        th, tw = self.template.shape

        start = (int(x_weighted_mean - tw/2), int(y_weighted_mean - th/2))
        end = (int(x_weighted_mean + tw/2), int(y_weighted_mean + th/2))

        cv2.rectangle(frame_in, start, end, (0,255,0), thickness=1)

        # create distribution
        # distance_mean = 0
        point_mean = np.array([x_weighted_mean, y_weighted_mean])
        distance_mean = np.sum(np.array([np.linalg.norm(self.particles[i] - point_mean)*self.weights[i] for i in range(self.num_particles)]))

        # for i in range(self.num_particles):
        #     distance_mean += np.linalg.norm(self.particles[i] - point_mean)*self.weights[i]

        cv2.circle(frame_in, (int(x_weighted_mean), int(y_weighted_mean)), int(distance_mean), (255,255,0), thickness=2)

        return frame_in


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        raise NotImplementedError


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        raise NotImplementedError
