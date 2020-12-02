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
        self.Q = Q
        self.R = R
        self.state = np.array([init_x, init_y, 0., 0.])  # state
        self.D = [[1., 0., 1., 0.],
                  [0., 1., 0., 1.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]]
        self.M = [[1., 0., 0., 0.],
                  [0., 1., 0., 0.]]
        self.K = np.matrix(np.zeros_like(np.transpose(self.M)))
        self.P = np.matrix(np.zeros_like(self.D))
        self.Y = np.array([init_x, init_y])
        # raise NotImplementedError

    def predict(self):
        self.state = np.dot(self.state, self.D)
        self.P = np.dot(self.D, np.dot(self.P, np.transpose(self.D))) + self.Q
        # raise NotImplementedError

    def correct(self, meas_x, meas_y):
        self.K = np.dot(np.dot(self.P, np.transpose(self.M)),
                        np.linalg.inv(self.M * self.P * np.transpose(self.M) + self.R))
        self.Y = np.array([meas_x, meas_y])
        # temp = self.Y - np.dot(self.M, self.state)
        # print(self.K.shape)
        self.state = self.state + np.dot((self.Y - np.dot(self.state, np.transpose(self.M))), np.transpose(self.K))
        self.P = self.P - np.dot(np.dot(self.K, self.M), self.P)
        # raise NotImplementedError

    def process(self, measurement_x, measurement_y):
        self.predict()
        self.correct(measurement_x, measurement_y)
        # print(self.state[1])
        return self.state[0, 0], self.state[0, 1]


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
        self.frame = self.get_gray_scale(frame)
        self.template = self.get_gray_scale(template)
        # self.template = template
        # self.frame = frame
        m, n = np.shape(self.frame)
        particles_x = np.random.choice(n, self.num_particles, True).astype(float)
        particles_y = np.random.choice(m, self.num_particles, True).astype(float)
        self.particles = np.stack((particles_x, particles_y),
                                  axis=-1)  # p.ones((self.num_particles,2)) * 150 # Initialize your particles array. Read the docstring.
        self.weights = np.ones(self.num_particles) * (
                    1 / self.num_particles)  # Initialize your weights array. Read the docstring.
        # Initialize any other components you may need when designing your filter.
        self.state = np.array([0., 0.])
        self.index = np.arange(self.num_particles)
        self.new_particle = np.zeros_like(self.particles)
        # raise NotImplementedError

    def get_gray_scale(self, frame):
        img_temp_R = frame[:, :, 0]
        img_temp_G = frame[:, :, 1]
        img_temp_B = frame[:, :, 2]
        img_temp = img_temp_R * 0.3 + img_temp_G * 0.58 + img_temp_B * 0.12
        return img_temp

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

        m, n = template.shape
        mse = np.sum(np.subtract(template, frame_cutout, dtype=np.float32) ** 2.)
        mse = mse / float(m * n)
        sim = (-1) * mse / 2. / (self.sigma_exp ** 2.)
        sim = np.exp(sim)
        # print(sim)
        return sim
        # return NotImplementedError

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.

        Returns:
            numpy.array: particles data structure.
        """

        sw, sh = self.template.shape
        mw, mh = self.frame.shape
        # sample new particle indices using the distribution of the weights
        j = np.random.choice(self.index, self.num_particles, True, p=self.weights)
        # sample the particles using the distribution of the weights
        # print(self.particles[0,7])
        for i in range(self.num_particles):
            tep = j[i]
            self.new_particle[i, :] = self.particles[tep, :]

        # clip particles in case the window goes out of the image limits
        self.new_particle[:, 0] = np.clip(self.new_particle[:, 0], 0, mh - 1)
        self.new_particle[:, 1] = np.clip(self.new_particle[:, 1], 0, mw - 1)

        return self.new_particle
        # return NotImplementedError

    def observe(self, img):
        # get patches corresponding to each particle
        mh, mw = np.shape(self.template)
        img = self.get_gray_scale(img)
        sh, sw = np.shape(img)
        minx = (self.particles[:, 0] - mw / 2).astype(np.int)
        miny = (self.particles[:, 1] - mh / 2).astype(np.int)
        minx = np.clip(minx, 0, sw - mw - 1)
        miny = np.clip(miny, 0, sh - mh - 1)
        candidates = [img[miny[i]:miny[i] + mh, minx[i]:minx[i] + mw]
                      for i in range(self.num_particles)]

        # print(minx[0],minx[0]+mw)

        # compute importance weight - similarity of each patch to the model
        self.weights = np.array([self.get_error_metric(self.template, cand) for cand in candidates])
        # normalize the weights
        # print(np.sum(self.weights))
        self.weights /= np.sum(self.weights)

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
        self.particles += np.random.normal(0, self.sigma_dyn, self.particles.shape)
        self.observe(frame)
        self.particles = self.resample_particles()
        state_idx = np.random.choice(self.index, 1, p=self.weights)
        self.state = self.particles[state_idx, :]

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

        def distance(pa, pb, a, b):
            dis = np.sqrt((pa - a) ** 2 + (pb - b) ** 2)
            return dis

        # frame_out = np.copy(frame_in)
        m, n = np.shape(self.template)

        x_weighted_mean = 0
        y_weighted_mean = 0

        # print(self.particles[0, 0], self.weights[0])

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
            cv2.circle(frame_in, (int(self.particles[i, 0]), int(self.particles[i, 1])), 2, (200, 0, 0), -1)

        cv2.rectangle(frame_in, (int(x_weighted_mean) - n // 2, int(y_weighted_mean) - m // 2),
                      (int(x_weighted_mean) + n // 2, int(y_weighted_mean) + m // 2), (0, 200, 0), 2)

        # temp = np.linalg.norm(self.particles - (x_weighted_mean,y_weighted_mean))
        # dis_weighted_mean = np.sum(temp * self.weights.reshape((-1,1)))
        # print(self.particles[0,0],self.particles[190,0])
        dis_weighted_mean = 0
        for i in range(self.num_particles):
            temp = distance(self.particles[i, 0], self.particles[i, 1], x_weighted_mean, y_weighted_mean)
            dis_weighted_mean += temp * self.weights[i]
        # print(dis_weighted_mean)

        cv2.circle(frame_in, (int(x_weighted_mean), int(y_weighted_mean)), int(dis_weighted_mean), (200, 200, 200), 2)

        # Complete the rest of the code as instructed.
        # cv2.imshow('test',frame_in)
        # cv2.waitKey(0)
        # frame_in = np.copy(frame_out)
        return frame_in
        # raise NotImplementedError


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

    def update_model(self, frame):
        sh, sw = np.shape(self.frame)
        mh, mw = np.shape(self.template)
        ind = np.argmax(self.weights)
        x_weighted_mean = self.particles[ind, 0]
        y_weighted_mean = self.particles[ind, 1]
        minx = (x_weighted_mean - mw / 2).astype(np.int)
        miny = (y_weighted_mean - mh / 2).astype(np.int)
        minx = np.clip(minx, 0, sw - mw - 1)
        miny = np.clip(miny, 0, sh - mh - 1)
        # print(minx)
        # print(frame.shape)
        temp_model = frame[miny:miny + mh, minx:minx + mw]
        # print(self.alpha)
        if temp_model.shape == self.template.shape:
            self.template = self.alpha * temp_model + (1. - self.alpha) * self.template
            self.template = self.template.astype(np.uint8)

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

        self.particles += np.random.normal(0, self.sigma_dyn, self.particles.shape)
        ParticleFilter.observe(self, frame)
        frame = ParticleFilter.get_gray_scale(self, frame)
        if self.alpha > 0:
            self.update_model(frame)
        self.particles = ParticleFilter.resample_particles(self)
        # print(self.state)

        # raise NotImplementedError


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

        self.beta = kwargs.get('beta')
        self.count = 0
        self.template_not_change = ParticleFilter.get_gray_scale(self, template)

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
        self.count += 1
        ParticleFilter.observe(self, frame)
        ind = np.argmax(self.weights)
        maxw = self.weights[ind]
        if (maxw >= 0.025 and (120 < self.count < 160)):
            self.particles[:, 0] -= 0.03
            pass
        elif (maxw >= 0.025 and 185 < self.count < 230):
            self.particles[:, 0] -= 0.5
            pass
        else:
            self.particles += np.random.normal(0, self.sigma_dyn, self.particles.shape)
            self.particles = ParticleFilter.resample_particles(self)
        # print(std)
        ratio = (self.beta) ** (self.count)
        self.template = cv2.resize(self.template_not_change, (0, 0), fx=ratio, fy=ratio)

        # raise NotImplementedError


