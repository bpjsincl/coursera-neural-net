import copy
import os

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import matplotlib.pyplot as plt

from courseraneuralnet.utility.utils import loadmat, logistic

__all__ = ['A4']


class A4(object):
    def __init__(self):
        """
        """
        a4_randomness_source = loadmat(os.path.join(os.getcwd(), 'Data/a4_randomness_source.mat'))
        data = loadmat(os.path.join(os.getcwd(), 'Data/data.mat'))
        self.randomness_source = a4_randomness_source['randomness_source']
        self.data_sets = data['data']
        self.report_calls_to_sample_bernoulli = False

        self.test_rbm_w = self.a4_rand([100, 256], 0) * 2 - 1
        self.small_test_rbm_w = self.a4_rand([10, 256], 0) * 2 - 1

        self.data_1_case = self.sample_bernoulli(self.extract_mini_batch(self.data_sets['training'], 1, 1)['inputs'])
        self.data_10_cases = self.sample_bernoulli(self.extract_mini_batch(self.data_sets['training'], 100,
                                                                           10)['inputs'])
        self.data_37_cases = self.sample_bernoulli(self.extract_mini_batch(self.data_sets['training'], 200,
                                                                           37)['inputs'])

        self.test_hidden_state_1_case = self.sample_bernoulli(self.a4_rand([100, 1], 0))
        self.test_hidden_state_10_cases = self.sample_bernoulli(self.a4_rand([100, 10], 1))
        self.test_hidden_state_37_cases = self.sample_bernoulli(self.a4_rand([100, 37], 2))

        self.report_calls_to_sample_bernoulli = True

    def a4_rand(self, requested_size, seed):
        start_i = int(round(seed) % round(len(self.randomness_source) / 10))
        if start_i + np.prod(requested_size) >= len(self.randomness_source):
            raise Exception('a4_rand failed to generate an array of that size (too big)')
        ret = np.reshape(self.randomness_source[start_i:start_i + np.prod(requested_size)], requested_size)
        return ret

    @staticmethod
    def extract_mini_batch(data_set, start_i, n_cases):
        mini_batch = dict()
        mini_batch['inputs'] = data_set['inputs'][:, start_i: start_i + n_cases]
        mini_batch['targets'] = data_set['targets'][:, start_i: start_i + n_cases]
        return mini_batch

    def sample_bernoulli(self, probabilities):
        if self.report_calls_to_sample_bernoulli:
            print 'sample_bernoulli() was called with a matrix of size {0} by {1}.'.format(np.shape(probabilities))
        seed = np.sum(probabilities)
        return np.array(probabilities > self.a4_rand(np.shape(probabilities), seed), dtype=float)

    def a4_main(self, n_hid, lr_rbm, lr_classification, n_iterations):
        # first, train the rbm
        self.report_calls_to_sample_bernoulli = False
        if np.prod(np.shape(self.data_sets)) != 1:
            raise Exception('You must run a4_init before you do anything else.')

        rbm_w = self.optimize([n_hid, 256],
                              self.cd1,
                              self.data_sets['training'],
                              lr_rbm,
                              n_iterations,
                              use_inputs_only=True)
        # rbm_w is now a weight matrix of <n_hid> by <number of visible units, i.e. 256>
        self.show_rbm(rbm_w)
        input_to_hid = rbm_w
        # calculate the hidden layer representation of the labeled data
        hidden_representation = logistic(np.dot(input_to_hid, self.data_sets['training']['inputs']))
        # train hid_to_class
        data_2 = dict()
        data_2['inputs'] = hidden_representation
        data_2['targets'] = self.data_sets['training']['targets']
        hid_to_class = self.optimize([10, n_hid],
                                     self.classification_phi_gradient,
                                     data_2,
                                     lr_classification,
                                     n_iterations,
                                     use_inputs_only=False)

        # report results
        for data_name, data in self.data_sets.iteritems():
            hid_input = np.dot(input_to_hid, data['inputs'])  # size: <number of hidden units> by <number of data cases>
            hid_output = logistic(hid_input)  # size: <number of hidden units> by <number of data cases>
            class_input = np.dot(hid_to_class, hid_output)  # size: <number of classes> by <number of data cases>
            # log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities.
            # size: <1> by <number of data cases>
            class_normalizer = self.log_sum_exp_over_rows(class_input)
            # log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
            log_class_prob = class_input - np.tile(class_normalizer, (np.size(class_input, 0), 1))
            error_rate = np.mean(np.argmax(class_input, axis=0) != np.argmax(data['targets'], axis=0))
            # scalar. select the right log class probability using that sum then take the mean over all data cases.
            loss = -np.mean(np.sum(log_class_prob * data['targets'], 0))
            print ('For the {0} data, the classification cross-entropy loss is {1}, and the classification error '
                   'rate (i.e. the misclassification rate) is {2}'.format(data_name, loss, error_rate))

        self.report_calls_to_sample_bernoulli = True

    def classification_phi_gradient(self, input_to_class, data):
        """This returns the gradient of phi (a.k.a. negative the loss) for the <input_to_class> matrix.
        Notes:
        * This is about a very simple model: there's an input layer, and a softmax output layer.
          There are no hidden layers, and no biases.

        Args:
            input_to_class (numpy.ndarray)  : is a matrix of size <number of classes> by <number of input units>.
            data (numpy.ndarray)            : has fields .inputs (matrix of size <number of input units> by
                <number of data cases>) and .targets (matrix of size <number of classes> by <number of data cases>).
        """
        # input to the components of the softmax. size: <number of classes> by <number of data cases>
        class_input = input_to_class * data['inputs']
        # log(sum(exp)) is what we subtract to get normalized log class probabilities.
        # size: <1> by <number of data cases>
        class_normalizer = self.log_sum_exp_over_rows(class_input)
        # log of probability of each class. size: <number of classes> by <number of data cases>
        log_class_prob = class_input - np.tile(class_normalizer, (np.size(class_input, 0), 1))
        # probability of each class. Each column (i.e. each case) sums to 1.
        # size: <number of classes> by <number of data cases>
        class_prob = np.exp(log_class_prob)
        # now: gradient computation
        # size: <number of classes> by <number of data cases>
        d_loss_by_d_class_input = -(data['targets'] - class_prob) / np.size(data['inputs'], 1)
        # size: <number of classes> by <number of input units>
        d_loss_by_d_input_to_class = d_loss_by_d_class_input * data['inputs'].T
        d_phi_by_d_input_to_class = -d_loss_by_d_input_to_class
        return d_phi_by_d_input_to_class

    @staticmethod
    def log_sum_exp_over_rows(matrix):
        """This computes log(sum(exp(a), 1)) in a numerically stable way"""
        maxs_small = np.max(matrix, axis=0)
        maxs_big = np.tile(maxs_small, (np.size(matrix, 0), 1))
        return np.log(sum(np.exp(matrix - maxs_big), 1)) + maxs_small

    def optimize(self, model_shape, gradient_function, training_data, learning_rate, n_iterations, use_inputs_only):
        """This trains a model that's defined by a single matrix of weights.

        Notes:
        * This uses mini-batches of size 100, momentum of 0.9, no weight decay, and no early stopping.

        Args:
            model_shape (tuple) : is the shape of the array of weights.
            gradient_function   : a function that takes parameters <model> and <data> and returns the gradient
                (or approximate gradient in the case of CD-1) of the function that we're maximizing.
                Note the contrast with the loss function that we saw in PA3, which we were minimizing.
                The returned gradient is an array of the same shape as the provided <model> parameter.

        Returns:
            nd.array : matrix of weights of the trained model
        """
        model = (self.a4_rand(model_shape, np.prod(model_shape)) * 2 - 1) * 0.1
        momentum_speed = np.zeros(model_shape)
        mini_batch_size = 100
        start_of_next_mini_batch = 0
        for iteration_number in xrange(1, n_iterations):
            mini_batch = self.extract_mini_batch(training_data, start_of_next_mini_batch, mini_batch_size)
            start_of_next_mini_batch = np.mod(start_of_next_mini_batch + mini_batch_size,
                                              np.size(training_data['inputs'], 1))
            mini_batch = mini_batch['inputs'] if use_inputs_only else mini_batch
            gradient = gradient_function(model, mini_batch)
            momentum_speed = 0.9 * momentum_speed + gradient
            model = model + momentum_speed * learning_rate
        return model

    def cd1(self, rbm_w, visible_data):
        """
        Args:
            rbm_w (numpy.array)         : a matrix of size <number of hidden units> by <number of visible units>
            visible_data (numpy.array)  : is a (possibly but not necessarily binary) matrix of
                                          size <number of visible units> by <number of data cases>

        Returns:
            The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
        """
        ret = dict()
        raise NotImplementedError('not yet implemented')

    def show_rbm(self, rbm_w):
        n_hid = np.size(rbm_w, 0)
        n_rows = int(np.ceil(np.sqrt(n_hid)))
        blank_lines = 4
        distance = 16 + blank_lines
        to_show = np.zeros([n_rows * distance + blank_lines, n_rows * distance + blank_lines])
        for i in xrange(0, n_hid):
            row_i = int(i / n_rows)  # take floor
            col_i = int(i % n_rows)
            pixels = np.reshape(rbm_w[i, :], (16, 16)).T
            row_base = row_i * distance + blank_lines
            col_base = col_i * distance + blank_lines
            to_show[row_base:row_base + 16, col_base:col_base + 16] = pixels

        extreme = np.max(abs(to_show))
        try:
            plt.imshow(to_show, vmin=-extreme, vmax=extreme)
            plt.title('hidden units of the RBM')
        except:
            print('Failed to display the RBM. No big deal (you do not need the display to finish the assignment), '
                  'but you are missing out on an interesting picture.')
            raise

    def configuration_goodness(self, rbm_w, visible_state, hidden_state):
        """

        Args:
            rbm_w (numpy.array)         : a matrix of size <number of hidden units> by <number of visible units>
            visible_state (numpy.array) : a binary matrix of size <number of visible units> by <number of configurations
                                          that we're handling in parallel>.
            hidden_state (numpy.array)  : a binary matrix of size <number of hidden units> by <number of configurations
                                          that we're handling in parallel>.

        Returns:
            This returns a scalar: the mean over cases of the goodness (negative energy) of the described
            configurations.
        """
        G = None
        raise NotImplementedError('not yet implemented')

    def configuration_goodness_gradient(self, visible_state, hidden_state):
        """
        Notes:
        * You don't need the model parameters for this computation.

        Args:
            visible_state (numpy.array) : is a binary matrix of size <number of visible units> by
                                          <number of configurations that we're handling in parallel>.
            hidden_state (numpy.array)  : is a (possibly but not necessarily binary) matrix of size
                                          <number of hidden units>
                                          by <number of configurations that we're handling in parallel>.

        Returns:
            This returns the gradient of the mean configuration goodness (negative energy, as computed by function
            <configuration_goodness>) with respect to the model parameters.
            Thus, the returned value is of the same shape as the model parameters, which by the way are not provided to
            this function. Notice that we're talking about the mean over data cases (as opposed to the sum over data
            cases).
        """
        d_G_by_rbm_w = None
        raise NotImplementedError('not yet implemented')

    def hidden_state_to_visible_probabilities(self, rbm_w, hidden_state):
        """This takes in the (binary) states of the hidden units, and returns the activation probabilities
         of the visible units, conditional on those states.
        Args:
            rbm_w (numpy.array)         : a matrix of size <number of hidden units> by <number of visible units>
            hidden_state (numpy.array)  : is a binary matrix of size <number of hidden units> by <number of
                                          configurations that we're handling in parallel>.

        Returns:
            The returned value is a matrix of size <number of visible units> by <number of configurations that we're
            handling in parallel>.
        """
        visible_probability = None
        raise NotImplementedError('not yet implemented')

    def visible_state_to_hidden_probabilities(self, rbm_w, visible_state):
        """This takes in the (binary) states of the visible units, and returns the activation probabilities of the
        hidden units conditional on those states.

        Args:
            rbm_w (numpy.array)         : is a matrix of size <number of hidden units> by <number of visible units>
            visible_state (numpy.array) : is a binary matrix of size <number of visible units> by <number of
                                          configurations that we're handling in parallel>.

        Returns:
            The returned value is a matrix of size <number of hidden units> by <number of configurations that we're
            handling in parallel>.
        """
        hidden_probability = None
        raise NotImplementedError('not yet implemented')

    def describe_matrix(self, matrix):
        print('Describing a matrix of size {0} by {1}. The mean of the elements is {2}. '
              'The sum of the elements is {3}'.format(np.size(matrix, 0), np.size(matrix, 1), np.mean(matrix),
                                                      sum(matrix)))

