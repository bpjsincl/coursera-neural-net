"""Implements Assignment 4 for Geoffrey Hinton's Neural Networks Course offered through Coursera.

* Trains a Feedforward neural network with pretraining using Restricted Boltzman Machines (RBMs)
* The RBM is used as the visible-to-hidden layer in a network exactly like the one made in programming assignment 3.
* The RBM is trained using Contrastive Divergence gradient estimator with 1 full Gibbs update, a.k.a. CD-1.
* Recognizes USPS handwritten digits.

Abstracts classifiers developed in the course into, a more pythonic Sklearn framework. And cleans up a lot of the
given code.
"""

import os
import matplotlib.pyplot as plt

import numpy as np
from sklearn.base import BaseEstimator

from courseraneuralnet.utility.utils import loadmat, logistic, log_sum_exp_over_rows, batches

__all__ = ['A4Run']


class A4Helper(object):
    """Helper class for neural network and rbm classifiers, as well as assignment initalization.
    """
    def __init__(self):
        """
        Notes:
        * All sizes requested are transposed to account for column matrix as default vector in matlab.
          a4_rand(..) then returns a transposed matrix to have the correct shape.
        """
        a4_randomness_source = loadmat(os.path.join(os.getcwd(), 'Data/a4_randomness_source.mat'))
        self.randomness_source = a4_randomness_source['randomness_source']

    def a4_rand(self, requested_size, seed):
        """Generates "random" matrix of requested size. Randomness is a function of the provided data source.

        Notes:
        * This is to reduce true randomness for grading.

        Args:
            requested_size (tuple)  : Tuple of ints for shape of returned array.
            seed (int)              : Used to compute start index.

        Returns:
            (numpy.array)   : Matrix of requested size from randomness source.
        """
        start_i = int(round(seed) % round(len(self.randomness_source) / 10.0))
        if start_i + np.prod(requested_size) >= len(self.randomness_source):
            raise Exception('a4_rand failed to generate an array of that size (too big)')
        return np.reshape(self.randomness_source[start_i:start_i + np.prod(requested_size)], requested_size).T

    def sample_bernoulli(self, probabilities, report_calls=False):
        """Compute states of given probability vector.

        Args:
            probabilites (numpy.ndarray) : probability vector

        Returns:
            numpy.ndarray : states of given probabilities
        """
        if report_calls:
            print 'sample_bernoulli() was called with a matrix of size {0} by {1}.'.format(*np.shape(probabilities))
        seed = np.sum(probabilities)
        return np.array(probabilities > self.a4_rand(np.shape(probabilities)[::-1], seed), dtype=float)

    @staticmethod
    def configuration_goodness(rbm_w, visible_state, hidden_state):
        """Computes negative energy.

        Args:
            rbm_w (numpy.array)         : a matrix of size <number of hidden units> by <number of visible units>
            visible_state (numpy.array) : a binary matrix of size <number of visible units> by <number of configurations
                                          that we're handling in parallel>.
            hidden_state (numpy.array)  : a binary matrix of size <number of hidden units> by <number of configurations
                                          that we're handling in parallel>.

        Returns:
            float: the mean over cases of the goodness (negative energy) of the described configurations.
        """
        return np.mean(np.sum(np.dot(rbm_w, visible_state) * hidden_state, 0))

    @staticmethod
    def configuration_goodness_gradient(visible_state, hidden_state):
        """Computes gradient of negative energy.

        Notes:
        * You don't need the model parameters for this computation.

        Args:
            visible_state (numpy.array) : is a binary matrix of size <number of visible units> by
                                          <number of configurations that we're handling in parallel>.
            hidden_state (numpy.array)  : is a (possibly but not necessarily binary) matrix of size
                                          <number of hidden units>
                                          by <number of configurations that we're handling in parallel>.

        Returns:
            (numpy.array)   : gradient of negative energy (same shape as as the model parameters)
        """
        return np.dot(hidden_state, visible_state.T) / np.size(visible_state, 1)

    @staticmethod
    def hidden_state_to_visible_probabilities(rbm_w, hidden_state):
        """This takes in the (binary) states of the hidden units, and returns the activation probabilities
         of the visible units, conditional on those states.

        Args:
            rbm_w (numpy.array)         : a matrix of size <number of hidden units> by <number of visible units>
            hidden_state (numpy.array)  : is a binary matrix of size <number of hidden units> by <number of
                                          configurations that we're handling in parallel>.

        Returns:
            (numpy.array)   : Activation probabilities of visible units. size <number of visible units> by
                              <number of configurations that we're handling in parallel>.
        """
        return logistic(np.dot(rbm_w.T, hidden_state))

    @staticmethod
    def visible_state_to_hidden_probabilities(rbm_w, visible_state):
        """This takes in the (binary) states of the visible units, and returns the activation probabilities of the
        hidden units conditional on those states.

        Args:
            rbm_w (numpy.array)         : is a matrix of size <number of hidden units> by <number of visible units>
            visible_state (numpy.array) : is a binary matrix of size <number of visible units> by <number of
                                          configurations that we're handling in parallel>.

        Returns:
            (numpy.array)   : Activation probabilities of hidden units. size <number of visible units> by
                              <number of configurations that we're handling in parallel>.
        """

        return logistic(np.dot(rbm_w, visible_state))

    @staticmethod
    def describe_matrix(matrix):
        print('Describing a matrix of size {0} by {1}. The mean of the elements is {2}. '
              'The sum of the elements is {3}'.format(np.size(matrix, 0), np.size(matrix, 1), np.mean(matrix),
                                                      np.sum(matrix)))


class RBM(BaseEstimator, A4Helper):
    """Implements pre-training for the RBM using CD-1 gradient function.
    """
    def __init__(self,
                 training_iters=1,
                 lr_rbm=0.01,
                 n_hid=300,
                 n_visible=256,
                 train_momentum=0.9,
                 mini_batch_size=100):
        """Initialize RBM model.

        Args:
            training_iters (int)    : number of training iterations
            lr_rbm (float)          : learning rate for RBM model
            n_hid (int)             : number of hidden units
            n_visible (int)         : number of visible units
            train_momentum (float)  : momentum used in training
            mini_batch_size (int)   : size of training batches
        """
        super(RBM, self).__init__()
        self.model_shape = (n_hid, n_visible)
        self.mini_batch_size = mini_batch_size
        self.lr_rbm = lr_rbm
        self.n_iterations = training_iters
        self.train_momentum = train_momentum

        # Model params
        self.rbm_w = None
        self.gradient = None

    def reset_classifier(self):
        """Resets the model parameters.
        """
        self.rbm_w = (self.a4_rand(self.model_shape[::-1], np.prod(self.model_shape)) * 2 - 1) * 0.1
        self.gradient = np.zeros(self.model_shape)

    def fit(self, X, y=None):
        """Fit a model using one step Contrastive Divergence CD-1.
        """
        self._cd1(visible_data=X)
        return self

    def _cd1(self, visible_data):
        """Implements single step contrastive divergence (CD-1).

        Args:
            visible_data (numpy.array)  : is a (possibly but not necessarily binary) matrix of
                                          size <number of visible units> by <number of data cases>

        Returns:
            (numpy.array)   : The returned value is the gradient approximation produced by CD-1.
                              It's of the same shape as <rbm_w>.
        """
        visible_data = self.sample_bernoulli(probabilities=visible_data)  # Question 8
        hidden_probs = self.visible_state_to_hidden_probabilities(rbm_w=self.rbm_w, visible_state=visible_data)
        hidden_states = self.sample_bernoulli(probabilities=hidden_probs)
        initial = self.configuration_goodness_gradient(visible_state=visible_data, hidden_state=hidden_states)
        visible_probs = self.hidden_state_to_visible_probabilities(rbm_w=self.rbm_w, hidden_state=hidden_states)
        visible_states = self.sample_bernoulli(probabilities=visible_probs)
        hidden_probs = self.visible_state_to_hidden_probabilities(rbm_w=self.rbm_w, visible_state=visible_states)
        # hidden_states = self.sample_bernoulli(probabilities=hidden_probs)  # Question 6
        reconstruction = self.configuration_goodness_gradient(visible_state=visible_states, hidden_state=hidden_probs)

        self.gradient = initial - reconstruction

    def train(self, sequences):
        """Implements optimize(..) from assignment. This trains a model that's defined by a single matrix of weights.

        Notes:
        * This uses mini-batches of size 100, momentum of 0.9, no weight decay, and no early stopping.

        Args:
            model_shape (tuple) : is the shape of the array of weights.
            gradient_function   : a function that takes parameters <model> and <data> and returns the gradient
                (or approximate gradient in the case of CD-1) of the function that we're maximizing.
                Note the contrast with the loss function that we saw in PA3, which we were minimizing.
                The returned gradient is an array of the same shape as the provided <model> parameter.

        Returns:
            numpy.array : matrix of weights of the trained model
        """
        self.reset_classifier()
        momentum_speed = np.zeros(self.model_shape)
        for i, mini_batch_x in enumerate(batches(sequences['inputs'], self.mini_batch_size)):
            if i >= self.n_iterations:
                break
            self.fit(mini_batch_x)
            momentum_speed = self.train_momentum * momentum_speed + self.gradient
            self.rbm_w += momentum_speed * self.lr_rbm


class FFNeuralNet(BaseEstimator, A4Helper):
    """Implements Feedforward Neural Network from Assignment 4.
    """
    def __init__(self,
                 rbm_w=None,
                 training_iters=1,
                 lr_net=0.02,
                 n_hid=300,
                 n_classes=10,
                 train_momentum=0.9,
                 mini_batch_size=100):
        """Initialize neural network.

        Args:
            rbm_w (numpy.array)     : weight matrix of RBM
            training_iters (int)    : number of training iterations
            lr_net (float)          : learning rate for neural net classifier
            n_hid (int)             : number of hidden units
            n_classes (int)         : number of classes
            train_momentum (float)  : momentum used in training
            mini_batch_size (int)   : size of training batches
        """
        super(FFNeuralNet, self).__init__()
        self.model_shape = (n_classes, n_hid)
        self.mini_batch_size = mini_batch_size
        self.lr_net = lr_net
        self.n_iterations = training_iters
        self.train_momentum = train_momentum

        # Model params
        assert rbm_w is not None
        self.rbm_w = rbm_w
        self.model = None
        self.d_phi_by_d_input_to_class = None

    def reset_classifier(self):
        """Resets the model parameters.
        """
        self.model = (self.a4_rand(self.model_shape[::-1], np.prod(self.model_shape)) * 2 - 1) * 0.1
        self.d_phi_by_d_input_to_class = np.zeros(self.model_shape)

    def fit(self, X, y):
        """Fit a model using Classification gradient descent.
        """
        self._classification_phi_gradient(inputs=X, targets=y)
        return self

    def train(self, sequences):
        """Implements optimize(..) from assignment. This trains a hid_to_class.

        Notes:
        * This uses mini-batches of size 100, momentum of 0.9, no weight decay, and no early stopping.

        Args:
            model_shape (tuple) : is the shape of the array of weights.
            gradient_function   : a function that takes parameters <model> and <data> and returns the gradient
                (or approximate gradient in the case of CD-1) of the function that we're maximizing.
                Note the contrast with the loss function that we saw in PA3, which we were minimizing.
                The returned gradient is an array of the same shape as the provided <model> parameter.

        Returns:
            (numpy.array) : matrix of weights of the trained model (hid_to_class)
        """
        self.reset_classifier()
        # calculate the hidden layer representation of the labeled data, rbm_w is input_to_hid
        hidden_representation = logistic(np.dot(self.rbm_w, sequences['inputs']))
        momentum_speed = np.zeros(self.model_shape)
        for i, (mini_batch_x, mini_batch_y) in enumerate(zip(batches(hidden_representation, self.mini_batch_size),
                                                             batches(sequences['targets'], self.mini_batch_size))):
            if i >= self.n_iterations:
                break
            self.fit(mini_batch_x, mini_batch_y)
            momentum_speed = self.train_momentum * momentum_speed + self.d_phi_by_d_input_to_class
            self.model += momentum_speed * self.lr_net

    def predict(self, x_sequences):
        """Predict a specific class from a given set of sequences.
        """
        return np.argmax(self.predict_sequences_proba(x_sequences=x_sequences), axis=0)

    def predict_proba(self, inputs):
        """Predict the probability of each class given data inputs.

        Returns:
            (numpy.array) : probability of classes
        """
        hid_input = np.dot(self.rbm_w, inputs)  # size: <number of hidden units> by <number of data cases>
        hid_output = logistic(hid_input)  # size: <number of hidden units> by <number of data cases>
        return np.dot(self.model, hid_output)

    def predict_sequences_proba(self, x_sequences):
        """Predict the probability of each class in a given set of sequences.

        Returns:
            (numpy.array) : class input (size: <number of classes> by <number of data cases>)
        """
        return self.predict_proba(x_sequences['inputs'])

    def predict_sequences_log_proba(self, x_sequences):
        """Predict the log probability of each class in a given set of sequences.

        Returns:
            (numpy.array) : log probability of each class (size: <number of classes, i.e. 10> by <number of data cases>)
        """
        class_input = self.predict_sequences_proba(x_sequences)
        return self.predict_log_proba(class_input)

    def predict_log_proba(self, class_input):
        """Predicts log probability of each class given class inputs

        Notes:
        * log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities.

        Args:
            class_input (numpy.array)   : probability of each class (see predict_sequences_proba(..))
                                          (size: <1> by <number of data cases>)

        Returns:
            (numpy.array) : log probability of each class.
        """
        class_normalizer = log_sum_exp_over_rows(class_input)
        return class_input - np.tile(class_normalizer, (np.size(class_input, 0), 1))

    def compute_error_and_loss(self, sequences, data_name=None):
        """Computes error rate and loss for given data set.

        Notes:
        * select the right log class probability using that sum then take the mean over all data cases.

        Args:
            sequences (numpy.array) : input
            data_name (string)      : name of data used in logging (i.e. training, validation, testing)
        """
        class_input = self.predict_sequences_proba(x_sequences=sequences)
        log_class_prob = self.predict_log_proba(class_input=class_input)
        error_rate = np.mean(np.argmax(class_input, axis=0) != np.argmax(sequences['targets'], axis=0))
        loss = -np.mean(np.sum(log_class_prob * sequences['targets'], 0))
        self.log_results(error_rate, loss, data_name)

    @staticmethod
    def log_results(error_rate, loss, data_name):
        data_name = 'given' if not data_name else data_name
        print ('For the {0} data, the classification cross-entropy loss is {1}, and the classification error '
               'rate (i.e. the misclassification rate) is {2}'.format(data_name, loss, error_rate))

    def _classification_phi_gradient(self, inputs, targets):
        """This returns the gradient of phi (a.k.a. negative of the loss) for the class input.

        Notes:
        * This is about a very simple model: there's an input layer, and a softmax output layer.
          There are no hidden layers, and no biases.

        Args:
            input_to_class (numpy.ndarray)  : input to the components of the softmax. size: <number of classes>
                                              by <number of data cases>
            data (numpy.ndarray)            : has fields .inputs (matrix of size <number of input units> by
                <number of data cases>) and .targets (matrix of size <number of classes> by <number of data cases>).
        """
        # log(sum(exp)) is what we subtract to get normalized log class probabilities.
        class_input = np.dot(self.model, inputs)
        class_prob = np.exp(self.predict_log_proba(class_input))
        # now: gradient computation
        d_loss_by_d_class_input = -(targets - class_prob) / np.size(inputs, 1)
        d_loss_by_d_input_to_class = np.dot(d_loss_by_d_class_input, inputs.T)
        self.d_phi_by_d_input_to_class = -d_loss_by_d_input_to_class


class A4Run(A4Helper):
    """Runs assignment 4.
    """
    def __init__(self):
        """Initialize data set and all test cases for assignment.

        Notes:
        * All sizes requested are transposed to account for column matrix as default vector in matlab.
          a4_rand(..) then returns a transposed matrix to have the correct shape.
        """
        super(A4Run, self).__init__()
        data = loadmat(os.path.join(os.getcwd(), 'Data/data.mat'))
        self.data_sets = data['data']

        self.test_rbm_w = self.a4_rand([256, 100], 0) * 2 - 1
        self.small_test_rbm_w = self.a4_rand([256, 10], 0) * 2 - 1

        self.data_1_case = self.sample_bernoulli(self.extract_mini_batch(self.data_sets['training'], 0, 1)['inputs'],
                                                 report_calls=False)
        self.data_10_cases = self.sample_bernoulli(self.extract_mini_batch(self.data_sets['training'], 99,
                                                                           10)['inputs'], report_calls=False)
        self.data_37_cases = self.sample_bernoulli(self.extract_mini_batch(self.data_sets['training'], 199,
                                                                           37)['inputs'], report_calls=False)

        self.test_hidden_state_1_case = self.sample_bernoulli(self.a4_rand([1, 100], 0), report_calls=False)
        self.test_hidden_state_10_cases = self.sample_bernoulli(self.a4_rand([10, 100], 1), report_calls=False)
        self.test_hidden_state_37_cases = self.sample_bernoulli(self.a4_rand([37, 100], 2), report_calls=False)

    @staticmethod
    def extract_mini_batch(data_set, start_i, n_cases):
        """Extract specified region of data.

        Notes:
        * This is just used for test data initalization and was replaced by batch(..) for use in models

        Args:
            data_set (numpy.array)  : target data set.
            start_i (int)           : starting index for mini batch.
            n_cases (int)           : number of cases to return.

        Returns:
            (numpy.array)   : data set split into n_case sub arrays.
        """
        mini_batch = dict()
        mini_batch['inputs'] = data_set['inputs'][:, start_i: start_i + n_cases]
        mini_batch['targets'] = data_set['targets'][:, start_i: start_i + n_cases]
        return mini_batch

    def build_neural_net_model(self, n_hid, lr_rbm, lr_net, training_iters, n_classes, train_momentum,
                               n_visible, mini_batch_size, show_rbm_weights):
        """Runs pre-training and returns neural network model.

        Args:
            n_hid (int)             : number of hidden units
            lr_rbm (float)          : learning rate for RBM model
            lr_net (float)          : learning rate for neural net classifier
            training_iters (int)    : number of training iterations
            n_classes (int)         : number of classes
            train_momentum (float)  : momentum used in training
            n_visible (int)         : number of visible units
            mini_batch_size (int)   : size of training batches
            show_rbm_weights (bool) : display rbm weights in colour plot with same dimension of rbm_w

        Returns:
            FFNeuralNet : instance of feedforward neural network classifier
        """
        rbm = RBM(training_iters=training_iters, lr_rbm=lr_rbm, n_hid=n_hid, n_visible=n_visible,
                  train_momentum=train_momentum, mini_batch_size=mini_batch_size)
        rbm.train(self.data_sets['training'])
        if show_rbm_weights:
            self.show_rbm(rbm.rbm_w)
        return FFNeuralNet(training_iters=training_iters, rbm_w=rbm.rbm_w, lr_net=lr_net, n_hid=n_hid,
                           n_classes=n_classes, train_momentum=train_momentum, mini_batch_size=mini_batch_size)

    def a4_main(self, n_hid, lr_rbm, lr_classification, n_iterations, show_rbm_weights=False, train_momentum=0.9):
        """Runs training and computes error and loss of training, testing, and validation training sets.
        """
        if np.prod(np.shape(self.data_sets)) != 1:
            raise Exception('You must run a4_init before you do anything else.')
        nn = self.build_neural_net_model(n_hid=n_hid, lr_rbm=lr_rbm, lr_net=lr_classification,
                                         training_iters=n_iterations, n_classes=10,
                                         train_momentum=train_momentum, n_visible=256, mini_batch_size=100,
                                         show_rbm_weights=show_rbm_weights)
        nn.train(self.data_sets['training'])

        for data_name, data in self.data_sets.iteritems():
            nn.compute_error_and_loss(data, data_name=data_name)

    def train_rbm_test_cases(self, data_cases):
        """Runs training on given test case. Answer for question 6 and 7 when called with describe_matrix(..)
        """
        rbm = RBM()
        rbm.rbm_w = self.test_rbm_w
        rbm.fit(data_cases)
        return rbm.gradient

    def question_10(self):
        """Prints logarithm (base e) of partition function for small_test_rbm_w. Answer for question 10.
        """
        print "Log (base e) of partition function for small_test_rbm_w is :", self.partition_log(self.small_test_rbm_w)

    def show_rbm(self, rbm_w):
        """Display rbm weights in colour plot with same dimension of rbm_w.
        """
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

    @staticmethod
    def partition_log(w):
        """Computes logarithm (base e) of partition function for given size of rbm (hidden units)

        Notes:
        * Answer for question 10

        Args:
            w (numpy.array) : given rbm weight matrix

        Returns:
            float : log of partition function
        """
        dec_2_bin = lambda x, n_bits: np.array(["{0:b}".format(val).zfill(n_bits) for val in x])
        binary = np.array([list(val) for val in dec_2_bin(range(pow(2, np.size(w, 0))), np.size(w, 0))], dtype=float)
        return np.log(np.sum(np.prod((np.exp(np.dot(binary, w)) + 1).T, axis=0)))
