"""Implements Assignment 3 for Geoffrey Hinton's Neural Networks Course offered through Coursera.

* Trains a simple Feedforward Neural Network with Backpropogation, for recognizing USPS handwritten digits.
* Assignment looks into efficient optimization, and into effective regularization.
* Recognizes USPS handwritten digits.

Abstracts classifiers developed in the course into, a more pythonic Sklearn framework. And cleans up a lot of the
given code.
"""
import copy
import os

import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_array_equal
from sklearn.base import BaseEstimator

from courseraneuralnet.utility.utils import loadmat, logistic, log_sum_exp_over_rows

NUM_INPUT_UNITS = 256
NUM_CLASSES = 10

__all__ = ['A3Run']


class FFNeuralNet(BaseEstimator):
    """Implements Feedforward Neural Network from Assignment 4 trained with Backpropagation.
    """
    def __init__(self,
                 training_iters,
                 validation_data,
                 wd_coeff=None,
                 lr_net=0.02,
                 n_hid=300,
                 n_classes=10,
                 n_input_units=256,
                 train_momentum=0.9,
                 mini_batch_size=100,
                 early_stopping=False):
        """Initialize neural network.

        Args:
            training_iters (int)    : number of training iterations
            validation_data (dict)  : contains 'inputs' and 'targets' data matrices
            wd_coeff (float)        : weight decay coefficient
            lr_net (float)          : learning rate for neural net classifier
            n_hid (int)             : number of hidden units
            n_classes (int)         : number of classes
            train_momentum (float)  : momentum used in training
            mini_batch_size (int)   : size of training batches
            early_stopping (bool)   : saves model at validation error minimum
        """
        super(FFNeuralNet, self).__init__()

        self.n_classes = n_classes
        self.wd_coeff = wd_coeff
        self.batch_size = mini_batch_size
        self.lr_net = lr_net
        self.n_iterations = training_iters
        self.train_momentum = train_momentum
        self.early_stopping = early_stopping
        self.validation_data = validation_data  # used for early stopping

        # model result params
        self.training_data_losses = []
        self.validation_data_losses = []

        # Model params
        # We don't use random initialization, for this assignment. This way, everybody will get the same results.
        self.n_params = (n_input_units + n_classes) * n_hid
        theta = np.transpose(np.column_stack(np.cos(range(self.n_params)))) * 0.1 if self.n_params else np.array([])
        self.model = self.theta_to_model(theta)
        self.theta = self.model_to_theta(self.model)
        assert_array_equal(theta.flatten(), self.theta)
        self.momentum_speed = self.theta * 0.0

    def reset_classifier(self):
        """Resets the model parameters.
        """
        theta = np.transpose(np.column_stack(np.cos(range(self.n_params)))) * 0.1 if self.n_params else np.array([])
        self.model = self.theta_to_model(theta)
        self.theta = self.model_to_theta(self.model)
        self.momentum_speed = self.theta * 0.0

    @staticmethod
    def model_to_theta(model):
        """Takes a model (or gradient in model form), and turns it into one long vector. See also theta_to_model."""
        model_copy = copy.deepcopy(model)
        return np.hstack((model_copy['inputToHid'].flatten(), model_copy['hidToClass'].flatten()))

    @staticmethod
    def theta_to_model(theta):
        """Takes a model (or gradient) in the form of one long vector (maybe produced by model_to_theta),
        and restores it to the structure format, i.e. with fields .input_to_hid and .hid_to_class, both matrices.
        """
        n_hid = np.size(theta, 0) / (NUM_INPUT_UNITS + NUM_CLASSES)
        return {'inputToHid': np.reshape(theta[:NUM_INPUT_UNITS * n_hid], (n_hid, NUM_INPUT_UNITS)),
                'hidToClass': np.reshape(theta[NUM_INPUT_UNITS * n_hid: np.size(theta, 0)], (NUM_CLASSES, n_hid))}

    def fit(self, X, y):
        """Fit a model using Classification gradient descent.
        """
        self._d_loss_by_d_model(inputs=X, targets=y)
        return self

    def train(self, sequences):
        """Implements optimize(..) from assignment. This trains using gradient descent with momentum.

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
        if self.early_stopping:
            best_so_far = dict()
            best_so_far['theta'] = None
            best_so_far['validationLoss'] = np.inf
            best_so_far['afterNIters'] = None

        n_training_cases = np.size(sequences['inputs'], 1)
        for i in xrange(self.n_iterations):
            training_batch_start = (i * self.batch_size) % n_training_cases
            training_batch_x = sequences['inputs'][:, training_batch_start: training_batch_start + self.batch_size]
            training_batch_y = sequences['targets'][:, training_batch_start: training_batch_start + self.batch_size]

            self.fit(training_batch_x, training_batch_y)
            self.momentum_speed = self.momentum_speed * self.train_momentum - self.gradient
            self.theta += self.momentum_speed * self.lr_net
            self.model = self.theta_to_model(self.theta)

            self.training_data_losses += [self.loss(sequences)]
            self.validation_data_losses += [self.loss(self.validation_data)]
            if self.early_stopping and self.validation_data_losses[-1] < best_so_far['validationLoss']:
                best_so_far['theta'] = copy.deepcopy(self.theta)  # deepcopy avoids memory reference bug
                best_so_far['validationLoss'] = self.validation_data_losses[-1]
                best_so_far['afterNIters'] = i

            if np.mod(i, round(self.n_iterations / float(self.n_classes))) == 0:
                print 'After {0} optimization iterations, training data loss is {1}, and validation data ' \
                      'loss is {2}'.format(i, self.training_data_losses[-1], self.validation_data_losses[-1])

            # check gradient again, this time with more typical parameters and with a different data size
            if i == self.n_iterations:
                print 'Now testing the gradient on just a mini-batch instead of the whole training set... '
                training_batch = {'inputs': training_batch_x, 'targets': training_batch_y}
                self.test_gradient(training_batch)

        if self.early_stopping:
            print 'Early stopping: validation loss was lowest after {0} iterations. ' \
                  'We chose the model that we had then.'.format(best_so_far['afterNIters'])
            self.theta = copy.deepcopy(best_so_far['theta'])  # deepcopy avoids memory reference bug

    def predict(self, x_sequences):
        """Predict a specific class from a given set of sequences.
        """
        return np.argmax(self.predict_sequences_proba(x_sequences=x_sequences), axis=0)

    def predict_sequences_proba(self, x_sequences):
        """Predict the probability of each class in a given set of sequences.

        Returns:
            (numpy.array) : class input (size: <number of classes> by <number of data cases>)
        """
        return self.predict_proba(x_sequences['inputs'])

    def predict_proba(self, inputs):
        """Predict the probability of each class given data inputs.

        Returns:
            (numpy.array) : probability of classes
        """
        hid_input = np.dot(self.model['inputToHid'], inputs)
        hid_output = logistic(hid_input)  # size: <number of hidden units> by <number of data cases>
        return np.dot(self.model['hidToClass'], hid_output)

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

    def loss(self, data):
        """Evaluate loss of given data set.

        Args:
            data (dict):
                    - 'inputs' is a matrix of size <number of inputs i.e. NUM_INPUT_UNITS> by <number of data cases>
                       Each column describes a different data case.
                    - 'targets' is a matrix of size <number of classes i.e. NUM_CLASSES> by <number of data cases>
                       Each column describes a different data case. It contains a one-of-N encoding of the class,
                       i.e. one element in every column is 1 and the others are 0.

        Returns:
            float : loss of model
        """
        log_class_prob = self.predict_sequences_log_proba(data)
        # select the right log class probability using that sum then take the mean over all data cases.
        classification_loss = -np.mean(sum(log_class_prob * data['targets'], 0))
        # weight decay loss. very straightforward: E = 1/2 * wd_coeffecient * theta^2
        wd_loss = sum(self.model_to_theta(self.model) ** 2) / 2.0 * self.wd_coeff
        return classification_loss + wd_loss

    def _d_loss_by_d_model(self, inputs, targets):
        """Compute derivative of loss.
        Args:
            data (dict):
                    - 'inputs' is a matrix of size <number of inputs i.e. NUM_INPUT_UNITS> by <number of data cases>
                    - 'targets' is a matrix of size <number of classes i.e. NUM_CLASSES> by <number of data cases>

        Returns:
            dict:   The returned object is supposed to be exactly like parameter <model>,
                    i.e. it has fields ret['inputToHid'] and ret['hidToClass'].
                    However, the contents of those matrices are gradients (d loss by d model parameter),
                    instead of model parameters.
        """
        ret_model = dict()

        hid_input = np.dot(self.model['inputToHid'], inputs)
        hid_output = logistic(hid_input)
        class_input = np.dot(self.model['hidToClass'], hid_output)
        class_prob = np.exp(self.predict_log_proba(class_input))

        # weight decay loss. very straightforward: E = 1/2 * wd_coeffecient * theta^2
        error_deriv = class_prob - targets
        hid_to_output_weights_gradient = np.dot(hid_output, error_deriv.T) / float(np.size(hid_output, axis=1))
        ret_model['hidToClass'] = hid_to_output_weights_gradient.T

        backpropagate_error_deriv = np.dot(self.model['hidToClass'].T, error_deriv)
        input_to_hidden_weights_gradient = np.dot(inputs, ((1.0 - hid_output) * hid_output *
                                                           backpropagate_error_deriv).T) / float(np.size(hid_output,
                                                                                                         axis=1))
        ret_model['inputToHid'] = input_to_hidden_weights_gradient.T

        ret_model['inputToHid'] += self.model['inputToHid'] * self.wd_coeff
        ret_model['hidToClass'] += self.model['hidToClass'] * self.wd_coeff
        self.gradient = self.model_to_theta(ret_model)

    def classification_performance(self, data):
        """This returns the fraction of data cases that is incorrectly classified by the model.
        """
        return np.mean(np.array(self.predict(data) != np.argmax(data['targets'], axis=0), dtype=float))

    def test_gradient(self, data):
        """Test the gradient using a finite difference approximation to the gradient.

        Notes:
        *  If that finite difference approximation results in an approximate gradient that's very different
          from the computed gradient produced, then the program prints an error message.
        """
        base_theta = self.model_to_theta(self.model)
        h = 1e-2
        correctness_threshold = 1e-5
        self._d_loss_by_d_model(data['inputs'], data['targets'])
        analytic_gradient_struct = self.theta_to_model(self.gradient)
        if np.size(analytic_gradient_struct.keys(), 0) != 2:
            raise Exception('The object returned by def d_loss_by_d_model should have exactly two field names: '
                            '.input_to_hid and .hid_to_class')

        if np.size(analytic_gradient_struct['inputToHid']) != np.size(self.model['inputToHid']):
            raise Exception('The size of .input_to_hid of the return value of d_loss_by_d_model (currently {0}) '
                            'should be same as the size of model[\'inputToHid\'] '
                            '(currently {1})'.format(np.size(analytic_gradient_struct['inputToHid']),
                                                     np.size(self.model['inputToHid'])))

        if np.size(analytic_gradient_struct['hidToClass']) != np.size(self.model['hidToClass']):
            raise Exception('The size of .hid_to_class of the return value of d_loss_by_d_model (currently {0}) '
                            'should be same as the size of model[\'hidToClass\'] '
                            '(currently {1})'.format(np.size(analytic_gradient_struct['hidToClass']),
                                                     np.size(self.model['hidToClass'])))

        analytic_gradient = self.gradient
        if any(np.isnan(analytic_gradient)) or any(np.isinf(analytic_gradient)):
            raise Exception('Your gradient computation produced a NaN or infinity. That is an error.')

        # We want to test the gradient not for every element of theta, because that's a lot of work.
        # Instead, we test for only a few elements. If there's an error, this is probably enough to find that error.
        input_to_hid_theta_size = np.prod(np.size(self.model['inputToHid']))
        hid_to_class_theta_size = np.prod(np.size(self.model['hidToClass']))
        big_prime = 1299721  # 1299721 is prime and thus ensures a somewhat random-like selection of indices.
        hid_to_class_indices_to_check = np.mod(big_prime * np.array(range(20)), hid_to_class_theta_size) + \
                                        input_to_hid_theta_size
        input_to_hid_indices_to_check = np.mod(big_prime * np.array(range(80)), input_to_hid_theta_size)
        indices_to_check = np.hstack((hid_to_class_indices_to_check, input_to_hid_indices_to_check))
        for i, test_index in enumerate(indices_to_check):
            analytic_here = analytic_gradient[test_index]
            theta_step = base_theta * 0.0
            theta_step[test_index] = h
            contribution_distances = range(-4, 0) + range(1, 5)
            contribution_weights = [1./280, -4./105, 1./5, -4./5, 4./5, -1./5, 4./105, -1./280]
            temp = 0.
            for distance, weight in zip(contribution_distances, contribution_weights):
                self.model = self.theta_to_model(base_theta + theta_step * distance)  # temporarily update model
                temp += self.loss(data) * weight
            fd_here = temp / h
            diff = abs(analytic_here - fd_here)
            if diff > correctness_threshold and diff / float(abs(analytic_here) + abs(fd_here)) > correctness_threshold:
                part_names = ['inputToHid', 'hidToClass']
                raise Exception('Theta element #{0} (part of {1}), with value {2}, has finite difference gradient {3} '
                                'but analytic gradient {4}. That looks like an error.'.format(test_index,
                                                                                              part_names[i <= 19],
                                                                                              base_theta[test_index],
                                                                                              fd_here,
                                                                                              analytic_here))

            if i == 19:
                print 'Gradient test passed for hid_to_class.'
            if i == 99:
                print 'Gradient test passed for input_to_hid.'

        self.model = self.theta_to_model(base_theta)  # make sure model is reset back to initial
        print 'Gradient test passed. That means that the gradient that your code computed is within 0.001%% of the ' \
              'gradient that the finite difference approximation computed, so the gradient calculation procedure is ' \
              'probably correct (not certainly, but probably).'


class A3Run(object):
    """Runs assignment 3.
    """
    def __init__(self):
        """Initialize data set and all test cases for assignment.
        """
        data = loadmat(os.path.join(os.getcwd(), 'Data/data.mat'))
        self.data_sets = data['data']

    def a3_main(self, wd_coeff, n_hid, n_iterations, lr_net, train_momentum=0.9,
                early_stopping=False, mini_batch_size=100):
        """Runs training and computes error and loss of training, testing, and validation training sets.

        Args:
            wd_coeff (float)        : weight decay coefficient
            n_hid (int)             : number of hidden units
            n_iterations (int)      : number of training iterations
            lr_net (float)          : learning rate for neural net classifier
            train_momentum (float)  : momentum used in training
            early_stopping (bool)   : saves model at validation error minimum
            mini_batch_size (int)   : size of training batches
        """
        nn = FFNeuralNet(training_iters=n_iterations, validation_data=self.data_sets['validation'],
                         wd_coeff=wd_coeff, lr_net=lr_net, n_hid=n_hid,
                         n_classes=10, n_input_units=256, train_momentum=train_momentum,
                         mini_batch_size=mini_batch_size, early_stopping=early_stopping)

        if n_iterations != 0:
            print 'Now testing the gradient on the whole training set... '
            nn.test_gradient(self.data_sets['training'])
            nn.train(self.data_sets['training'])

        # the optimization is finished. Now do some reporting.
        if n_iterations != 0 and nn.training_data_losses and nn.validation_data_losses:
            plt.hold(True)
            plt.plot(nn.training_data_losses, 'b')
            plt.plot(nn.validation_data_losses, 'r')
            plt.legend(['training', 'validation'])
            plt.ylabel('loss')
            plt.xlabel('iteration number')
            plt.hold(False)

        for data_name, data_segment in self.data_sets.iteritems():
            print 'The loss on the {0} data is {1}'.format(data_name, nn.loss(data_segment))
            if wd_coeff != 0:
                nn.wd_coeff = 0
                print 'The classification loss (i.e. without weight decay) ' \
                      'on the {0} data is {1}'.format(data_name, nn.loss(data_segment))
            print 'The classification error rate ' \
                  'on the {0} data is {1}'.format(data_name, nn.classification_performance(data_segment))
