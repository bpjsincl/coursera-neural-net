"""Implements Assignment 2 for Geoffrey Hinton's Neural Networks Course offered through Coursera.

* Implements a basic framework for training neural nets with mini-batch gradient descent for a language model.
* Assignment covers hyperparameter search and observations through average cross entropy error.
    * i.e. number of training epochs, embedding and hidden layer size, training momentum

Abstracts classifiers developed in the course into, a more pythonic Sklearn framework. And cleans up a lot of the
given code.
"""
import time
import os

import numpy as np
from sklearn.base import BaseEstimator

from courseraneuralnet.utility.utils import zip_safe, loadmat

__all__ = ['EvaluateCrossEntropy',
           'NeuralNet',
           'load_data',
           'display_nearest_words',
           'word_distance',
           'A2Run'
           ]


def load_data(data, batch_size=100):
    """This method loads the training, validation and test set. It also divides the training set into mini-batches.

    Notes:
    -----
    * Subtract 1 from each index in `input` and `target` to fix matlab to python indexing

    Args:
        data (dict)         : From mat file.
        batch_size (int)    : Mini-batch size.

    Returns:
        dict: With keys `train`, `valid`, `test`, `vocab`
            train_input (numpy.array)   : An array of size d X n X m, where
                d: number of input dimensions (in this case, 3).
                n: size of each mini-batch (in this case, 100).
                m: number of minibatches.
            train_target (numpy.array)  : An array of size 1 X n X m.
            valid_input (numpy.array)   : An array of size D X number of points in the validation set.
            test (numpy.array)          : An array of size D X number of points in the test set.
            vocab (numpy.array)         : Vocabulary containing index to word mapping.
    """
    d = np.size(data['trainData'], 0) - 1
    m = int(np.size(data['trainData'], axis=1) / batch_size)

    sequences = {key: dict() for key in ['train', 'valid', 'test']}

    sequences['train']['input'] = np.reshape(data['trainData'][:d, :batch_size * m], (d, batch_size, m)) - 1
    sequences['train']['target'] = np.reshape(data['trainData'][d, :batch_size * m], (1, batch_size, m)) - 1
    sequences['valid']['input'] = data['validData'][:d, :] - 1
    sequences['valid']['target'] = data['validData'][d, :] - 1
    sequences['test']['input'] = data['testData'][:d, :] - 1
    sequences['test']['target'] = data['testData'][d, :] - 1
    sequences['vocab'] = data['vocab']

    return sequences


class NeuralNet(BaseEstimator):
    """Implements assignment 2 of Neural Networks for Machine Learning (Coursera) for Learning word representations.
    """
    def __init__(self,
                 epochs=1,
                 learning_rate=0.1,
                 momentum=0.9,
                 numhid1=50,
                 numhid2=200,
                 init_wt=0.01,
                 validation_ce_after=1000,
                 vocab_size=None,
                 num_words=None):
        """Initialize NeuralNet instance with training and visualization params.

        Args:
            epochs (int)                : Number of epochs to run.
            learning_rate (float)       : Learning rate.
            momentum (float)            : Momentum default.
            numhid1 (int)               : Dimensionality of embedding space.
            numhid2 (int)               : Number of units in hidden layer.
            init_wt (float)             : Standard deviation of the normal distribution which is sampled to
                                          get the initial weights
            validation_ce_after (int)   : Show cross-entropy calculation after specified samples during validation
            vocab_size (int)            : Length of vocabulary in dataset.
            num_words (int)             : Num words used in each training sample (given from dataset).
                                          In the assignment case, there's 3
        """
        assert vocab_size and num_words

        # Set Hyper params
        self.epochs = epochs
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.numhid1 = numhid1
        self.numhid2 = numhid2
        self.init_wt = init_wt
        self.show_validation_ce_after = validation_ce_after

        # INITIALIZE WEIGHTS AND BIASES
        self.word_embedding_weights = None
        self.embed_to_hid_weights = None
        self.hid_to_output_weights = None
        self.hid_bias = None
        self.output_bias = None

        self.word_embedding_weights_delta = None
        self.embed_to_hid_weights_delta = None
        self.hid_to_output_weights_delta = None
        self.hid_bias_delta = None
        self.output_bias_delta = None
        self.reset_classifier(vocab_size, num_words)

        # Initialize evaluation params
        self.tiny = np.exp(-30)
        self.batch_iteration = 0  # this is count in Matlab code
        self.trainset_ce = 0.0

    def reset_classifier(self, vocab_size, num_words):
        """Resets state of the classifier given vocab_size and num_words in dataset.
        """
        self.word_embedding_weights = self.init_wt * np.random.rand(vocab_size, self.numhid1)
        self.embed_to_hid_weights = self.init_wt * np.random.rand(num_words * self.numhid1, self.numhid2)
        self.hid_to_output_weights = self.init_wt * np.random.rand(self.numhid2, vocab_size)
        self.hid_bias = np.zeros((self.numhid2, 1))
        self.output_bias = np.zeros((vocab_size, 1))

        self.word_embedding_weights_delta = np.zeros((vocab_size, self.numhid1))
        self.embed_to_hid_weights_delta = np.zeros((num_words * self.numhid1, self.numhid2))
        self.hid_to_output_weights_delta = np.zeros((self.numhid2, vocab_size))
        self.hid_bias_delta = np.zeros((self.numhid2, 1))
        self.output_bias_delta = np.zeros((vocab_size, 1))

    def fit(self, X, y):
        """Fit model given matrix X and target y.

        Args:
            X   (numpy.ndarray) : input matrix
            y   (numpy.ndarray) : target matrix

        Returns:
            self (model) contains:
                word_embedding_weights
                embed_to_hid_weights
                hid_to_output_weights
                hid_bias
                output_bias
        """
        numwords, batch_size = np.shape(X)
        # FORWARD PROPAGATE.
        # Compute the state of each layer in the network given the input batch
        # and all weights and biases
        embedding_layer_state, hidden_layer_state, output_layer_state = self.fprop(X)
        assert all([all(row == False) for row in np.isnan(output_layer_state)])
        # COMPUTE DERIVATIVE.
        # Expand the target to a sparse 1-of-K vector.
        expanded_y = np.eye(self.vocab_size)[:, y]
        # Compute derivative of cross-entropy loss function.
        error_deriv = output_layer_state - expanded_y

        # MEASURE LOSS FUNCTION.
        ce = -sum(sum(np.multiply(expanded_y,
                                  np.log(output_layer_state + self.tiny)))) / float(batch_size)
        self.trainset_ce += (ce - self.trainset_ce) / float(self.batch_iteration)

        # BACK PROPAGATE.
        # OUTPUT LAYER.
        hid_to_output_weights_gradient = np.dot(hidden_layer_state, error_deriv.T)
        output_bias_gradient = np.column_stack(np.sum(error_deriv, axis=1)).T

        back_propagated_deriv_1 = np.multiply(np.multiply(np.dot(self.hid_to_output_weights, error_deriv),
                                                          hidden_layer_state), (1 - hidden_layer_state))

        # HIDDEN LAYER.
        embed_to_hid_weights_gradient = np.dot(embedding_layer_state, back_propagated_deriv_1.T)
        assert (self.numhid1 * numwords, self.numhid2) == embed_to_hid_weights_gradient.shape
        hid_bias_gradient = np.column_stack(np.sum(back_propagated_deriv_1, axis=1)).T
        assert (self.numhid2, 1) == hid_bias_gradient.shape
        back_propagated_deriv_2 = np.dot(self.embed_to_hid_weights, back_propagated_deriv_1)
        assert back_propagated_deriv_2.shape == (numwords * self.numhid1, batch_size)

        word_embedding_weights_gradient = np.zeros((self.vocab_size, self.numhid1))
        # EMBEDDING LAYER.
        for w in xrange(1, numwords):
            word_embedding_weights_gradient += np.dot(np.eye(self.vocab_size)[:, X[w, :]],
                                                      back_propagated_deriv_2[
                                                      (w - 1) * self.numhid1: w * self.numhid1, :].T)
        self.__update_weights_and_biases(batch_size, word_embedding_weights_gradient,
                                         embed_to_hid_weights_gradient, hid_to_output_weights_gradient,
                                         hid_bias_gradient, output_bias_gradient)
        return self

    def __update_weights_and_biases(self,
                                    batch_size,
                                    word_embedding_weights_gradient,
                                    embed_to_hid_weights_gradient,
                                    hid_to_output_weights_gradient,
                                    hid_bias_gradient,
                                    output_bias_gradient):
        """Update weights and biases
        """
        self.word_embedding_weights_delta = self.momentum * self.word_embedding_weights_delta + \
                                            word_embedding_weights_gradient / float(batch_size)
        self.word_embedding_weights -= self.learning_rate * self.word_embedding_weights_delta

        self.embed_to_hid_weights_delta = self.momentum * self.embed_to_hid_weights_delta + \
                                          embed_to_hid_weights_gradient / float(batch_size)
        self.embed_to_hid_weights -= self.learning_rate * self.embed_to_hid_weights_delta

        self.hid_to_output_weights_delta = self.momentum * self.hid_to_output_weights_delta + \
                                           hid_to_output_weights_gradient / float(batch_size)
        self.hid_to_output_weights -= self.learning_rate * self.hid_to_output_weights_delta

        self.hid_bias_delta = self.momentum * self.hid_bias_delta + hid_bias_gradient / float(batch_size)
        self.hid_bias -= self.learning_rate * self.hid_bias_delta

        self.output_bias_delta = self.momentum * self.output_bias_delta + output_bias_gradient / float(batch_size)
        self.output_bias -= self.learning_rate * self.output_bias_delta

    def train(self, sequences):
        """This function trains a neural network language model and validates as well. (These should be split up)

        Args:
            sequences (dict) : input data

        Returns:
            struct: contains the learned weights and biases and vocabulary.
        """
        self.reset_classifier(vocab_size=len(sequences['vocab']), num_words=len(sequences['train']['input']))
        for epoch in xrange(1, self.epochs + 1):
            print 'Epoch %d\n' % epoch
            self.trainset_ce = 0.0
            # LOOP OVER MINI-BATCHES.
            for m, (input_batch, target_batch) in enumerate(zip_safe(sequences['train']['input'].T,
                                                                     sequences['train']['target'].T)):
                self.batch_iteration += 1
                target_batch = target_batch.flatten()
                self.fit(input_batch.T, target_batch)

                # VALIDATE.
                if self.show_validation_ce_after and (m + 1) % self.show_validation_ce_after == 0:
                    print '\rRunning validation ... Validation CE after %d : %.3f' % \
                          (m + 1, EvaluateCrossEntropy(self).compute_ce(sequences['valid'], vocab_size=self.vocab_size))
            print '\rAverage Training CE : %.3f' % self.trainset_ce
        print 'Final Training CE : %.3f' % self.trainset_ce

    def fprop(self, input_batch):
        """This method forward propagates through a neural network.

        Args:
            input_batch (numpy.ndarray)                 : The input data as a matrix of size numwords X batchsize where,
                    *   numwords is the number of words.
                    *   batchsize is the number of data points.
                So, if input_batch(i, j) = k then the ith word in data point j is word index k of the vocabulary.

        Returns:
            tuple   :
                embedding_layer_state (numpy.ndarray)   : State of units in the embedding layer as a matrix of
                    size numhid1*numwords X batchsize
                hidden_layer_state (numpy.ndarray)      : State of units in the hidden layer as a matrix of
                    size numhid2 X batchsize
                output_layer_state (numpy.ndarray)      : State of units in the output layer as a matrix of size
                    vocab_size X batchsize
        """

        numwords, batch_size = np.shape(input_batch)
        vocab_size, numhid1 = np.shape(self.word_embedding_weights)
        numhid2 = np.size(self.embed_to_hid_weights, axis=1)

        # COMPUTE STATE OF WORD EMBEDDING LAYER.
        # Look up the inputs word indices in the word_embedding_weights matrix.
        embedding_layer_state = np.reshape(self.word_embedding_weights[input_batch.flatten()].T,
                                           (numhid1 * numwords, -1))
        # COMPUTE STATE OF HIDDEN LAYER.
        # Compute inputs to hidden units.
        inputs_to_hidden_units = np.dot(self.embed_to_hid_weights.T, embedding_layer_state) + np.tile(self.hid_bias,
                                                                                                      (1, batch_size))
        # Apply logistic activation function.
        hidden_layer_state = 1.0 / (1.0 + np.exp(-inputs_to_hidden_units))
        assert hidden_layer_state.shape == (numhid2, batch_size)

        # COMPUTE STATE OF OUTPUT LAYER.
        # Compute inputs to softmax.
        inputs_to_softmax = np.dot(self.hid_to_output_weights.T, hidden_layer_state) + \
                            np.tile(self.output_bias, (1, batch_size))
        assert inputs_to_softmax.shape == (vocab_size, batch_size)

        # Subtract maximum.
        inputs_to_softmax -= np.tile(np.max(inputs_to_softmax), (vocab_size, 1))

        # Compute exp.
        output_layer_state = np.exp(inputs_to_softmax)
        sum_output = np.sum(output_layer_state, axis=0)
        # correct for min float -- Matlab didn't have this problem (it must assume this instead of outputting 0.0)
        sum_output[np.where(sum_output == 0.0)] = np.finfo(float).min
        # Normalize to get probability distribution.
        output_layer_state = np.divide(output_layer_state, np.tile(sum_output, (vocab_size, 1)))

        return embedding_layer_state, hidden_layer_state, output_layer_state

    def predict_next_word(self, sentence, vocab, k):
        """Predicts the next word.
        Example usage:
            predict_next_word('john', 'might', 'be', 3)
            predict_next_word('life', 'in', 'new', 3)

        Args:
            sentence (iterable) : 3 word iterable containing
                word1 (str)         : The first word as a string.
                word2 (str)         : The second word as a string.
                word3 (str)         : The third word as a string.
            vocab (numpy.array) : vocabulary in model
            k (int)             : The k most probable predictions are shown.
        """
        input_ = np.array([np.where(vocab == word)[0] if np.where(vocab == word)[0] else [None] for word in sentence])
        for i, vocab_idx in enumerate(input_):
            if not vocab_idx:
                print 'Word %s not in vocabulary.\n' % sentence[i]
                return

        _, _, output_layer_state = self.fprop(input_)

        prob = np.sort(output_layer_state, axis=0)[::-1]
        indices = np.argsort(output_layer_state, axis=0)[::-1]
        for i in xrange(0, k):
            # noinspection PyStringFormat
            print '"%s %s %s %s" -- [Prob: %.5f]' % (sentence + (vocab[indices[i]][-1], prob[i]))


class EvaluateCrossEntropy(object):
    """Computes cross entropy given classifier model.
    """
    def __init__(self, estimator):
        """Initialize EvaluateCrossEntropy instance.
        """
        self.estimator = estimator

    def run_evaluation(self, sequences):
        # EVALUATE ON VALIDATION SET.
        print 'Running validation ... Final Validation CE : %.3f' % \
              self.compute_ce(sequences['valid'], vocab_size=len(sequences['vocab']))
        print 'Running test ... Final Test CE : %.3f' % \
              self.compute_ce(sequences['test'], vocab_size=len(sequences['vocab']))

    def compute_ce(self, data, vocab_size):
        """Compute Cross-Entropy

        Args:
            data (dict)     : Contains `input` and `target` keys each containing numpy.array
            vocab_size (int): Number of words in vocabulary.

        Returns:
            float : Cross-Entropy
        """
        embedding_layer_state, hidden_layer_state, output_layer_state = self.estimator.fprop(data['input'])
        datasetsize = np.size(data['input'], 1)
        expanded_target = np.eye(vocab_size)[:, data['target']]
        return -sum(sum(np.multiply(expanded_target, np.log(output_layer_state + np.exp(-30))))) / float(datasetsize)


def word_distance(word1, word2, model, vocab):
    """Shows the L2 distance between word1 and word2 in the word_embedding_weights.

    Example:
    -----
        word_distance('school', 'university', model, vocab)

    Args:
        word1 (str)         : The first word as a string.
        word2 (str)         : The second word as a string.
        model (NeuralNet)   : Model returned by estimator
        vocab (numpy.array) : vocabulary in model

    Return:
        distance
    """
    words = (word1, word2)
    idxs = np.array([np.where(vocab == word)[0][0] if np.where(vocab == word)[0] else None for word in words])
    for i, vocab_idx in enumerate(idxs):
        if not vocab_idx:
            print 'Word %s not in vocabulary.\n' % words[i]
            return
    diff = model.word_embedding_weights[idxs[0], :] - model.word_embedding_weights[idxs[1], :]
    return np.sqrt(sum(np.multiply(diff, diff)))


def display_nearest_words(word, model, k, vocab):
    """Shows the k-nearest words to the query word.
    Example:
    -----
      display_nearest_words('school', model, 10)

    Args:
        word (str)          : The query word as a string.
        model (NeuralNet)   : Model returned by estimator
        k (int)             : The number of nearest words to display.
        vocab (numpy.array) : vocabulary in model
    """
    idx = np.where(vocab == word)[0]
    if not idx:
        print 'Word %s not in vocabulary.\n' % word
        return

    # Compute distance to every other word.
    word_rep = model.word_embedding_weights[idx][-1]
    diff = model.word_embedding_weights - np.tile(word_rep, (len(vocab), 1))
    distance = np.sqrt(np.sum(np.multiply(diff, diff), axis=1))

    # Sort by distance.
    order = np.argsort(distance)
    order = order[1: k+1]  # The nearest word is the query word itself, skip that.
    for i in xrange(k):
        print 'Word\t: %s \nDistance: %.2f\n' % (vocab[order[i]], distance[order[i]])


class A2Run(object):
    """Runs assignment 2.
    """
    def __init__(self):
        """Initialize data set and all test cases for assignment.
        """
        data = loadmat(os.path.join(os.getcwd(), 'Data/data.mat'))
        self.data_sets = data['data']
        self.classifier = None

    def run_evaluation(self, **estimator_params):
        """Runs 4-gram Neural Network evaluation.

        Args:
            estimator_params (dict) : Contains parameters for NN. See NeuralNet(..)
        """
        start_time = time.time()
        sequences = load_data(self.data_sets, batch_size=100)
        self.classifier = NeuralNet(vocab_size=len(sequences['vocab']),
                                    num_words=len(sequences['train']['input']),
                                    **estimator_params)
        self.classifier.train(sequences)
        print 'Training took %.2f seconds\n', start_time - time.time()
        EvaluateCrossEntropy(self.classifier).run_evaluation(sequences)

    def a2_main(self, epochs=1, learning_rate=.10, momentum=0.9, numhid1=50, numhid2=200, init_wt=0.01,
                validation_ce_after=1000):
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
        self.run_evaluation(epochs=epochs,
                            learning_rate=learning_rate,
                            momentum=momentum,
                            numhid1=numhid1,
                            numhid2=numhid2,
                            init_wt=init_wt,
                            validation_ce_after=validation_ce_after)
