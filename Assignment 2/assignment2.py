import time
import scipy.io as sio

import numpy as np

__all__ = ['NeuralNet',
           'load_data',
           'loadmat',
           'display_nearest_words',
           'predict_next_word',
           'word_distance'
           ]


# noinspection PyUnresolvedReferences
class NeuralNet(object):
    """Implements assignment 2 of Neural Networks for Machine Learning (Coursera) for Learning word representations.
    """
    def __init__(self,
                 epochs=1,
                 batch_size=100,
                 learning_rate=0.1,
                 momentum=0.9,
                 numhid1=50,
                 numhid2=200,
                 init_wt=0.01,
                 show_training_ce_after=100,
                 show_validation_ce_after=1000):
        """Initialize NeuralNet instance with training and visualization params.

        Args:
            epochs (int)            : Number of epochs to run.
            batchsize (int)         : Mini-batch size.
            learning_rate (float)   : Learning rate default = 0.1.
            momentum (float)        : Momentum default = 0.9.
            numhid1 (int)           : Dimensionality of embedding space default = 50.
            numhid2 (int)           : Number of units in hidden layer default = 200.
            init_wt (float)         : Standard deviation of the normal distribution which is sampled to
                get the initial weights default = 0.01
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.numhid1 = numhid1
        self.numhid2 = numhid2
        self.init_wt = init_wt
        self.show_training_ce_after = show_training_ce_after
        self.show_validation_ce_after = show_validation_ce_after

    def train(self, data):
        """This function trains a neural network language model.

        Args:
            data (dict) : input data

        Returns:
            struct: contains the learned weights and biases and vocabulary.
        """
        start_time = time.time()

        # LOAD DATA.
        train_input, train_target, valid_input, valid_target, test_input, test_target, vocab = \
            load_data(data, n=self.batch_size)
        numwords, batchsize, numbatches = np.shape(train_input)
        assert batchsize == self.batch_size
        vocab_size = len(vocab)

        # INITIALIZE WEIGHTS AND BIASES.
        word_embedding_weights = self.init_wt * np.random.rand(vocab_size, self.numhid1)
        embed_to_hid_weights = self.init_wt * np.random.rand(numwords * self.numhid1, self.numhid2)
        hid_to_output_weights = self.init_wt * np.random.rand(self.numhid2, vocab_size)
        hid_bias = np.zeros((self.numhid2, 1))
        output_bias = np.zeros((vocab_size, 1))

        word_embedding_weights_delta = np.zeros((vocab_size, self.numhid1))
        word_embedding_weights_gradient = np.zeros((vocab_size, self.numhid1))
        embed_to_hid_weights_delta = np.zeros((numwords * self.numhid1, self.numhid2))
        hid_to_output_weights_delta = np.zeros((self.numhid2, vocab_size))
        hid_bias_delta = np.zeros((self.numhid2, 1))
        output_bias_delta = np.zeros((vocab_size, 1))
        expansion_matrix = np.eye(vocab_size)

        count = 0
        tiny = np.exp(-30)
        trainset_ce = 0.0

        # TRAIN.
        for epoch in xrange(1, self.epochs + 1):
            print 'Epoch %d\n' % epoch
            this_chunk_ce = 0.0
            trainset_ce = 0.0
            # LOOP OVER MINI-BATCHES.
            for m in xrange(numbatches):
                input_batch = train_input[:, :, m]
                target_batch = train_target[:, :, m][-1]

                # FORWARD PROPAGATE.
                # Compute the state of each layer in the network given the input batch
                # and all weights and biases
                embedding_layer_state, hidden_layer_state, output_layer_state = self.fprop(input_batch,
                                                                                           word_embedding_weights,
                                                                                           embed_to_hid_weights,
                                                                                           hid_to_output_weights,
                                                                                           hid_bias,
                                                                                           output_bias)
                assert all([all(row == False) for row in np.isnan(output_layer_state)])
                # assert all(np.isnan(output_layer_state))
                # COMPUTE DERIVATIVE.
                # Expand the target to a sparse 1-of-K vector.
                expanded_target_batch = expansion_matrix[:, target_batch]
                # Compute derivative of cross-entropy loss function.
                error_deriv = output_layer_state - expanded_target_batch
                # MEASURE LOSS FUNCTION.
                ce = -sum(sum(np.multiply(expanded_target_batch,
                                          np.log(output_layer_state + tiny)))) / float(self.batch_size)
                # if any([any(row <= 0.0) for row in (output_layer_state + tiny)]):
                #     print (output_layer_state + tiny)
                #     print
                count += 1
                this_chunk_ce += (ce - this_chunk_ce) / float(count)
                trainset_ce += (ce - trainset_ce) / float(m + 1)
                if (m + 1) % self.show_training_ce_after == 0:
                    print '\rBatch %d Train CE %.3f' % (m + 1, this_chunk_ce)
                    count = 0
                    this_chunk_ce = 0.0

                # BACK PROPAGATE.
                # OUTPUT LAYER.
                hid_to_output_weights_gradient = np.dot(hidden_layer_state, error_deriv.T)
                output_bias_gradient = np.column_stack(np.sum(error_deriv, axis=1)).T

                back_propagated_deriv_1 = np.multiply(np.multiply(np.dot(hid_to_output_weights, error_deriv),
                                                                  hidden_layer_state), (1 - hidden_layer_state))

                # HIDDEN LAYER.
                embed_to_hid_weights_gradient = np.dot(embedding_layer_state, back_propagated_deriv_1.T)
                assert (self.numhid1 * numwords, self.numhid2) == embed_to_hid_weights_gradient.shape

                hid_bias_gradient = np.column_stack(np.sum(back_propagated_deriv_1, axis=1)).T
                assert (self.numhid2, 1) == hid_bias_gradient.shape

                back_propagated_deriv_2 = np.dot(embed_to_hid_weights, back_propagated_deriv_1)
                assert back_propagated_deriv_2.shape == (numwords * self.numhid1, self.batch_size)

                word_embedding_weights_gradient[:] = 0.0
                # EMBEDDING LAYER.
                for w in xrange(1, numwords):
                    word_embedding_weights_gradient += np.dot(expansion_matrix[:, input_batch[w, :]],
                                                              back_propagated_deriv_2[
                                                              (w - 1) * self.numhid1: w * self.numhid1, :].T)
                # UPDATE WEIGHTS AND BIASES.
                word_embedding_weights_delta = self.momentum * word_embedding_weights_delta + \
                                               word_embedding_weights_gradient / float(self.batch_size)
                word_embedding_weights -= self.learning_rate * word_embedding_weights_delta

                embed_to_hid_weights_delta = self.momentum * embed_to_hid_weights_delta + \
                                             embed_to_hid_weights_gradient / float(self.batch_size)
                embed_to_hid_weights -= self.learning_rate * embed_to_hid_weights_delta

                hid_to_output_weights_delta = self.momentum * hid_to_output_weights_delta + \
                                              hid_to_output_weights_gradient / float(self.batch_size)
                hid_to_output_weights -= self.learning_rate * hid_to_output_weights_delta

                hid_bias_delta = self.momentum * hid_bias_delta + hid_bias_gradient / float(self.batch_size)
                hid_bias -= self.learning_rate * hid_bias_delta

                output_bias_delta = self.momentum * output_bias_delta + output_bias_gradient / float(self.batch_size)
                output_bias -= self.learning_rate * output_bias_delta

                # VALIDATE.
                if (m + 1) % self.show_validation_ce_after == 0:
                    print '\rRunning validation ...'
                    embedding_layer_state, hidden_layer_state, output_layer_state = self.fprop(valid_input,
                                                                                               word_embedding_weights,
                                                                                               embed_to_hid_weights,
                                                                                               hid_to_output_weights,
                                                                                               hid_bias,
                                                                                               output_bias)
                    datasetsize = np.size(valid_input, 1)
                    expanded_valid_target = expansion_matrix[:, valid_target]
                    ce = -sum(sum(np.multiply(expanded_valid_target,
                                              np.log(output_layer_state + tiny)))) / float(datasetsize)
                    print 'Validation CE %.3f\n' % ce
            print '\rAverage Training CE %.3f\n' % trainset_ce
        print 'Finished Training.\n'
        print 'Final Training CE %.3f\n' % trainset_ce

        # EVALUATE ON VALIDATION SET.
        print '\rRunning validation ...'
        embedding_layer_state, hidden_layer_state, output_layer_state = self.fprop(valid_input,
                                                                                   word_embedding_weights,
                                                                                   embed_to_hid_weights,
                                                                                   hid_to_output_weights,
                                                                                   hid_bias,
                                                                                   output_bias)
        datasetsize = np.size(valid_input, 1)
        expanded_valid_target = expansion_matrix[:, valid_target]
        ce = -sum(sum(np.multiply(expanded_valid_target, np.log(output_layer_state + tiny)))) / float(datasetsize)
        print '\rFinal Validation CE %.3f\n' % ce

        # EVALUATE ON TEST SET.
        print '\rRunning test ...'
        embedding_layer_state, hidden_layer_state, output_layer_state = self.fprop(test_input,
                                                                                   word_embedding_weights,
                                                                                   embed_to_hid_weights,
                                                                                   hid_to_output_weights,
                                                                                   hid_bias,
                                                                                   output_bias)
        datasetsize = np.size(test_input, 1)
        expanded_test_target = expansion_matrix[:, test_target]
        ce = -sum(sum(np.multiply(expanded_test_target, np.log(output_layer_state + tiny)))) / float(datasetsize)
        print '\rFinal Test CE %.3f\n' % ce

        model = dict()
        model['wordEmbeddingWeights'] = word_embedding_weights
        model['embededToHidWeights'] = embed_to_hid_weights
        model['hidToOutputWeights'] = hid_to_output_weights
        model['hidBias'] = hid_bias
        model['outputBias'] = output_bias
        model['vocab'] = vocab

        print 'Training took %.2f seconds\n' % (time.time() - start_time)

        return model

    @staticmethod
    def fprop(input_batch, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias,
              output_bias):
        """This method forward propagates through a neural network.

        Args:
            input_batch (numpy.ndarray)             : The input data as a matrix of size numwords X batchsize where,
                    *   numwords is the number of words.
                    *   batchsize is the number of data points.
                So, if input_batch(i, j) = k then the ith word in data point j is word index k of the vocabulary.
            word_embedding_weights (numpy.ndarray)  : Word embedding as a matrix of size vocab_size X numhid1, where
                    *   vocab_size is the size of the vocabulary.
                    *   numhid1 is the dimensionality of the embedding space.
            embed_to_hid_weights (numpy.ndarray)    : Weights between the word embedding layer and hidden
                layer as a matrix of soze numhid1*numwords X numhid2, numhid2 is the number of hidden units.
            hid_to_output_weights (numpy.ndarray)   : Weights between the hidden layer and output softmax
                unit as a matrix of size numhid2 X vocab_size
            hid_bias (numpy.ndarray)                : Bias of the hidden layer as a matrix of size numhid2 X 1.
            output_bias (numpy.ndarray)             : Bias of the output layer as a matrix of size vocab_size X 1.

        Returns:
            tuple   :
                embedding_layer_state (numpy.ndarray)   : State of units in the embedding layer as a matrix of
                    size numhid1*numwords X batchsize
                hidden_layer_state (numpy.ndarray)      : State of units in the hidden layer as a matrix of
                    size numhid2 X batchsize
                output_layer_state (numpy.ndarray)      : State of units in the output layer as a matrix of size
                    vocab_size X batchsize
        """

        numwords, batchsize = np.shape(input_batch)
        vocab_size, numhid1 = np.shape(word_embedding_weights)
        numhid2 = np.size(embed_to_hid_weights, axis=1)

        # COMPUTE STATE OF WORD EMBEDDING LAYER.
        # Look up the inputs word indices in the word_embedding_weights matrix.
        embedding_layer_state = np.reshape(word_embedding_weights[input_batch.flatten()].T, (numhid1 * numwords, -1))
        # COMPUTE STATE OF HIDDEN LAYER.
        # Compute inputs to hidden units.
        inputs_to_hidden_units = np.dot(embed_to_hid_weights.T, embedding_layer_state) + np.tile(hid_bias,
                                                                                                 (1, batchsize))
        # Apply logistic activation function.
        hidden_layer_state = 1.0 / (1.0 + np.exp(-inputs_to_hidden_units))
        assert hidden_layer_state.shape == (numhid2, batchsize)

        # COMPUTE STATE OF OUTPUT LAYER.
        # Compute inputs to softmax.
        inputs_to_softmax = np.dot(hid_to_output_weights.T, hidden_layer_state) + np.tile(output_bias, (1, batchsize))
        assert inputs_to_softmax.shape == (vocab_size, batchsize)

        # Subtract maximum.
        # Remember that adding or subtracting the same constant from each input to a
        # softmax unit does not affect the outputs. Here we are subtracting maximum to
        # make all inputs <= 0. This prevents overflows when computing their
        # exponents.
        inputs_to_softmax -= np.tile(np.max(inputs_to_softmax), (vocab_size, 1))

        # Compute exp.
        output_layer_state = np.exp(inputs_to_softmax)
        sum_output = np.sum(output_layer_state, axis=0)
        # correct for min float64 -- Matlab didn't have this problem (it must assume this instead of outputting 0.0)
        sum_output = np.array([1e-308 if val == 0.0 else val for val in sum_output], dtype='float64')
        # Normalize to get probability distribution.
        output_layer_state = np.divide(output_layer_state, np.tile(sum_output, (vocab_size, 1)))

        return embedding_layer_state, hidden_layer_state, output_layer_state


def load_data(data, n=100):
    """This method loads the training, validation and test set. It also divides the training set into mini-batches.

    Notes:
    -----
    * Subtract 1 from each index in `input` and `target` to fix matlab to python indexing

    Args:
        data (dict) : From mat file.
        n (int)     : Mini-batch size.

    Returns:
        train_input: An array of size d X n X m, where
                    d: number of input dimensions (in this case, 3).
                    n: size of each mini-batch (in this case, 100).
                    m: number of minibatches.
        train_target: An larray of size 1 X n X m.
        valid_input: An array of size D X number of points in the validation set.
        test: An array of size D X number of points in the test set.
        vocab: Vocabulary containing index to word mapping.
    """
    d = np.size(data['trainData'], 0) - 1
    m = int(np.size(data['trainData'], axis=1) / n)

    train_input = np.reshape(data['trainData'][:d, :n * m], (d, n, m)) - 1
    train_target = np.reshape(data['trainData'][d, :n * m], (1, n, m)) - 1
    valid_input = data['validData'][:d, :] - 1
    valid_target = data['validData'][d, :] - 1
    test_input = data['testData'][:d, :] - 1
    test_target = data['testData'][d, :] - 1
    vocab = data['vocab']

    return train_input, train_target, valid_input, valid_target, test_input, test_target, vocab


def loadmat(filename):
    '''
    This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects.
    '''
    return _check_keys(sio.loadmat(filename, struct_as_record=False, squeeze_me=True))


def _check_keys(data):
    '''
    Checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in data:
        if isinstance(data[key], sio.matlab.mio5_params.mat_struct):
            data[key] = _todict(data[key])
    return data


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    data = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            data[strg] = _todict(elem)
        else:
            data[strg] = elem
    return data


def word_distance(word1, word2, model):
    """Shows the L2 distance between word1 and word2 in the word_embedding_weights.
    Example usage:
        word_distance('school', 'university', model)

    Args:
        word1: The first word as a string.
        word2: The second word as a string.
        model: Model returned by the training script.

    Return:
        distance
    """

    word_embedding_weights = model['wordEmbeddingWeights']
    vocab = model['vocab']
    id1 = np.where(vocab == word1)
    id2 = np.where(vocab == word2)

    if not any(id1):
        print 'Word %s not in vocabulary.\n' % word1
        return 0.0

    if not any(id2):
        print 'Word %s not in vocabulary.\n' % word2
        return 0.0
    word_rep1 = word_embedding_weights[id1][-1]
    word_rep2 = word_embedding_weights[id2][-1]
    diff = word_rep1 - word_rep2
    return np.sqrt(sum(np.multiply(diff, diff)))


def display_nearest_words(word, model, k):
    """Shows the k-nearest words to the query word.
    Inputs:
      word: The query word as a string.
      model: Model returned by the training script.
      k: The number of nearest words to display.
    Example usage:
      display_nearest_words('school', model, 10)
    """

    word_embedding_weights = model['wordEmbeddingWeights']
    vocab = model['vocab']
    idx = np.where(vocab == word)

    if not any(idx):
        print 'Word %s not in vocabulary.\n' % word
        return
    # Compute distance to every other word.
    vocab_size = len(vocab)
    word_rep = word_embedding_weights[idx][-1]
    diff = word_embedding_weights - np.tile(word_rep, (vocab_size, 1))
    distance = np.sqrt(np.sum(np.multiply(diff, diff), axis=1))
    # Sort by distance.
    order = np.argsort(distance)
    order = order[1: k+1]  # The nearest word is the query word itself, skip that.
    for i in xrange(k):
        print 'Word\t: %s \nDistance: %.2f\n' % (vocab[order[i]], distance[order[i]])


def predict_next_word(word1, word2, word3, model, k):
    """Predicts the next word.
    Example usage:
        predict_next_word('john', 'might', 'be', model, 3)
        predict_next_word('life', 'in', 'new', model, 3)

    Args:
        word1: The first word as a string.
        word2: The second word as a string.
        word3: The third word as a string.
        model: Model returned by the training script.
        k: The k most probable predictions are shown.
    """
    vocab = model['vocab']
    id1 = np.where(vocab == word1)[0]
    id2 = np.where(vocab == word2)[0]
    id3 = np.where(vocab == word3)[0]

    if not id1:
        print 'Word %s not in vocabulary.\n' % word1
        return -1
    if not id2:
        print 'Word %s not in vocabulary.\n' % word2
        return -1
    if not id3:
        print 'Word %s not in vocabulary.\n' % word3
        return -1

    input_ = np.array([id1, id2, id3])
    embedding_layer_state, hidden_layer_state, output_layer_state = NeuralNet.fprop(input_,
                                                                                    model['wordEmbeddingWeights'],
                                                                                    model['embededToHidWeights'],
                                                                                    model['hidToOutputWeights'],
                                                                                    model['hidBias'],
                                                                                    model['outputBias'])

    prob = np.sort(output_layer_state, axis=0)[::-1]
    indices = np.argsort(output_layer_state, axis=0)[::-1]
    for i in xrange(0, k):
        print '"%s %s %s %s" -- [Prob: %.5f]' % (word1, word2, word3, vocab[indices[i]][-1], prob[i])
