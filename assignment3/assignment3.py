import numpy as np
import copy
from numpy.testing import assert_array_equal, assert_allclose
import matplotlib.pyplot as plt

NUM_INPUT_UNITS = 256
NUM_CLASSES = 10

__all__ = ['a3',
           'classification_performance',
           'initial_model',
           'logistic',
           'log_sum_exp_over_rows',
           'loss',
           'model_to_theta',
           'theta_to_model',
           'test_gradient',
           ]


def a3(wd_coefficient, n_hid, n_iters, learning_rate, momentum_multiplier, do_early_stopping,
       mini_batch_size, data):
    model = initial_model(n_hid)
    training_batch = dict()
    n_training_cases = np.size(data['training']['inputs'], 1)
    if n_iters != 0:
        print 'Now testing the gradient on the whole training set... '
        test_gradient(model, data['training'], wd_coefficient)

    # optimization
    theta = model_to_theta(model)
    momentum_speed = theta * 0.0
    training_data_losses = []
    validation_data_losses = []
    if do_early_stopping:
        best_so_far = dict()
        best_so_far['theta'] = None  # this will be overwritten soon
        best_so_far['validationLoss'] = np.inf
        best_so_far['afterNIters'] = None

    for optimization_iteration_i in xrange(n_iters):
        model = theta_to_model(theta)

        training_batch_start = np.mod((optimization_iteration_i) * mini_batch_size, n_training_cases)
        training_batch['inputs'] = data['training']['inputs'][:, training_batch_start: training_batch_start +
                                                                                        mini_batch_size]
        training_batch['targets'] = data['training']['targets'][:, training_batch_start: training_batch_start +
                                                                                          mini_batch_size]
        gradient = model_to_theta(d_loss_by_d_model(model, training_batch, wd_coefficient))
        momentum_speed = momentum_speed * momentum_multiplier - gradient
        theta += momentum_speed * learning_rate

        model = theta_to_model(theta)
        training_data_losses += [loss(model, data['training'], wd_coefficient)]
        validation_data_losses += [loss(model, data['validation'], wd_coefficient)]
        if do_early_stopping and validation_data_losses[-1] < best_so_far['validationLoss']:
            best_so_far['theta'] = copy.deepcopy(theta)  # deepcopy avoids memory reference bug
            best_so_far['validationLoss'] = validation_data_losses[-1]
            best_so_far['afterNIters'] = optimization_iteration_i

        if np.mod(optimization_iteration_i, round(n_iters / float(NUM_CLASSES))) == 0:
            print 'After {0} optimization iterations, training data loss is {1}, and validation data ' \
                  'loss is {2}'.format(optimization_iteration_i, training_data_losses[-1], validation_data_losses[-1])

        # check gradient again, this time with more typical parameters and with a different data size
        if optimization_iteration_i == n_iters:
            print 'Now testing the gradient on just a mini-batch instead of the whole training set... '
            test_gradient(model, training_batch, wd_coefficient)

    if do_early_stopping:
        print 'Early stopping: validation loss was lowest after {0} iterations. ' \
              'We chose the model that we had then.'.format(best_so_far['afterNIters'])
        theta = copy.deepcopy(best_so_far['theta'])  # deepcopy avoids memory reference bug

    # the optimization is finished. Now do some reporting.
    model = theta_to_model(theta)
    if n_iters != 0:
        plt.hold(True)
        plt.plot(training_data_losses, 'b')
        plt.plot(validation_data_losses, 'r')
        plt.legend(['training', 'validation'])
        plt.ylabel('loss')
        plt.xlabel('iteration number')
        plt.hold(False)

    for data_name, data_segment in data.iteritems():
        print 'The loss on the {0} data is {1}'.format(data_name, loss(model, data_segment, wd_coefficient))
        if wd_coefficient != 0:
            print 'The classification loss (i.e. without weight decay) ' \
                  'on the {0} data is {1}'.format(data_name, loss(model, data_segment, 0))
        print 'The classification error rate on the {0} data is {1}'.format(data_name,
                                                                            classification_performance(model,
                                                                                                       data_segment))


def test_gradient(model, data, wd_coefficient):
    base_theta = model_to_theta(model)
    h = 1e-2
    correctness_threshold = 1e-5
    analytic_gradient_struct = d_loss_by_d_model(model, data, wd_coefficient)
    if np.size(analytic_gradient_struct.keys(), 0) != 2:
        raise Exception('The object returned by def d_loss_by_d_model should have exactly two field names: '
                        '.input_to_hid and .hid_to_class')
    
    if np.size(analytic_gradient_struct['inputToHid']) != np.size(model['inputToHid']):
        raise Exception('The size of .input_to_hid of the return value of d_loss_by_d_model (currently {0}) '
                        'should be same as the size of model[\'inputToHid\'] '
                        '(currently {1})'.format(np.size(analytic_gradient_struct['inputToHid']),
                                                 np.size(model['inputToHid'])))
    
    if np.size(analytic_gradient_struct['hidToClass']) != np.size(model['hidToClass']):
        raise Exception('The size of .hid_to_class of the return value of d_loss_by_d_model (currently {0}) '
                        'should be same as the size of model[\'hidToClass\'] '
                        '(currently {1})'.format(np.size(analytic_gradient_struct['hidToClass']),
                                                 np.size(model['hidToClass'])))
    
    analytic_gradient = model_to_theta(analytic_gradient_struct)
    if any(np.isnan(analytic_gradient)) or any(np.isinf(analytic_gradient)):
        raise Exception('Your gradient computation produced a NaN or infinity. That is an error.')
    
    # We want to test the gradient not for every element of theta, because that's a lot of work.
    # Instead, we test for only a few elements. If there's an error, this is probably enough to find that error.
    # We want to first test the hid_to_class gradient, because that's most likely to be correct (it's the easier one).
    # Let's build a list of theta indices to check. We'll check 20 elements of hid_to_class, and 80 elements of input_
    # to_hid (it's bigger than hid_to_class).
    input_to_hid_theta_size = np.prod(np.size(model['inputToHid']))
    hid_to_class_theta_size = np.prod(np.size(model['hidToClass']))
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
            temp += loss(theta_to_model(base_theta + theta_step * distance), data, wd_coefficient) * weight
        fd_here = temp / h
        diff = abs(analytic_here - fd_here)
        if diff > correctness_threshold and diff / float(abs(analytic_here) + abs(fd_here)) > correctness_threshold:
            part_names = ['inputToHid', 'hidToClass']
            raise Exception('Theta element #{0} (part of {1}), with value {2}, has finite difference gradient {3} but '
                            'analytic gradient {4}. That looks like an error.'.format(test_index,
                                                                                      part_names[i <= 19],
                                                                                      base_theta[test_index],
                                                                                      fd_here,
                                                                                      analytic_here))
        
        if i == 19:
            print 'Gradient test passed for hid_to_class.'
        if i == 99:
            print 'Gradient test passed for input_to_hid.'
    
    print 'Gradient test passed. That means that the gradient that your code computed is within 0.001%% of the ' \
          'gradient that the finite difference approximation computed, so the gradient calculation procedure is ' \
          'probably correct (not certainly, but probably).'


def logistic(input):
    return 1. / (1. + np.exp(-input))


def log_sum_exp_over_rows(a):
    """Computes log(sum(np.exp(a), 1)) in a numerically stable way."""
    col_maxs = np.max(a, axis=0)
    return np.log(sum(np.exp(a - np.tile(col_maxs, (np.size(a, 0), 1))), 0)) + col_maxs


def loss(model, data, wd_coefficient):
    """
    Notes:
    * Before we can calculate the loss, we need to calculate a variety of intermediate values,
      like the state of the hidden units. This is the forward pass, and you'll likely want to copy it
      into d_loss_by_d_model, because these values are also very useful for that def.

    Args:
        model :
                - 'inputToHid' is a matrix of size <number of hidden units> by <number of inputs i.e. NUM_INPUT_UNITS>
                   It contains the weights from the input units to the hidden units.
                - 'hidToClass' is a matrix of size <number of classes i.e. NUM_CLASSES> by <number of hidden units>
                   It contains the weights from the hidden units to the softmax units.
        data :
                - 'inputs' is a matrix of size <number of inputs i.e. NUM_INPUT_UNITS> by <number of data cases>
                   Each column describes a different data case.
                - 'targets' is a matrix of size <number of classes i.e. NUM_CLASSES> by <number of data cases>
                   Each column describes a different data case. It contains a one-of-N encoding of the class,
                   i.e. one element in every column is 1 and the others are 0.

    Returns:
        float : loss of model
    """
    hid_input = np.dot(model['inputToHid'], data['inputs'])
    hid_output = logistic(hid_input)
    class_input = np.dot(model['hidToClass'], hid_output)

    # The following three lines of code implement the softmax.
    # However, it's written differently from what the lectures say.
    # In the lectures, a softmax is described using an exponential divided by a sum of exponentials.
    # What we do here is exactly equivalent (you can check the math or just check it in practice),
    # but this is more numerically stable.
    # log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities.
    # size: <1> by <number of data cases>
    class_normalizer = log_sum_exp_over_rows(class_input)
    # class_normalizer = class_normalizer * (1. - class_normalizer)
    # log of probability of each class. size: <number of classes, i.e. NUM_CLASSES> by <number of data cases>
    log_class_prob = class_input - np.tile(class_normalizer, (np.size(class_input, 0), 1))
    # probability of each class. Each column (i.e. each case) sums to 1. size: <number of classes,
    # i.e. NUM_CLASSES> by <number of data cases>
    class_prob = np.exp(log_class_prob)
    # select the right log class probability using that sum then take the mean over all data cases.
    classification_loss = -np.mean(sum(log_class_prob * data['targets'], 0))
    # weight decay loss. very straightforward: E = 1/2 * wd_coeffecient * theta^2
    wd_loss = sum(model_to_theta(model) ** 2) / 2.0 * wd_coefficient
    return classification_loss + wd_loss


def d_loss_by_d_model(model, data, wd_coefficient):
    """
    Notes:
    * This is the only def that you're expected to change. Right now, it just returns a lot of zeros,
      which is obviously not the correct output. Your job is to change that.

    Args:
        model :
                - 'inputToHid' is a matrix of size <number of hidden units> by <number of inputs i.e. NUM_INPUT_UNITS>
                - 'hidToClass' is a matrix of size <number of classes i.e. NUM_CLASSES> by <number of hidden units>
        data :
                - 'inputs' is a matrix of size <number of inputs i.e. NUM_INPUT_UNITS> by <number of data cases>
                - 'targets' is a matrix of size <number of classes i.e. NUM_CLASSES> by <number of data cases>

    Returns:
        dict:   The returned object is supposed to be exactly like parameter <model>,
                i.e. it has fields ret['inputToHid'] and ret['hidToClass'].
                However, the contents of those matrices are gradients (d loss by d model parameter),
                instead of model parameters.
    """
    ret = dict()

    hid_input = np.dot(model['inputToHid'], data['inputs'])
    hid_output = logistic(hid_input)
    class_input = np.dot(model['hidToClass'], hid_output)
    class_normalizer = log_sum_exp_over_rows(class_input)
    log_class_prob = class_input - np.tile(class_normalizer, (np.size(class_input, 0), 1))
    class_prob = np.exp(log_class_prob)

    # weight decay loss. very straightforward: E = 1/2 * wd_coeffecient * theta^2
    error_deriv = class_prob - data['targets']
    hid_to_output_weights_gradient = np.dot(hid_output, error_deriv.T) / float(np.size(hid_output, axis=1))
    ret['hidToClass'] = hid_to_output_weights_gradient.T

    backpropagate_error_deriv = np.dot(model['hidToClass'].T, error_deriv)
    input_to_hidden_weights_gradient = np.dot(data['inputs'], ((1.0 - hid_output) * hid_output *
                                                               backpropagate_error_deriv).T) / float(np.size(hid_output,
                                                                                                             axis=1))
    ret['inputToHid'] = input_to_hidden_weights_gradient.T

    ret['inputToHid'] += model['inputToHid'] * wd_coefficient
    ret['hidToClass'] += model['hidToClass'] * wd_coefficient
    return ret


def model_to_theta(model):
    """Takes a model (or gradient in model form), and turns it into one long vector. See also theta_to_model."""
    model_copy = copy.deepcopy(model)
    return np.hstack((model_copy['inputToHid'].flatten(), model_copy['hidToClass'].flatten()))


def theta_to_model(theta):
    """Takes a model (or gradient) in the form of one long vector (maybe produced by model_to_theta),
    and restores it to the structure format, i.e. with fields .input_to_hid and .hid_to_class, both matrices.
    """
    n_hid = np.size(theta, 0) / (NUM_INPUT_UNITS + NUM_CLASSES)
    return {'inputToHid': np.reshape(theta[:NUM_INPUT_UNITS * n_hid], (n_hid, NUM_INPUT_UNITS)),
            'hidToClass': np.reshape(theta[NUM_INPUT_UNITS * n_hid: np.size(theta, 0)],
                                                  (NUM_CLASSES, n_hid))}


def initial_model(n_hid):
    n_params = (NUM_INPUT_UNITS + NUM_CLASSES) * n_hid
    as_row_vector = np.cos(range(n_params))
    # We don't use random initialization, for this assignment. This way, everybody will get the same results.
    return theta_to_model(np.transpose(np.column_stack(as_row_vector)) * 0.1)


def classification_performance(model, data):
    """This returns the fraction of data cases that is incorrectly classified by the model."""
    # input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
    hid_input = np.dot(model['inputToHid'], data['inputs'])
    # output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
    hid_output = logistic(hid_input)
    # input to the components of the softmax. size: <number of classes, i.e. NUM_CLASSES> by <number of data cases>
    class_input = np.dot(model['hidToClass'], hid_output)
    
    choices = np.argmax(class_input, axis=0)
    targets = np.argmax(data['targets'], axis=0)

    return np.mean(np.array(choices != targets, dtype=float))
