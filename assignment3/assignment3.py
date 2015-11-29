import numpy as np
import matplotlib.pyplot as plt


def a3(wd_coefficient, n_hid, n_iters, learning_rate, momentum_multiplier, do_early_stopping, mini_batch_size, from_data_file):
    model = initial_model(n_hid)
    best_so_far = dict()
    training_batch = dict()
    # from_data_file = load('data.mat')
    datas = from_data_file.data
    n_training_cases = np.size(datas['training']['inputs'], 1)
    if n_iters != 0:
        print 'Now testing the gradient on the whole training set... '
        test_gradient(model, datas.training, wd_coefficient)
    

    # optimization
    theta = model_to_theta(model)
    momentum_speed = theta * 0
    training_data_losses = []
    validation_data_losses = []
    if do_early_stopping:
        best_so_far['theta'] = -1 # this will be overwritten soon
        best_so_far['validationLoss'] = np.inf
        best_so_far['afterNIters'] = -1
    
    for optimization_iteration_i in xrange(1, n_iters):
        model = theta_to_model(theta)
        
        training_batch_start = np.mod((optimization_iteration_i-1) * mini_batch_size, n_training_cases)+1
        training_batch['inputs'] = datas['training']['inputs'][:, training_batch_start : training_batch_start + mini_batch_size - 1]
        training_batch['targets'] = datas['training']['targets'][:, training_batch_start : training_batch_start + mini_batch_size - 1]
        gradient = model_to_theta(d_loss_by_d_model(model, training_batch, wd_coefficient))
        momentum_speed = momentum_speed * momentum_multiplier - gradient
        theta = theta + momentum_speed * learning_rate

        model = theta_to_model(theta)
        training_data_losses = [training_data_losses, loss(model, datas.training, wd_coefficient)]
        validation_data_losses = [validation_data_losses, loss(model, datas.validation, wd_coefficient)]
        if do_early_stopping and validation_data_losses() < best_so_far['validationLoss']:
            best_so_far['theta'] = theta # this will be overwritten soon
            best_so_far['validationLoss'] = validation_data_losses()
            best_so_far['afterNIters'] = optimization_iteration_i
        
        if np.mod(optimization_iteration_i, round(n_iters/10)) == 0:
            print 'After %d optimization iterations, training data loss is %f, and validation data loss is %f' % \
                  (optimization_iteration_i, training_data_losses(), validation_data_losses())
        
        if optimization_iteration_i == n_iters: # check gradient again, this time with more typical parameters and with a different data size
            print 'Now testing the gradient on just a mini-batch instead of the whole training set... '
            test_gradient(model, training_batch, wd_coefficient)
         
    
    if do_early_stopping:
        print 'Early stopping: validation loss was lowest after %d iterations. We chose the model that we had then.' % \
              best_so_far['afterNIters']
        theta = best_so_far['theta']
    
    # the optimization is finished. Now do some reporting.
    model = theta_to_model(theta)
    if n_iters != 0:
        plt.hold(True)
        plt.plot(training_data_losses, 'b')
        plt.plot(validation_data_losses, 'r')
        plt.legend('training', 'validation')
        plt.ylabel('loss')
        plt.xlabel('iteration number')
        plt.hold(False)

    
    datas2 = {'training': datas['training'], 'validation': datas['validation'], 'test': datas['test']}
    for data, data_name in datas2.iteritems():
        print '\nThe loss on the %s data is %f\n' % (data_name, loss(model, data, wd_coefficient))
        if wd_coefficient != 0:
            print 'The classification loss (i.e. without weight decay) on the %s data is %f\n' % \
                  (data_name, loss(model, data, 0))
        print 'The classification error rate on the %s data is %f' % \
              (data_name, classification_performance(model, data))
    


def test_gradient(model, data, wd_coefficient):
    base_theta = model_to_theta(model)
    h = 1e-2
    correctness_threshold = 1e-5
    analytic_gradient_struct = d_loss_by_d_model(model, data, wd_coefficient)
    if np.size(analytic_gradient_struct.keys(), 0) != 2:
         raise Exception('The object returned by def d_loss_by_d_model should have exactly two field names: '
                         '.input_to_hid and .hid_to_class')
    
    if any(np.size(analytic_gradient_struct['inputToHid']) != np.size(model['inputToHid'])):
         raise Exception('The size of .input_to_hid of the return value of d_loss_by_d_model (currently [%d, %d]) '
                         'should be same as the size of model[\'inputToHid\'] (currently [%d, %d])' %
                         (np.size(analytic_gradient_struct['inputToHid']), np.size(model['inputToHid'])))
    
    if any(np.size(analytic_gradient_struct.hid_to_class) != np.size(model['hidToClass'])):
         raise Exception('The size of .hid_to_class of the return value of d_loss_by_d_model (currently [%d, %d]) '
                         'should be same as the size of model[\'hidToClass\'] (currently [%d, %d])' %
                         (np.size(analytic_gradient_struct.hid_to_class), np.size(model['hidToClass'])))
    
    analytic_gradient = model_to_theta(analytic_gradient_struct)
    if any(np.isnan(analytic_gradient)) or any(np.isinf(analytic_gradient)):
         raise Exception('Your gradient computation produced a NaN or infinity. That is an error.')
    
    # We want to test the gradient not for every element of theta, because that's a lot of work. Instead, we test for only a few elements. If there's an error, this is probably enough to find that error.
    # We want to first test the hid_to_class gradient, because that's most likely to be correct (it's the easier one).
    # Let's build a list of theta indices to check. We'll check 20 elements of hid_to_class, and 80 elements of input_to_hid (it's bigger than hid_to_class).
    input_to_hid_theta_size = np.prod(np.size(model['inputToHid']))
    hid_to_class_theta_size = np.prod(np.size(model['hidToClass']))
    big_prime = 1299721 # 1299721 is prime and thus ensures a somewhat random-like selection of indices.
    hid_to_class_indices_to_check = np.mod(big_prime * np.array(range(1,20)), hid_to_class_theta_size) + 1 + input_to_hid_theta_size
    input_to_hid_indices_to_check = np.mod(big_prime * np.array(range(1,80)), input_to_hid_theta_size) + 1
    indices_to_check = [hid_to_class_indices_to_check, input_to_hid_indices_to_check]
    for i in xrange(1,100):
        test_index = indices_to_check(i)
        analytic_here = analytic_gradient(test_index)
        theta_step = base_theta * 0
        theta_step[test_index] = h
        contribution_distances = range(-4,-1) + range(1, 4)
        contribution_weights = [1/280, -4/105, 1/5, -4/5, 4/5, -1/5, 4/105, -1/280]
        temp = 0
        for contribution_index in xrange(1,8):
            temp = temp + loss(theta_to_model(base_theta + theta_step * contribution_distances(contribution_index)), data, wd_coefficient) * contribution_weights(contribution_index)
        
        fd_here = temp / h
        diff = abs(analytic_here - fd_here)
        # print '#d #e #e #e #e\n', test_index, base_theta(test_index), diff, fd_here, analytic_here)
        if (diff > correctness_threshold) and (diff / (abs(analytic_here) + abs(fd_here)) > correctness_threshold):
            part_names = {'input_to_hid', 'hid_to_class'}
            raise Exception('Theta element %%d (part of %s), with value #e, has finite difference gradient %e but '
                            'analytic gradient #e. That looks like an error.\n' % 
                            (test_index, part_names[(i<=20)+1], base_theta(test_index), fd_here, analytic_here))
        
        if i == 20:
            print 'Gradient test passed for hid_to_class. '
        if i == 100:
            print 'Gradient test passed for input_to_hid. '
    
    print 'Gradient test passed. That means that the gradient that your code computed is within 0.001## of the gradient that the finite difference approximation computed, so the gradient calculation procedure is probably correct (not certainly, but probably).\n'


def logistic(input):
    return 1 / (1 + np.np.exp(-input))


def log_sum_exp_over_rows(a):
    # This computes log(sum(np.exp(a), 1)) in a numerically stable way
    maxs_small = max(a, [], 1)
    maxs_big = np.tile(maxs_small, [np.size(a, 0), 1])
    return np.log(sum(np.exp(a - maxs_big), 1)) + maxs_small


def loss(model, data, wd_coefficient):
    # model['inputToHid'] is a matrix of size <number of hidden units> by <number of inputs i.e. 256>. It contains the weights from the input units to the hidden units.
    # model['hidToClass'] is a matrix of size <number of classes i.e. 10> by <number of hidden units>. It contains the weights from the hidden units to the softmax units.
    # data['inputs'] is a matrix of size <number of inputs i.e. 256> by <number of data cases>. Each column describes a different data case. 
    # data['targets'] is a matrix of size <number of classes i.e. 10> by <number of data cases>. Each column describes a different data case. It contains a one-of-N encoding of the class, i.e. one element in every column is 1 and the others are 0.
	 
    # Before we can calculate the loss, we need to calculate a variety of intermediate values, like the state of the hidden units. This is the
    # forward pass, and you'll likely want to copy it into d_loss_by_d_model, because these values are also very useful for that def.
    hid_input = model['inputToHid'] * data['inputs'] # input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
    hid_output = logistic(hid_input) # output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
    class_input = model['hidToClass'] * hid_output # input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>
    
    # The following three lines of code implement the softmax.
    # However, it's written differently from what the lectures say.
    # In the lectures, a softmax is described using an exponential divided by a sum of exponentials.
    # What we do here is exactly equivalent (you can check the math or just check it in practice), but this is more numerically stable. 
    # "Numerically stable" means that this way, there will never be really big numbers involved.
    # The exponential in the lectures can lead to really big numbers, which are fine in mathematical equations, but can lead to all sorts of problems in Octave.
    # Octave isn't well prepared to deal with really large numbers, like the number 10 to the power 1000. Computations with such numbers get unstable, so we avoid them.
    class_normalizer = log_sum_exp_over_rows(class_input) # log(sum(np.exp of class_input)) is what we subtract to get properly normalized log class probabilities. size: <1> by <number of data cases>
    log_class_prob = class_input - np.tile(class_normalizer, (np.size(class_input, 0), 1)) # log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
    class_prob = np.exp(log_class_prob) # probability of each class. Each column (i.e. each case) sums to 1. size: <number of classes, i.e. 10> by <number of data cases>
    
    classification_loss = -np.mean(sum(log_class_prob * data['targets'], 1)) # select the right log class probability using that sum then take the mean over all data cases.
    wd_loss = sum(np.power(model_to_theta(model), 2)) / 2 * wd_coefficient # weight decay loss. very straightforward: E = 1/2 * wd_coeffecient * theta^2
    return classification_loss + wd_loss


def d_loss_by_d_model(model, data, wd_coefficient):
    # model['inputToHid'] is a matrix of size <number of hidden units> by <number of inputs i.e. 256>
    # model['hidToClass'] is a matrix of size <number of classes i.e. 10> by <number of hidden units>
    # data['inputs'] is a matrix of size <number of inputs i.e. 256> by <number of data cases>
    # data['targets'] is a matrix of size <number of classes i.e. 10> by <number of data cases>
    ret = dict()
    # The returned object is supposed to be exactly like parameter <model>, i.e. it has fields ret['inputToHid'] and ret['hidToClass']. However, the contents of those matrices are gradients (d loss by d model parameter), instead of model parameters.
	 
    # This is the only def that you're np.expected to change. Right now, it just returns a lot of zeros, which is obviously not the correct output. Your job is to change that.
    ret['inputToHid'] = model['inputToHid'] * 0
    ret['hidToClass'] = model['hidToClass'] * 0
    return ret


def model_to_theta(model):
    # This def takes a model (or gradient in model form), and turns it into one long vector. See also theta_to_model.
    return np.vstack(np.transpose(model['inputToHid']), np.transpose(model['hidToClass']))


def theta_to_model(theta):
    ret = dict()
    # This def takes a model (or gradient) in the form of one long vector (maybe produced by model_to_theta), and restores it to the structure format, i.e. with fields .input_to_hid and .hid_to_class, both matrices.
    n_hid = np.size(theta, 0) / (256+10)
    ret['inputToHid'] = np.transpose(np.reshape(theta[1: 256*n_hid], 256, n_hid))
    ret['hidToClass'] = np.transpose(np.reshape(theta[256 * n_hid + 1 : np.size(theta, 0)], n_hid, 10))
    return ret


def initial_model(n_hid):
    n_params = (256+10) * n_hid
    as_row_vector = np.cos(range(0,(n_params-1)))
    return theta_to_model(as_row_vector * 0.1) # We don't use random initialization, for this assignment. This way, everybody will get the same results.


def classification_performance(model, data):
    # This returns the fraction of data cases that is incorrectly classified by the model.
    hid_input = model['inputToHid'] * data['inputs'] # input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
    hid_output = logistic(hid_input) # output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
    class_input = model['hidToClass'] * hid_output # input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>
    
    dump = np.max(class_input) # choices is integer: the chosen class, plus 1.
    choices = np.argmax(class_input)
    dump = np.max(data['targets']) # targets is integer: the target class, plus 1.
    targets = np.argmax(data['targets'])
    return np.mean(float(choices != targets))
