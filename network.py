#!/usr/bin/env python
import theano
import os
from theano import tensor as tt
from six.moves import cPickle
import numpy as np
import pickle as pkl
import sys
import timeit

rng = np.random.RandomState(1111)


def add_zeros(sample):
    s = [len(i) for i in sample]
    v = max(s)
    z = [0, 0, 0, 0, 0]
    for i in sample:
        if len(i) < v:
            try:
                np.array(i).shape
            except ValueError:
                print "ff"
                print i
                sys.exit()
            for j in range(v-len(i)):
                i += [[z, z]]
    return np.array(sample, dtype="int32"), np.array(s, dtype="int32")


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = tt.nnet.softmax(tt.dot(input, self.W) + self.b)

        self.y_pred = tt.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

        self.input = input

    def negative_log_likelihood(self, y):
        return -tt.mean(tt.log(self.p_y_given_x)[tt.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return tt.mean(tt.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def t(self, ndx):
        return self.input[ndx]


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=tt.tanh):
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == tt.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = tt.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class ConvLayer(object):
    """Convolutional Layer. """
    def __init__(self, rng, input_in_shape, cut_pose, length_z, hyper_cf):
        self.input = input_in_shape
        W_bound = np.sqrt(6. / (hyper_cf*3))
        self.W1 = theano.shared(
            value=np.asarray(
                rng.uniform(low=-W_bound,
                            high=W_bound,
                            size=(length_z, hyper_cf)),
                dtype=theano.config.floatX
            ),
            name='W1',
            borrow=True
        )

        b_values = np.zeros((hyper_cf,), dtype=theano.config.floatX)
        self.b1 = theano.shared(value=b_values, name='b1', borrow=True)

        def set_col_in_u(branch, location, u, w, b):
            value = tt.tanh(tt.dot(branch, w)+b)
            return tt.set_subtensor(u[:, location], value)

        def get_pose(pose, branches_in_pose):
            #branches_in_pose = pose.shape[0]
            length = tt.arange(branches_in_pose)
            U = tt.zeros_like(tt.eye(hyper_cf, branches_in_pose))
            conca5, up_5 = theano.scan(set_col_in_u,
                                       sequences=[pose[:branches_in_pose], length],
                                       outputs_info=U,
                                       non_sequences=[self.W1,
                                                      self.b1],
                                       strict=True)
            final_result = conca5[-1]
            return tt.max(final_result, axis=1)

        # x = tt.tensor3("x")
        # y = tt.ivector("y")
        initial, up_i = theano.scan(get_pose,
                                    sequences=[input_in_shape, cut_pose],
                                    outputs_info=None)

        # self.output = theano.function(inputs=[input_in_shape,y], outputs=initial)
        self.output = initial
        self.params = [self.W1, self.b1]


class DataTreatmentLayer(object):
    """Data Treatment Layer. Take the input a plays around with it
       to get it into shape."""
    def __init__(self, rng, list_of_poses, cut_positions,
                 hyper_neighbors, num_branch_dict):
        self.list_of_poses = list_of_poses
        self.cut_positions = cut_positions
        W_bound = np.sqrt(6. / (hyper_neighbors*3))

        w_distance = np.asarray(
                             rng.uniform(low=-W_bound,
                                         high=W_bound,
                                         size=(hyper_neighbors, num_branch_dict+1)),
                             dtype=theano.config.floatX
                           )
        w_distance[:, 0] = 0
        w_branch = np.asarray(
                             rng.uniform(low=-W_bound,
                                         high=W_bound,
                                         size=(hyper_neighbors, num_branch_dict+1)),
                             dtype=theano.config.floatX
                           )
        w_branch[:, 0] = 0

        self.W_distance_bin = theano.shared(value=w_distance,
                                            name='W_distance_bin', borrow=True)
        self.W_branch_type = theano.shared(value=w_branch,
                                           name='W_branch_type', borrow=True)

        self.params = [self.W_distance_bin, self.W_branch_type]

        def pose_feature_vector(index, ps, side):
            """ Make a single feature vector for each
            in the list """
            return ps[index][side]

        def get_pose(pose):
            conca0, up_0 = theano.scan(pose_feature_vector,
                                       sequences=[tt.arange(pose.shape[0])],
                                       outputs_info=None,
                                       non_sequences=[pose, 0])
            conca1, up_1 = theano.scan(pose_feature_vector,
                                       sequences=[tt.arange(pose.shape[0])],
                                       outputs_info=None,
                                       non_sequences=[pose, 1])
            conca2, up_2 = theano.scan(fn=lambda col, wf: wf[:, tt.cast(col, 'int32')].T,
                                       sequences=[conca0],
                                       outputs_info=None,
                                       non_sequences=[self.W_branch_type])
            conca3, up_3 = theano.scan(fn=lambda col, wf: wf[:, tt.cast(col, 'int32')].T,
                                       sequences=[conca1],
                                       outputs_info=None,
                                       non_sequences=[self.W_distance_bin])
            conca4, up_4 = theano.scan(fn=lambda col1, col2: tt.concatenate([col1.flatten(1), col2.flatten(1)]),
                                       sequences=[conca2, conca3],
                                       outputs_info=None)

            return conca4

        # x = tt.ltensor4("x")
        initial, up_i = theano.scan(get_pose,
                                    sequences=[list_of_poses],
                                    outputs_info=None)

        self.output = initial

    def t(self, ndx):
        return self.list_of_poses[ndx]


def save_model_good(layers, epoch):
    if not os.path.exists("models"):
        os.makedirs("models")
    for idx, layer in enumerate(layers):
        for param in layer.params:
            filename = "models/epoch_{}_layer_{}_{}.pkl".format(
                str(epoch), idx, param.name)
            with open(filename, "wb") as f:
                cPickle.dump(param.get_value(), f,
                             protocol=cPickle.HIGHEST_PROTOCOL)


def save_model_naive(model, epoch):
    with open("epoch_{}_complete_model.pkl".format(epoch), "wb") as f:
        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)


def main(train=True):
    print("Reading dataset")
    with open(sys.argv[1], "rb") as f:
        dataset = pkl.load(f)
    print("Loaded {} samples in dataset".format(len(dataset)))
    train_perc = 0.8
    test_perc = 0.1
    valid_perc = 0.1

    total = len(dataset)
    train_index = int(total*train_perc)
    test_index = int(train_index + total*test_perc)
    valid_index = int(test_index + total*valid_perc)

    train_set = dataset[:train_index]
    test_set = dataset[train_index:test_index]
    valid_set = dataset[test_index:]

    print("{}, {}, {} samples in train, test and validation datasets respectively ({}%, {}%, {}%)"
        .format(len(train_set), len(test_set), len(valid_set), train_perc*100, test_perc*100, valid_perc*100))
    train_set_x, train_set_y = zip(*train_set)
    test_set_x, test_set_y = zip(*test_set)
    valid_set_x, valid_set_y = zip(*valid_set)

    test_set_x,  s_t = add_zeros(test_set_x)
    test_set_x = theano.shared(test_set_x)
    s_t = theano.shared(s_t, borrow=True)
    test_set_y = theano.shared(np.asarray(test_set_y, dtype="int32"))

    train_set_x, s_s = add_zeros(train_set_x)
    train_set_x = theano.shared(train_set_x)
    s_s = theano.shared(s_s, borrow=True)
    train_set_y = theano.shared(np.asarray(train_set_y, dtype="int32"))

    valid_set_x, s_v = add_zeros(valid_set_x)
    valid_set_x = theano.shared(valid_set_x)
    s_v = theano.shared(s_v, borrow=True)
    valid_set_y = theano.shared(np.asarray(valid_set_y, dtype="int32"))



    batch_size = 50
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size


    z = tt.ivector()
    x = tt.itensor4()
    y = tt.ivector()
    index = tt.iscalar()

    BRANCHES_IN_DICT = 4472+1
    HYPER_NEIGHBORS = 80
    HYPER_CF = 150
    HIDDEN_1 = 50
    HIDDEN_2 = 20
    OUT = 2

    layer_0 = DataTreatmentLayer(rng, x, z, HYPER_NEIGHBORS, BRANCHES_IN_DICT)
    layer_1 = ConvLayer(rng, layer_0.output, layer_0.cut_positions,
                        HYPER_NEIGHBORS * 10, HYPER_CF)
    layer_2 = HiddenLayer(rng, layer_1.output, HYPER_CF, HIDDEN_1)
    layer_3 = HiddenLayer(rng, layer_2.output, HIDDEN_1, HIDDEN_2)
    layer_out = LogisticRegression(layer_3.output, HIDDEN_2, OUT)

    layers = [layer_0, layer_1, layer_2, layer_3, layer_out]

    cost = layer_out.negative_log_likelihood(y)


    test_model = theano.function(
        [index],
        layer_out.errors(y),
        givens={
            x: test_set_x[index*batch_size:(index+1)*batch_size],
            y: test_set_y[index*batch_size:(index+1)*batch_size],
            z: s_t[index*batch_size:(index+1)*batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer_out.errors(y),
        givens={
            x: valid_set_x[index*batch_size:(index+1)*batch_size],
            y: valid_set_y[index*batch_size:(index+1)*batch_size],
            z: s_v[index*batch_size:(index+1)*batch_size]
        }
    )

    params = layer_out.params + layer_3.params + layer_2.params + layer_1.params + layer_0.params

    grads = tt.grad(cost, params)

    learning_rate = 0.05
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size],
            z: s_s[index*batch_size:(index+1)*batch_size]
        }
    )

    start_time = timeit.default_timer()
    if train:
        epoch = 0
        done_looping = False
        n_epochs = 1000
        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.0

        patience = 10000
        patience_increase = 2
        improvement_threshold = 0.995

        validation_frequency = n_train_batches

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):

                iter = (epoch - 1) * n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print('training @ iter = ', iter)
                cost_ij = train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i)
                                         for i in range(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)

                    print('epoch {}, minibatch {}/{}, validation error {} %'
                        .format(epoch, minibatch_index + 1, n_train_batches,
                                this_validation_loss * 100))
                    save_model_good(layers, epoch)
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        # improve patience if loss improvement is good enough
                        if this_validation_loss < (best_validation_loss
                                                   * improvement_threshold):
                            patience = max(patience, iter * patience_increase)

                            # save best validation score and iteration number
                            best_validation_loss = this_validation_loss
                            best_iter = iter

                            # test it on the test set
                            test_losses = [test_model(i)
                                        for i in range(n_test_batches)]
                            test_score = np.mean(test_losses)
                            print(('   epoch {}, minibatch {}/{}, test error of best model {} %')
                                .format(epoch, minibatch_index + 1, n_train_batches,
                                        test_score * 100.))
                if iter >= patience:
                    print("Breaking to prevent overfitting")
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation score of {} % obtained at iteration {}, with test performance {} %'
            .format(best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code ' + ' ran for %.2fm' % ((end_time - start_time) / 60.))

    sys.exit()


if __name__ == '__main__':
    main()
