import theano
import numpy as np
from Stacked_AutoEncoder import SdA
from six.moves import cPickle as pickle
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle
import matplotlib.cm as cm
import os, timeit, sys
import theano.tensor as tensor



letter_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9}

def shared_dataset(data_xy, borrow=True):

    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
    return shared_x, tensor.cast(shared_y, 'int32')



with open('data','rb') as f:
    training,labels = pickle.load(f)
    training,labels = shuffle(training,labels)

training = training + 0.5

def test_SdA(datasets, finetune_lr=0.3, pretraining_epochs=15,
             pretrain_lr=0.1, training_epochs=100, batch_size=100):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """

    train_set_x, train_set_y = datasets[0]
    val_set_x, val_set_y = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size

    # np random generator
    # start-snippet-3
    numpy_rng = np.random.RandomState(89677)
    print('... building the model')
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=28 * 28,
        hidden_layers=[1000, 900, 800],
        n_outs=10
    )
    # end-snippet-3 start-snippet-4
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('... getting the pretraining functions')
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print('... pre-training the model')
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    corruption_levels = [.5, .4, .3]
    for i in range(len(sda.neural_layers)):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, np.mean(c)))

    end_time = timeit.default_timer()

    print(('The pretraining code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))
    # end-snippet-4
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print('... getting the finetuning functions')
    train_fn, validate_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate = finetune_lr
    )

    print('... finetunning the model')
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = np.mean(validation_losses)
                print ('epoch %i, minibatch %i/%i, validation error %f %%' % (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print(('The training code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))

labels = np.array([letter_dict[i] for i in labels])
index = np.random.choice(xrange(len(training)), len(training) * 0.2,replace=False)
validation = training[index,:]
v_labels = labels[index,]
training = np.delete(training, index, axis = 0)
labels = np.delete(labels, index, axis = 0)
training,labels = shared_dataset((training,labels))
validation,v_labels = shared_dataset((validation,v_labels))
test_SdA([(training,labels),(validation,v_labels)])





