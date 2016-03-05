import theano
import numpy as numpy
from AutoEncoder import  dA
from six.moves import cPickle as pickle
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle
import matplotlib.cm as cm
import os, timeit, sys
import theano.tensor as tensor
import pandas as pd
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T
from Optimizer import rmsprop

try:
    import PIL.Image as Image
except ImportError:
    import Image



def shared_dataset(data_xy, borrow=True):

    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
    return shared_x, tensor.cast(shared_y, 'int32')



with open('data','rb') as f:
    training,labels = pickle.load(f)



def test_dA(datasets, learning_rate=0.1, training_epochs=15,
            batch_size=20, output_folder='dA_plots'):

    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # start-snippet-2
    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    # end-snippet-2

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28,
        n_hidden=500
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates = updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in range(training_epochs):
        # go through trainng set
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        print('Training epoch %d, cost ' % epoch, numpy.mean(c))

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print('The no corruption code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((training_time) / 60.))

    # start-snippet-3
    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28,
        n_hidden=500
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.3,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in range(training_epochs):
        # go through trainng set
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        print('Training epoch %d, cost ' % epoch, numpy.mean(c))

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print ('The 30% corruption code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % (training_time / 60.))
    # end-snippet-3

    # end-snippet-4

    os.chdir('../')

training,labels = shared_dataset((training,pd.get_dummies(labels)))
test_dA([(training,labels)])
