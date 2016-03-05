import theano
import numpy
from HiddenLayer import HiddenLayer
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams
from AutoEncoder import dA



def RMSprop(params, grads, lr=0.001, rho=0.9, epsilon=1e-6):
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = tensor.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates



class SdA(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(self,numpy_rng, hidden_layers, theano_rng=None,n_ins=784,n_outs=10,corruption_levels=[0.1, 0.1], learning_rate = 0.1, momentum = 0.1):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano_Class random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        self.neural_layers = []
        self.output_layer = None
        self.dA_layers = []
        self.params = []
        self.in_out = (n_ins, n_outs)
        self.rng = numpy_rng
        self.learning_rate = learning_rate
        self.momentum = momentum

        assert hidden_layers > 0

        if not theano_rng:
            self.theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = tensor.matrix('x')  # the data is presented as rasterized images
        self.y = tensor.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels
        self.construct_network(hidden_layers)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood

        self.finetune_cost = tensor.sum(tensor.nnet.categorical_crossentropy(self.output_layer.output, self.y))
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = tensor.mean(tensor.neq(tensor.argmax(self.output_layer.output, axis=1), self.y))


    def construct_network(self, hidden_size, tied = True):
        l = len(hidden_size)
        input_dim = self.in_out[0]
        feed = self.x
        for j in xrange(l):
            dim = hidden_size[j]
            neural_layer = HiddenLayer(rng=self.rng,input=feed,n_in=input_dim,n_out=dim,layer_name= 'Hidden' + str(j),activation=tensor.nnet.sigmoid)
            dA_layer = dA(numpy_rng=self.rng,
                          theano_rng=self.theano_rng,
                          input=feed,
                          n_visible=input_dim,
                          n_hidden=dim,
                          W=neural_layer.W,
                          bhid=neural_layer.b)
            self.dA_layers.append(dA_layer)
            self.neural_layers.append(neural_layer)
            self.params.extend(neural_layer.params)
            input_dim = dim
            feed = neural_layer.output
        # Adding the final softmax layer
        self.output_layer = HiddenLayer(rng=self.rng,input=feed,n_in=hidden_size[len(hidden_size) - 1],n_out=self.in_out[1],layer_name= 'Softmax', activation=tensor.nnet.softmax)
        self.params.extend(self.output_layer.params)


    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = tensor.lscalar('index')  # index to a minibatch
        corruption_level = tensor.scalar('corruption')  # % of corruption to use
        learning_rate = tensor.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    corruption_level,
                    learning_rate
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)
        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano_Class variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size

        index = tensor.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = tensor.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = RMSprop(self.params,gparams, lr = learning_rate)

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        return train_fn, valid_score

