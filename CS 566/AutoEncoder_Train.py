from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from six.moves import cPickle as pickle
import tensorflow as tf
from sklearn.cross_validation import KFold
from sklearn.utils import shuffle
import matplotlib.cm as cm

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
seed = 99 # Random seed

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))


def get_dataset(pickle_file):
  try:
    with open(pickle_file, 'rb') as f:
      letter_set = pickle.load(f)
      # let's shuffle the letters to have random validation and training set
      np.random.shuffle(letter_set)
      train = letter_set.reshape(letter_set.shape[0], letter_set.shape[1] * letter_set.shape[2])
  except Exception as e:
    print('Unable to process data from', pickle_file, ':', e)
    raise

  return train.T


train_size = 200000
valid_size = 10000
test_size = 10000

hidden_size = 196
train_dataset = get_dataset('notMNIST_large/A.pickle').T
test_dataset = get_dataset('notMNIST_small/A.pickle').T
#train_dataset = train_dataset / np.sqrt((train_dataset ** 2).sum(axis = 1))[:, np.newaxis]
#test_dataset = test_dataset / np.sqrt((test_dataset ** 2).sum(axis = 1))[:, np.newaxis]

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0/(fan_in+fan_out))
    high = constant * np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out), minval = low, maxval = high, dtype = tf.float32)



class AutoEncoder(object):
    def __init__(self,layers,learning_rate = 0.01, batch_size = 100):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight = []
        self.bias = []
        self.activation = []
        self.output = []
        self.layer_size = layers



        self.data = tf.placeholder(tf.float32, [None, layers[0]])
        self.mask = tf.placeholder("float", [None, layers[0]], name='mask')
        self.keep_prob = tf.placeholder("float")
        self.noise = tf.placeholder("float", [None, layers[0]], name = "noise")
        self.create_network()
        self.encoder_network()
        self.create_loss_optimizer()

    def start(self):
        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def create_network(self, tied = True):

        l = len(self.layer_size)
        input_dim = self.layer_size[0]
        counter = 1
        for i in xrange(1,l):
            dim = self.layer_size[i]
            if tied and i > len(self.layer_size) / 2 :
                self.weight.append(tf.transpose(self.weight[len(self.layer_size) - 1 - i]))
            else:
                self.weight.append(tf.Variable(xavier_init(input_dim,dim), name = "W" + "{:02d}".format(counter)))
            self.bias.append(tf.Variable(tf.zeros([dim]), name = "B" + "{:02d}".format(counter)))
            input_dim = dim
            counter += 1


    def encoder_network(self):
        self.output.append(self.mask * tf.add(self.data, self.noise))
        for l in xrange(len(self.layer_size) - 1):
            self.output.append(tf.nn.tanh(tf.matmul(self.output[l], self.weight[l])+ self.bias[l]))

    def create_loss_optimizer(self):
        reconstr_cost = tf.sqrt(tf.reduce_mean(tf.square(self.data - self.output[len(self.output) - 1])))
        self.cost = reconstr_cost
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

    def partial_fit(self, data, prob, mask, noise):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.data: data, self.keep_prob: prob, self.mask : mask, self.noise : noise})
        return cost
    def train(self, data, prob, training_epochs = 10, display_step = 5):
        self.num_samples = data.shape[0]
        prev = 1000000000
        for epoch in xrange(training_epochs):
            noise = np.random.multivariate_normal(mu, covariance, data.shape[0])
            avg_cost = 0
            total_batch = int(1.0 * self.num_samples / self.batch_size)
            for i in xrange(total_batch):
                if i == total_batch - 1 : upper = self.num_samples
                else: upper = (i+1) * self.batch_size
                sub = data[i * self.batch_size : upper , ]
                mask_np = np.random.binomial(1, prob, sub.shape)
                cost = self.partial_fit(sub, prob,mask_np, noise[i * self.batch_size : upper , ])
                avg_cost += cost / self.num_samples * self.batch_size
            if (prev - avg_cost) < 1e-4:
                print (epoch + 1)
                break
            else:
                prev = avg_cost
            if epoch % display_step == 0:
                print ("Epoch:", "{:04d}".format(epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    def reconstruct(self, data, mask, noise):
        self.output[0] = self.data
        zeron = np.zeros(data.shape)
        return (self.sess.run(self.output[len(self.output) - 1], feed_dict = {self.data: data, self.keep_prob : 1.0,self.mask : mask, self.noise : noise}),
                self.cost.eval(feed_dict = {self.data: data, self.keep_prob : 1.0, self.mask : mask, self.noise : zeron}))
    def visual_hidden(self, num):
        matrix = self.weight[num]
        #denominator = tf.sqrt(tf.reduce_sum(tf.square(matrix),0))
        #hidden_image = tf.split(1, self.layer_size[num + 1], tf.div(matrix, denominator))
        w = tf.split(1, self.layer_size[num + 1], matrix)
        return [i.eval() for i in w]
    def cleanup(self):
        self.sess.close()

def corrupt_generator(data, portion, mean, cov):
    row = data.shape[0]
    col = data.shape[1]
    s= np.random.multivariate_normal(mean, cov, int(row * portion))
    concat = np.concatenate((data, s))
    return shuffle(concat, random_state = seed)


#Some simple cv
def cv(folds, dataset, prob, r):
    for H_size in r:
        cv_cost = 0
        kf = KFold(dataset.shape[0], n_folds= folds)
        ae = AutoEncoder(layers = [784,H_size ** 2,784], learning_rate= 0.1)
        for train_index, test_index in kf:
            ae.start()
            data_train, data_test = dataset[train_index,:], dataset[test_index,:]
            mask_np = np.random.binomial(1, prob, 784)
            ae.train(data = data_train,training_epochs = 20, prob = prob)
            rec, err = ae.reconstruct(data_test,  mask= mask_np)
            cv_cost += err / folds
            ae.cleanup()
        print ("Hidden Neurons:", H_size ** 2,  "CV_Error:",cv_cost)
    return


# 5-Fold Cross validation to determine the optimal hidden neuron size
#cv(5, train_dataset, 1.0, range(10, 33, 4))


#Generate a corrupted version of data
mu = [0.01] * (image_size ** 2)
covariance = np.eye(image_size ** 2, dtype= np.float32) * 0.04
corrupt = corrupt_generator(train_dataset, 0.1, mu, covariance)

#Denoising AutoEncoder
#Trial 1 : Dropout at input with original version
#cv(5, train_dataset, 0.9, range(10, 33, 4))

#Trial 2: A corrupted version without dropout
#cv(5, corrupt, 1.0, range(10, 33, 4))

#Trial 3:
#cv(5, corrupt, 0.9, range(10, 33, 4))
#Visualization below
def visualize(data, title, prob):
    data = shuffle(data)
    ae = AutoEncoder(layers = [784,100,784], learning_rate= 0.1, batch_size= 10)
    ae.start()
    ae.train(data = data[:5000,], training_epochs = 20, prob = 1.0)
    weights = ae.visual_hidden(0)
    plt.figure(figsize=(10, 10))
    #Visualize hidden layers
    for i in xrange(len(weights)):
        plt.subplot2grid((10,10),(i / 10, i % 10))
        plt.imshow(weights[i].reshape(image_size ,image_size ) * 5, cmap = cm.gray)
        plt.axis('off')
    plt.show()
    x_sample = data[:100,]
    mask_np = np.random.binomial(1, prob, x_sample.shape)
    noise = np.random.multivariate_normal(mu, covariance, x_sample.shape[0])
    x_reconstruct, err = ae.reconstruct(x_sample, mask_np, noise)
    #Visualize reconstruction
    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow((x_sample[i] + noise[i]).reshape(28, 28))
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28))
        plt.colorbar()
    plt.show()
    #Report error on test
    mask_np = np.random.binomial(1, prob, test_dataset.shape)
    noise = np.random.multivariate_normal(mu, covariance, test_dataset.shape[0])
    t_reconstruct, test_err = ae.reconstruct(test_dataset, mask_np, noise)
    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(test_dataset[i, :].reshape(28, 28))
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(t_reconstruct[i].reshape(28, 28))
        plt.colorbar()
    plt.show()
    print ("Test Error", test_err)
#visualize(corrupt, "Blurred Autoencoder", 1.0)
#visualize(corrupt, "Blurred Autoencoder with denoising", 0.5)
#visualize(train_dataset, "Plain Autoencoder", 1.0)
visualize(train_dataset, "Denoising Autoencoder", 1.0)

