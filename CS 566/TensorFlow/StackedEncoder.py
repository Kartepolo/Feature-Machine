import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from AutoEncoder_Train import AutoEncoder
from six.moves import cPickle as pickle
from sklearn.cross_validation import KFold
from sklearn.utils import shuffle

import tensorflow as tf


class StackedEncoder(AutoEncoder):
    def __init__(self, lam, **kwargs):
        super(StackedEncoder,self).__init__(**kwargs)
        self.lam = lam

    def create_loss_optimizer(self):
        reconstr_cost = tf.sqrt(tf.reduce_mean(tf.square(self.data - self.output[len(self.output) - 1])))
        for i in xrange(len(self.weight)):
            regular_cost += tf.nn.l2_loss(self.weight[i])+tf.nn.l2_loss(self.bias[i])
        self.cost = reconstr_cost + regular_cost
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
    def Pre_training(self, data, prob):
        prev = self.layer_size[0]
        feed = data
        for i in xrange(self.layer_size / 2):
            ae = AutoEncoder(layers = [prev,i,prev],learning_rate= self.learning_rate, batch_size= self.batch_size)
            ae.start()
            self.sess.run(ae.train, feed_dict = {self.data: feed, self.keep_prob : prob})
            prev = i
            self.weight[i - 1] = ae.weight[0]
            self.weight[len(self.layer_size) - 1 - i] = ae.weight[1]
            self.bias[i - 1] = ae.bias[0]
            self.bias[len(self.layer_size) - 1 - i] = ae.bias[1]
            feed = ae.output[1]
            ae.cleanup()

def visualize(data, prob):
    data = shuffle(data)
    ae = StackedEncoder(lam = 3e-3, layers = [784,100,64,100,784], learning_rate= 0.1, batch_size= 10)
    ae.start()
    ae.Pre_training(data = data[:5000,], prob = prob)
    ae.train(data = data[:5000,], training_epochs = 20, prob = 1.0)
    weights = ae.visual_hidden(0)
    plt.figure(figsize=(10, 10))
    #Visualize hidden layers
    for i in xrange(len(weights)):
        plt.subplot2grid((10,10),(i / 10, i % 10))
        plt.imshow(weights[i].reshape(image_size ,image_size ) * 5, cmap = cm.gray)
        plt.axis('off')
    plt.show()
    weights = ae.visual_hidden(1)
    plt.figure(figsize=(8, 8))
    #Visualize hidden layers
    for i in xrange(len(weights)):
        plt.subplot2grid((8,8),(i / 8, i % 8))
        plt.imshow(weights[i].reshape(image_size ,image_size ) * 5, cmap = cm.gray)
        plt.axis('off')
    plt.show()




