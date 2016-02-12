from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from NNet import Sparse_AutoEncoder as SAE
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

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
train_dataset = get_dataset('notMNIST_large\\A.pickle')
test_dataset = get_dataset('notMNIST_small\A.pickle')

model = SAE(train_dataset.shape[0], hidden_size, 3e-3, 0.1, 3)
model.train(train_dataset)
print ("a")

