from __future__ import print_function
import numpy as np
import os
from six.moves import cPickle as pickle


def get_dataset(pickle_file):
  try:
    with open(pickle_file, 'rb') as f:
      letter_set = pickle.load(f)
      # let's shuffle the letters to have random validation and training set
      np.random.shuffle(letter_set)
      train = letter_set.reshape(letter_set.shape[0], letter_set.shape[1] * letter_set.shape[2])
  except Exception as e:
    print('Unable to process data from', pickle_file, ':', e)
    raise e

  return train

def get_data(folder, num):
    image_files = os.listdir(folder)
    files = [folder+'/'+fn for fn in image_files if 'pickle' in fn]
    charas = [fn[:1] for fn in image_files if 'pickle' in fn]
    dataset = []
    labels = []
    for i in xrange(len(files)):
        images = get_dataset(files[i])
        #index = np.random.choice(xrange(len(images)), num, replace= False)
        dataset.extend(images)
        labels.extend([charas[i]] * images.shape[0])
        print (charas[i], ' Done')
    dataset = np.array(dataset)
    labels = np.array(labels)
    return dataset, labels

X_test,labels = get_data('./notMNIST_small', 10000)
with open("all_test", "wb") as f:
    pickle.dump((X_test,labels),f)
print ('all Done')