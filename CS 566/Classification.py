import pandas as pd
from NNet import NNet as model
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import scale

def sign(x,y):
    return np.sum(x == y)

def norm(x):
    #Normalization
    s = 1.0 / x.shape[1] * np.tile(np.sum(x,axis = 1), (x.shape[1],1)).transpose()
    x = x - s
    eps = np.zeros((x.shape[0],x.shape[1]))
    eps.fill(np.finfo(float).eps)
    stdev = np.std(x,axis = 1) + np.finfo(float).eps
    x = x / (stdev * np.sqrt(x.shape[1])) [:, np.newaxis]
    return x

def dummy(array, col_vals, cols):
    non = [u for u in xrange(array.shape[0]) if u not in cols]
    result = array[non,]
    obs = array.shape[1]
    for r in cols:
        k,indices = np.unique(array[r,], return_inverse= True)
        tmp = np.zeros((len(k),obs))
        tmp[indices, xrange(obs)] = 1
        if (len(k) < len(col_vals[r])):
            pos = 14
            tmp = np.insert(tmp, pos - 1, np.zeros(obs),0)
        result = np.concatenate((result, tmp[1:,]), axis = 0)
    return result

def crossvalidate(k, input, output, L, beta, d):
    kf = StratifiedKFold(output, n_folds= k, shuffle = True)
    Layers = []
    for l in L:
        Layers.append({'type':l[0],'params':l[1]})
    lr = model(Layers = Layers, beta = beta)
    err = []
    for train_index, test_index in kf:
        data_train, data_test = input[:,train_index], input[:,test_index]
        output_train, output_test = output[train_index], output[test_index]
        result = lr.train(data = data_train, output = output_train, debug = False)
        pred = lr.predict(data_test)
        err.append(1.0 * np.sum(sign(pred, output_test)) / data_test.shape[1])
        lr.init()
    return 1.0 * sum(err) / len(err)


def classification():
    header = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race', 'sex','capital-gain',
              'capital-loss','hours-per-week','native-country','label']
    types = ['np.float32','S32','np.float32','S32','np.float32','S32','S32','S32','S32','S32','np.float32','np.float32','np.float32','S32','S32']
    dt = dict(zip(header, types))
    train = pd.read_csv('adult.data', sep= ',', na_values = '?',skipinitialspace = True, names = header)
    test = pd.read_csv('adult.test', sep= ',', na_values = '?',skipinitialspace = True, names = header)
    train_missing = train.isnull().any(axis=0)
    test_missing = test.isnull().any(axis=0)
    np_train = np.array(train.dropna(axis = 0)).transpose()
    np_test = np.array(test.dropna(axis = 0)).transpose()
    col_vals = {}
    for r in xrange(np_train.shape[0]):
        col_vals[r] = np.unique(np_train[r,])
    train_output = np.array(pd.get_dummies(np_train[14,])['>50K'])
    test_output = np.array(pd.get_dummies(np_test[14,])['>50K.'])
    hidden_size = 10
    # making dummy variables
    np_train= dummy(np_train[:14,],col_vals,[1,3,5,6,7,8,9,13])
    np_test= dummy(np_test[:14,],col_vals,[1,3,5,6,7,8,9,13])
    #Normalization
    np_train = np_train.astype(np.float64)
    np_test = np_test.astype(np.float64)
    np_train= norm(np_train.astype(np.float64))
    np_test=norm(np_test.astype(np.float64))

    #Backward interpolation


    L = [('NLayer',{"size": np_train.shape[0], "ac_func":'sigmoid', "lam": 0,"sparsity_param":0}),
         ('Classifier',{"size": 2, "ac_func":'softmax', "lam": 0,"sparsity_param":0})]
    d = {'decay' : 0.95, 'l_rate' : 0.01, 'batches' : 100, 'Uafterbatch' : True,'debug' : False}
    err = []
    #Logistic
    err= []
    sub_err = []
    for j in xrange(10):
        cv_result = crossvalidate(10,np_train, train_output.T, L, 3, d)
        sub_err.append(cv_result)
    err.append(1.0 * sum(sub_err) / len(sub_err))
    print err
    Layers = []
    for l in L:
        Layers.append({'type':l[0],'params':l[1]})
    lr = model(Layers = Layers, beta = 3)
    lr.train(data = np_train, output = train_output.T, debug = False)
    pred = lr.predict(np_test)
    print 1.0 * np.sum(sign(pred, test_output.T)) / np_test.shape[1]
    #Added one hidden layer
    L.insert(1,('NLayer',{"size": 5, "ac_func":'sigmoid', "lam": 1e-5,"sparsity_param":0}))
    L[0][1]['lam'] = 1e-5
    L[1][1]['lam'] = 1e-5
    L[2][1]['lam'] = 1e-5
    err = []
    test_err = []
    for i in range(5,95,10):
        L[1][1]['size'] = i
        err.append(crossvalidate(10,np_train, train_output.T, L, 3, d))
        print err[len(err) -1]
        Layers = []
        for l in L:
            Layers.append({'type':l[0],'params':l[1]})
        lr = model(Layers = Layers, beta = 3)
        lr.train(data = np_train, output = train_output.T, debug = False)
        pred = lr.predict(np_test)
        test_err.append(1.0 * np.sum(sign(pred, test_output.T)) / np_test.shape[1])
    print err
    print test_err


classification()
