import csv
from NNet import NNet as model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.preprocessing import scale


def crossvalidate(data, output, lam, pd):
    kf = KFold(data.shape[1], n_folds= 5)
    L = []
    L.append({"type":'NLayer',"params":{"size": data.shape[0], "ac_func":'sigmoid', "lam": lam,"sparsity_param":0} })
    L.append({"type":'Regressor',"params":{"size": 1, "ac_func":'sigmoid', "lam": lam,"sparsity_param":0} })
    lr = model(Layers = L,lam = lam, beta = 0)
    err = []
    for train_index, test_index in kf:
        data_train, data_test = data[:,train_index], data[:,test_index]
        output_train, output_test = output[train_index], output[test_index]
        data_train = norm(data_train,'norm')
        data_test = norm(data_test,'norm')
        result = lr.naive_SGD(data = data_train, output = output_train, **pd)
        pred = lr.predict(data_test)
        err.append(1.0 * np.sum((pred - output_test) ** 2) / data_test.shape[1])
    return 1.0 * sum(err) / len(err)

def regularization():
    with open('mortality-dataset.csv','rU') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        counter = 0
        result = []
        for row in f_csv:
            result.append(row)
            counter +=1
        f.close()
    cols = len(headers)
    result = np.array(result,dtype=np.float).transpose()
    data = result[1:,]
    output = result[0,].transpose()
    summary = []
    ndata = norm(data, 'norm')
    pd = {'decay' : 0.85, 'l_rate' : 0.5, 'batches' : 1, 'Uafterbatch' : True}
    for lam in [0,0.01,0.1,0.3,0.5,0.7,1,2]:
        err = []
        for i in xrange(100):
            err.append(crossvalidate(ndata,output, lam, pd))
        summary.append((err,lam))
    p0, = plt.plot(xrange(1, 101) , summary[0][0])
    p1, = plt.plot(xrange(1, 101) , summary[1][0])
    plt.legend([p0,p1],['No Regularization','Lambda = 0.01'])
    plt.ylabel('MSE')
    plt.xlabel('Runs of Experiment')
    plt.title('Regularization Results')
    plt.show()
    mean = [1.0 * sum(x[0])/ len(x[0]) for x in summary]
    lams = np.log10([x[1]+1 for x in summary])
    plt.plot(lams, mean,'o')
    plt.plot(lams, mean)
    plt.ylabel('Averaged MSE over 100 Experiments')
    plt.xlabel('Log10(Lambda + 1)')
    plt.title('CV Error for different lambda')
    plt.show()
    print 'End'





def hundredcv():
    with open('mortality-dataset.csv','rU') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        counter = 0
        result = []
        for row in f_csv:
            result.append(row)
            counter +=1
        f.close()
    cols = len(headers)
    result = np.array(result,dtype=np.float).transpose()
    data = result[1:,]
    output = result[0,].transpose()
    summary = []
    L = []
    L.append({"type":'NLayer',"params":{"size": cols - 1, "ac_func":'sigmoid', "lam": 0,"sparsity_param":0} })
    L.append({"type":'Regressor',"params":{"size": 1, "ac_func":'sigmoid', "lam": 0,"sparsity_param":0} })
    lr = model(Layers = L,lam = 0, beta = 0)
    err = []
    ndata = norm(data, 'norm')
    for i in xrange(100):
        result0 = lr.naive_SGD(data = ndata, output = output, decay = 0.99,l_rate = 0.5,batches = 1)
        err.append(result0['cost'][len(result0['cost']) - 1])
    summary.append(err)
    pd = {'decay' : 0.99, 'l_rate' : 0.5, 'batches' : 1, 'Uafterbatch' : True}
    err = []
    for i in xrange(100):
        err.append(crossvalidate(data,output, lam = 0, pd = pd))
    summary.append(err)
    #CV for decay
    for d in [0.7,0.75,0.8,0.85,0.9,0.95,0.99]:
        cv_error = []
        pd['decay'] = d
        for i in xrange(100):
            cv_error.append(crossvalidate(data,output, lam = 0, pd = pd))
        summary.append((cv_error, d))
    #CV for step
    pd['decay'] = 0.85
    for l in [0.1,0.3,0.5,0.7,0.9]:
        cv_error = []
        pd['l_rate'] = l
        for i in xrange(100):
            cv_error.append(crossvalidate(data,output, lam = 0, pd = pd))
        summary.append((cv_error, l))
    p0, = plt.plot(xrange(1, 101) , summary[0])
    p1, = plt.plot(xrange(1, 101) , summary[1])
    plt.legend([p0,p1],['Training Error','5-Fold CV Error'])
    plt.ylabel('Mean of Squared Error')
    plt.xlabel('Runs of Experiment')
    plt.title('Error Comparison with Decay = 0.9, Step = 0.5')
    plt.show()
    mean = [1.0 * sum(x[0])/len(x[0]) for x in summary[2:9]]
    Decays = [x[1] for x in summary[2:9]]
    plt.plot(Decays, mean)
    plt.ylabel('Averaged MSE over 100 Experiments')
    plt.xlabel('Decay')
    plt.title('CV Error for different Decay')
    plt.show()
    mean = [1.0 * sum(x[0])/len(x[0]) for x in summary[9:14]]
    Steps = [x[1] for x in summary[9:14]]
    plt.plot(Steps, mean)
    plt.ylabel('Averaged MSE over 100 Experiments')
    plt.xlabel('Step')
    plt.title('CV Error for different Step')
    plt.show()
    print 'End'

def norm(data, norm_method):
    if norm_method == 'max':
        return (data.T / (data.max(axis=1) + np.spacing(0))).T
    else:
        return scale(data.T).T


def test():
    with open('mortality-dataset.csv','rU') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        counter = 0
        result = []
        for row in f_csv:
            result.append(row)
            counter +=1
        f.close()
    cols = len(headers)
    result = np.array(result,dtype=np.float).transpose()
    data = result[1:,]
    output = result[0,].transpose()
    L = []
    L.append({"type":'NLayer',"params":{"size": cols - 1, "ac_func":'sigmoid', "lam": 1,"sparsity_param":0} })
    L.append({"type":'Regressor',"params":{"size": 1, "ac_func":'sigmoid', "lam": 1,"sparsity_param":0} })
    lr = model(Layers = L,lam = 0, beta = 0)
    result1 = lr.naive_SGD(data = data, output = output, decay = 0.99, norm_method='max', l_rate = 0.7,batches = 53, Uafterbatch= False)
    result2 = lr.naive_SGD(data = data, output = output, decay = 0.99, norm_method='norm', l_rate = 0.7,batches = 1, Uafterbatch= False)
    norm_max, = plt.plot(xrange(1, len(result1['cost']) + 1) , result1['cost'],'b')
    norm_norm, = plt.plot(xrange(1, len(result2['cost']) + 1) , result2['cost'],'g')
    plt.legend([norm_max,norm_norm],['Max','Z-score'])
    plt.ylabel('Average SSE')
    plt.xlabel('Runs of Batch Updating')
    plt.title("Comparison of Normalization Methods")
    plt.show()

    result1 = lr.naive_SGD(data = data, output = output, decay = 0.8, norm_method='norm', l_rate = 0.1,batches = 1, Uafterbatch= True)
    result2 = lr.naive_SGD(data = data, output = output, decay = 0.8, norm_method='norm', l_rate = 0.1,batches = 5, Uafterbatch= True)
    result3 = lr.naive_SGD(data = data, output = output, decay = 0.8, norm_method='norm', l_rate = 0.1,batches = 53, Uafterbatch= True)
    b1, = plt.plot(xrange(1, len(result1['cost']) + 1) , result1['cost'],'b')
    b5, = plt.plot(xrange(1, len(result2['cost']) + 1) , result2['cost'],'g')
    b53, = plt.plot(xrange(1, len(result3['cost']) + 1) , result3['cost'],'r')
    plt.legend([b1,b5,b53],['Batch','Every 5','Each'])
    plt.ylabel('Average SSE')
    plt.xlabel('Runs of Batch Updating')
    plt.title("Impact of Epoch Size")
    plt.show()


    Decay = [0.99, 0.96]
    Norm = ['max','norm']
    Batch = [1, 5]
    Uafter = [True, False]
    l_rate = [0.3,0.7]
    Mean_Max_min= {}
    for i in xrange(16):
        Mean_Max_min[i + 1] = []
    counter = 1
    for d in Decay:
        for n in Norm:
            for l in l_rate:
                for u in Uafter:
                    for b in Batch:
                        Experiements = []
                        for runs in xrange(100):
                            result = lr.naive_SGD(data = data, output = output, decay = d, norm_method=n, l_rate = l,batches = b, Uafterbatch= u)
                            Experiements.append(result['cost'])
                        final = [min(s) for s in Experiements]
                        mean = sum(final) * 1.0 / len(final)
                        Mean_Max_min[counter].append([(mean, max(final), min(final)),(d,n,l,u)])
                    counter += 1

    print 'aaaa'

hundredcv()