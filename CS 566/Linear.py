import csv
from NNet import NNet as model
import numpy as np
import matplotlib.pyplot as plt


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
    L.append({"type":'NLayer',"params":{"size": cols - 1, "ac_func":'sigmoid', "lam": 0,"sparsity_param":0} })
    L.append({"type":'Regressor',"params":{"size": 1, "ac_func":'sigmoid', "lam": 0,"sparsity_param":0} })
    lr = model(Layers = L,lam = 0, beta = 0)
    result1 = lr.naive_SGD(data = data, output = output, decay = 0.99, norm_method='max', l_rate = 0.7,batches = 1, Uafterbatch= True)
    result2 = lr.naive_SGD(data = data, output = output, decay = 0.99, norm_method='norm', l_rate = 0.7,batches = 1, Uafterbatch= True)
    norm_max, = plt.plot(xrange(1, len(result1['cost']) + 1) , result1['cost'],'b')
    norm_norm, = plt.plot(xrange(1, len(result2['cost']) + 1) , result2['cost'],'g')
    plt.legend([norm_max,norm_norm],['Max','Z-score'])
    plt.ylabel('Average SSE')
    plt.xlabel('Runs of Batch Updating')
    plt.title("Comparison of Normalization Methods")
    plt.show()

    result1 = lr.naive_SGD(data = data, output = output, decay = 0.99, norm_method='norm', l_rate = 0.7,batches = 1, Uafterbatch= True)
    result2 = lr.naive_SGD(data = data, output = output, decay = 0.99, norm_method='norm', l_rate = 0.7,batches = 5, Uafterbatch= True)
    result3 = lr.naive_SGD(data = data, output = output, decay = 0.99, norm_method='norm', l_rate = 0.7,batches = 53, Uafterbatch= True)
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

test()