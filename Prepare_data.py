import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
import sys,os
with open('./data/sample_submission.csv', 'r') as reader:
    i=0
    
    for line in reader:
        if i == 0 :
            fields = line.split(',')
            sample = np.array([[0. for i in range(len(fields))] for i in range(10000)])
        else:
            sample[i-1] = line.strip().split(',')
        i +=1
    
    print(sample)
    

with open('./data/test.csv', 'r') as reader:
    i=0
    
    for line in reader:
        if i==0:
            fields = line.split(',')
            test = np.array([[0. for i in range(len(fields))] for i in range(10000)])
        else:
            tmp = line.strip().split(',')
            tmp = [r.zfill(1) for r in tmp]
            test[i-1] = tmp
        i +=1
    

with open('./data/train.csv', 'r') as reader:
    i=0
    
    for line in reader:
        if i == 0 :
            fields = line.split(',')
            train = np.array([[0. for i in range(len(fields))] for i in range(10000)])
        else:
            tmp = line.strip().split(',')
            tmp = [r.zfill(1) for r in tmp]
            train[i-1] = tmp
        i +=1

x_train = train[:,1:-4]
y_train = train[:,-4:]

x_test = test[:,1:]

np.savez('./data/dataset',x_train,y_train,x_test)