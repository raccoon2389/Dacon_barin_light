import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input,Dropout
from keras.layers.merge import concatenate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import keras.losses
import sys,os

npload = np.load('./data/dataset.npz')

x_train, y_train, x_pred = npload['x_train'], npload['y_train'], npload['x_test']

x_train_rho = np.zeros((10000,1))
x_train_src = np.zeros((10000,35))
x_train_dst = np.zeros((10000,35))



print(x_train.shape)
# x_pred_tmp = np.zeros((100000,34,3))

for i in range(10000):
    x_train_rho[i]= x_train[i,0]
    x_train_src[i]= x_train[i,1:36]
    x_train_dst[1]= x_train[i,36:]

        # x_pred_tmp[i,c,0]= x_pred[i,0]
        # x_pred_tmp[i,c,1]= x_pred[i,c+1]
        # x_pred_tmp[1,c,2]= x_pred[i,c+17]

rho_scal = StandardScaler()
src_scal = StandardScaler()
dst_scal = StandardScaler()
rho_scal.fit(x_train_rho)
src_scal.fit(x_train_src)
dst_scal.fit(x_train_dst)
x_train_rho = rho_scal.transform(x_train_rho)
x_train_src = src_scal.transform(x_train_src)
x_train_dst = dst_scal.transform(x_train_dst)

x_train_rho,x_test_rho, x_train_src,x_test_src,x_train_dst,x_test_dst = train_test_split(x_train_rho,x_train_src,x_train_dst,test_size=0.2,shuffle=True)




# x_train = x_train[0:8000,:]
# x_test = x_train[8000:,:]
# x_test = x_train[8000:,:]
# x_test = x_train[8000:,:]
y_train_tmp = y_train

y_train = y_train_tmp[0:8000,:]
y_test = y_train_tmp[8000:,:]

#g650_src,660_src,670_src,680_src,690_src,700_src,710_src,720_src,730_src,740_src,750_src,760_src,
# 770_src,780_src,790_src,800_src,810_src,820_src,830_src,840_src,850_src,860_src,870_src,880_src,890_src,900_src,
# 910_src,920_src,930_src,940_src,950_src,960_src,970_src,980_src,990_src
# print(x_train_tmp.shape)

# print(x_train.shape)
input_rho = Input(shape=(1,))
input_src = Input(shape=(35,))
input_dst = Input(shape=(35,))

skf = KFold()
accuracy = []
for train, validation in skf.split(x_train_rho, y_train):
    
    dense_rho_1 = Dense(1, activation = 'relu')(input_rho)

    dense_src_1 = Dense(100, activation = 'relu')(input_src)
    dense_src_1 = Dropout(0.2)(dense_src_1)

    dense_src_2 = Dense(100, activation = 'relu')(dense_src_1)
    dense_src_2 = Dropout(0.2)(dense_src_2)
    dense_src_3 = Dense(100, activation = 'relu')(dense_src_2)
    dense_src_3 = Dropout(0.2)(dense_src_3)

    dense_dst_1 = Dense(100, activation = 'relu')(input_dst)
    dense_dst_1 = Dropout(0.2)(dense_dst_1)

    dense_dst_2 = Dense(100, activation = 'relu')(dense_dst_1)
    dense_dst_2 = Dropout(0.2)(dense_dst_2)

    dense_dst_3 = Dense(100, activation = 'relu')(dense_dst_2)
    dense_dst_3 = Dropout(0.2)(dense_dst_3)

    merge_srd_dst = concatenate([dense_src_3, dense_dst_3])

    dense_merge1 = Dense(100, activation = 'relu')(merge_srd_dst)

    merge_all = concatenate([dense_rho_1, dense_merge1])

    output1 = Dense(100, activation = 'relu')(merge_all)
    output3 = Dense(4)(output1)

    model = Model(inputs = [input_rho, input_src, input_dst], outputs = [output3])

    model.compile(loss='mse', optimizer='adam',
                metrics=['accuracy'])

    # 학습 데이터를 이용해서 학습
    model.fit([[x_train_rho[train],x_train_src[train],x_train_dst[train]]], y_train[train], epochs=100, batch_size=5)

    # 테스트 데이터를 이용해서 검증
    k_accuracy = '%.4f' % (model.evaluate([x_train_rho[validation],x_train_src[validation],x_train_dst[validation]], Y[validation])[1])
    accuracy.append(k_accuracy)


# model.add(LSTM(20,activation='tanh',input_shape =(34,3)))

model.summary()

# model.compile(optimizer='sgd',loss=keras.losses.mean_absolute_error,metrics=['acc'])
# model.fit([x_train_rho,x_train_src,x_train_dst],y_train,batch_size=50,epochs=500,validation_split=0.25)

# loss, mse = model.evaluate([x_test_rho,x_test_src,x_test_dst],y_test,batch_size=50)

# print(f"loss : {loss}\nmse : {mse}")