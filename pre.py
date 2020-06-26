import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

## 데이터 불러오기
train = pd.read_csv('./data/train.csv', sep=',', index_col = 0, header = 0)
test = pd.read_csv('./data/test.csv', sep=',', index_col = 0, header = 0)


## 데이터 분해 및 columns이름 저장
train_col = train.columns[:-4]
test_col = test.columns
y_train_col = train.columns[-4:]

y_train = train.values[:,-4:]
train = train.values[:,:-4]
test = test.values

# scaler = MinMaxScaler()
# train = scaler.fit_transform(train)
# test = scaler.transform(test)


## NaN값이 있는 train,test값을 다시 데이터 프레임으로 감싸주기
train = pd.DataFrame(train, columns=train_col)
test = pd.DataFrame(test, columns=test_col)


train_src = train.filter(regex='_src$',axis=1).T.interpolate(limit_direction='both').T.values # 선형보간법
# train_damp = 1
# train_damp = 625/train.values[:,0:1]/train.values[:,0:1]*(10**(625/train.values[:,0:1]/train.values[:,0:1] - 1))
train_damp = np.exp(np.pi*(10 - train.values[:,0:1])/3.44)
train_dst = train.filter(regex='_dst$',axis=1).T.interpolate(limit_direction='both').T.values / train_damp# 선형보간법

test_src = test.filter(regex='_src$',axis=1).T.interpolate(limit_direction='both').T.values
# test_damp = 1
# test_damp = 625/test.values[:,0:1]/test.values[:,0:1]*(10**(625/test.values[:,0:1]/test.values[:,0:1] - 1))
test_damp = np.exp(np.pi*(10 - test.values[:,0:1])/3.44)
test_dst = test.filter(regex='_dst$',axis=1).T.interpolate(limit_direction='both').T.values / test_damp

num_train = 0
num_test = 0

for i in range(10000):
    if train_dst[i].sum() == 0:
        num_train += 1
    if test_dst[i].sum() == 0:
        num_test += 1 

print(num_train)
print(num_test)




max_test = 0
max_train = 0
train_fu_real = []
train_fu_imag = []
test_fu_real = []
test_fu_imag = []
train_src_dst_fu = []
test_src_dst_fu = []

train_dst_mean = []
test_dst_mean = []

rho_10 = []
rho_15 = []
rho_20 = []
rho_25 = []
scaler = StandardScaler()

times = 70

for i in range(10000):
    tmp_x = 0
    tmp_y = 0
    for j in range(35):
        if train_src[i, j] == 0 and train_dst[i,j] != 0:
            # train_src[i,j] = 1e-10
            train_src[i,j] = train_dst[i,j]
            # train_dst[i,j] = 0

        if test_src[i, j] == 0 and test_dst[i,j] != 0:
            # test_src[i,j] = 1e-10
            test_src[i,j] = test_dst[i,j]
            # test_dst[i,j] = 0

    if train['rho'][i] == 10:
        rho_10.append(i)
    if train['rho'][i] == 15:
        rho_15.append(i)
    if train['rho'][i] == 20:
        rho_20.append(i)
    if train['rho'][i] == 25:
        rho_25.append(i)
    # plt.plot(range(35), train_src[i])
    # plt.show()

    train_fu_real.append(np.fft.fft(scaler.fit_transform(train_dst[i:i+1].T).T[0], n=times).real)
    train_fu_imag.append(np.fft.fft(scaler.fit_transform(train_dst[i:i+1].T).T[0], n=times).imag)
    test_fu_real.append(np.fft.fft(scaler.fit_transform(test_dst[i:i+1].T).T[0], n=times).real)
    test_fu_imag.append(np.fft.fft(scaler.fit_transform(test_dst[i:i+1].T).T[0], n=times).imag)
    train_src_dst_fu.append(np.fft.ifft(train_src[i]-train_dst[i], n=times).real)
    test_src_dst_fu.append(np.fft.ifft(test_src[i]-test_dst[i], n=times).real)
    train_dst_mean.append([train_dst[i].mean()])
    test_dst_mean.append([test_dst[i].mean()])

print("==========================")
print(((train_src - train_dst) < 0).sum())
print(((test_src - test_dst) < 0).sum())
print("==========================")

trian_dst_mean = np.array(train_dst_mean)
test_dst_mean = np.array(test_dst_mean)

seq = [10000,times]

train_2fu_real = np.fft.fft2(scaler.fit_transform(train_dst.T).T, s = seq).real
train_2fu_imag = np.fft.fft2(scaler.fit_transform(train_dst.T).T, s = seq).imag
test_2fu_real = np.fft.fft2(scaler.fit_transform(test_dst.T).T, s = seq).real
test_2fu_imag = np.fft.fft2(scaler.fit_transform(test_dst.T).T, s = seq).imag    

print(max_train)
print(max_test)
print("RHO")
# print(rho_10/nrho_10)
# print(rho_15/nrho_15)
# print(rho_20/nrho_20)
# print(rho_25/nrho_25)





small = 1e-20



x_train = np.concatenate([train.values[:,0:1]**2,trian_dst_mean, train_damp, train_dst, train_src-train_dst, train_src/(train_dst+small), train_fu_real, train_fu_imag] , axis = 1)
x_pred = np.concatenate([test.values[:,0:1]**2,test_dst_mean, test_damp, test_dst, test_src-test_dst, test_src/(test_dst+small),test_fu_real,test_fu_imag], axis = 1)


# x_train = np.concatenate([train.values[:,0:1]**2, train_src, train_dst, train_src - train_dst,train_src/(train_dst+small)], axis = 1)
# x_pred = np.concatenate([test.values[:,0:1]**2, test_src, test_dst, test_src - test_dst,test_src/(test_dst+small)], axis = 1)

# x_train = np.concatenate([train.values[:,0:1]**2, train_src, train_dst, train_src - train_dst,train_src/(train_dst+small), np.log10((train_src + small)/(train_dst+small))], axis = 1)
# x_pred = np.concatenate([test.values[:,0:1]**2, test_src, train_dst, test_src - test_dst,test_src/(test_dst+small), np.log10((test_src+small)/(test_dst+small))], axis = 1)
# print(pd.DataFrame(x_train).isnull().sum())
# print(pd.DataFrame(np.log10(train_src.values) - np.log10(train_dst.values)))

print(x_train.shape)
print(y_train.shape)
print(x_pred.shape)

np.save('./x.npy', arr=x_train)
np.save('./y.npy', arr=y_train)
np.save('./test.npy', arr=x_pred)
