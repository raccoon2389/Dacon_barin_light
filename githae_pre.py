
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

def outliers(data, axis= 0):
    if type(data) == pd.DataFrame:
        data = data.values
    if len(data.shape) == 1:
        quartile_1, quartile_3 = np.percentile(data,[25,75])
        print("1사분위 : ", quartile_1)
        print("3사분위 : ", quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)  ## 아래
        upper_bound = quartile_3 + (iqr * 1.5)  ## 위
        return np.where((data > upper_bound) | (data < lower_bound))
    else:
        output = []
        for i in range(data.shape[axis]):
            if axis == 0:
                quartile_1, quartile_3 = np.percentile(data[i, :],[25,75])
            else:
                quartile_1, quartile_3 = np.percentile(data[:, i],[25,75])
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr * 1.5)  ## 아래
            upper_bound = quartile_3 + (iqr * 1.5)  ## 위
            if axis == 0:
                output.append(np.where((data[i, :] > upper_bound) | (data[i, :] < lower_bound))[0])
            else:
                output.append(np.where((data[:, i] > upper_bound) | (data[:, i] < lower_bound))[0])
    return np.array(output)


## 데이터 불러오기
train = pd.read_csv('./data/train.csv', sep=',', index_col = 0, header = 0)
test = pd.read_csv('./data/test.csv', sep=',', index_col = 0, header = 0)


## 데이터 분해 및 columns이름 저장
train_col = train.columns[:-4]
test_col = test.columns
y_train_col = train.columns[-4:]
y_train = train.loc[:,"hhb":"na"].values
train = train.values[:,:-4]
test = test.values

# scaler = MinMaxScaler()
# train = scaler.fit_transform(train)
# test = scaler.transform(test)


## NaN값이 있는 train,test값을 다시 데이터 프레임으로 감싸주기
train = pd.DataFrame(train, columns=train_col)
test = pd.DataFrame(test, columns=test_col)

# tmp = outliers(test,1)
# print(tmp.shape)
# temp = []
# for i in range(tmp.shape[0]):
#     temp += list(tmp[i])
# print(len(list(set(temp))))

# tmp = outliers(train,1)
# print(tmp.shape)
# temp = []
# for i in range(tmp.shape[0]):
#     temp += list(tmp[i])
# print(len(list(set(temp))))

train_src = train.filter(regex='_src$',axis=1).interpolate(limit_direction='both',axis=1) # 선형보간법
src_col = list(train_src)
train_dst = train.filter(regex='_dst$',axis=1).interpolate(limit_direction='both',axis=1)
dst_col = list(train_dst)
train_damp = np.exp(np.pi*(10 - train["rho"].values)/3.44)
# print(train_damp)
# print(train_dst.iloc[0,:])
for i in range(10000):
    train_dst.iloc[i,:] = train_dst.iloc[i,:]/train_damp[i] # 선형보간법
# print(train_dst.loc[:,'650_dst':].isnull().sum())

test_src = test.filter(regex='_src$',axis=1).interpolate(limit_direction='both',axis=1)
test_dst = test.filter(regex='_dst$',axis=1).interpolate(limit_direction='both',axis=1)
test_damp = np.exp(np.pi*(10 - test["rho"].values)/3.44)
# tete = pd.DataFrame(np.zeros((10000,35)),columns=dst_col)
for i in range(10000):
    test_dst.iloc[i,:] = test_dst.iloc[i,:]/test_damp[i] # 선형보간법

print(test_dst)
# for i in range(5):
#     tete.iloc[i,:] = test_dst.iloc[i,:]/test_damp[i] # 선형보간법
# print(tete)


# print(test_dst.loc[:,'650_dst':].isnull().sum())


# print(train_dst.isnull().sum())


# print(train_dst.isnull().sum())

# print(test_dst.isnull().sum())

# for i in range(10):
#     plt.subplot(2,2,1)
#     plt.plot(range(35), train_src[i])
#     plt.subplot(2,2,2)
#     plt.plot(range(35), train_dst[i])
#     plt.subplot(2,2,3)
#     plt.plot(range(35), test_src[i])
#     plt.subplot(2,2,4)
#     plt.plot(range(35), test_dst[i])
#     plt.show()


# (ca, cd) = pywt.dwt(train_src, "haar")
# cat = pywt.threshold(ca, np.std(ca), mode="hard")
# cdt = pywt.threshold(cd, np.std(cd), mode="hard")
# train_src = pywt.idwt(cat, cdt, "haar")[:,:35]

# scaler = MinMaxScaler()
# train_dst = scaler.fit_transform(train_dst.T).T
# (ca, cd) = pywt.dwt(train_dst, "haar")
# cat = pywt.threshold(ca, np.std(ca), mode="hard")
# cdt = pywt.threshold(cd, np.std(cd), mode="hard")
# train_dst = pywt.idwt(cat, cdt, "haar")
# train_dst = scaler.inverse_transform(train_dst[:,:35].T).T

# (ca, cd) = pywt.dwt(test_src, "haar")
# cat = pywt.threshold(ca, np.std(ca), mode="hard")
# cdt = pywt.threshold(cd, np.std(cd), mode="hard")
# test_src = pywt.idwt(cat, cdt, "haar")[:,:35]

# test_dst = scaler.fit_transform(test_dst.T).T
# (ca, cd) = pywt.dwt(test_dst, "haar")
# cat = pywt.threshold(ca, np.std(ca), mode="hard")
# cdt = pywt.threshold(cd, np.std(cd), mode="hard")
# test_dst = pywt.idwt(cat, cdt, "haar")
# test_dst = scaler.inverse_transform(test_dst[:,:35].T).T

# for i in range(10):
#     plt.subplot(2,2,1)
#     plt.plot(range(36), train_src[i])
#     plt.subplot(2,2,2)
#     plt.plot(range(35), train_dst[i])
#     plt.subplot(2,2,3)
#     plt.plot(range(36), test_src[i])
#     plt.subplot(2,2,4)
#     plt.plot(range(35), test_dst[i])
#     plt.show()


max_test = 0
max_train = 0
train_fu_real = np.array([[]for i in range(35)])
train_fu_imag = np.array([[]for i in range(35)])
test_fu_real = np.array([[]for i in range(35)])
test_fu_imag = np.array([[]for i in range(35)])

for i in range(10000):
    # tmp_x = 0
    # tmp_y = 0
    for j in range(35):
        if train_src.iloc[i, j] == 0 and train_dst.iloc[i, j] != 0:
                # train_src[i,j] = 1e-10
            train_src.iloc[i, j] = train_dst.iloc[i, j]
            # train_dst[i,j] = 0

        if test_src.iloc[i, j] == 0 and test_dst.iloc[i, j] != 0:
            # test_src[i,j] = 1e-10
            test_src.iloc[i, j] = test_dst.iloc[i, j]
            # test_dst[i,j] = 0
    # if tmp_x > max_train:
    #     max_train = tmp_x
    # if tmp_y > max_test:
    #     max_test = tmp_y
    train_fu_real =np.append(train_fu_real,np.fft.fft(train_dst.iloc[i,:].values-train_dst.iloc[i,:].values.mean()).real)
    train_fu_imag =np.append(train_fu_imag,np.fft.fft(train_dst.iloc[i,:].values-train_dst.iloc[i,:].values.mean()).imag)
    test_fu_real = np.append(test_fu_real,np.fft.fft(test_dst.iloc[i,:].values-test_dst.iloc[i,:].values.mean()).real)
    test_fu_imag = np.append(test_fu_imag,np.fft.fft(test_dst.iloc[i,:].values-test_dst.iloc[i,:].values.mean()).imag)
# print(train_fu_real.shape)
# print(train_fu_imag.shape)
train_fu_real=train_fu_real.reshape(-1,35)
train_fu_imag=train_fu_imag.reshape(-1,35)
test_fu_real=test_fu_real.reshape(-1,35)
test_fu_imag=test_fu_imag.reshape(-1,35)
# train_src_rank = np.argsort(train_src)[::-1][:, :10]
# train_dst_rank = np.argsort(train_dst)[::-1][:, :10]
# test_src_rank = np.argsort(test_src)[::-1][:, :10]
# test_dst_rank = np.argsort(test_dst)[::-1][:, :10]
# print(np.array(train_src - train_dst)[:2,:])






small = 1e-30

# x_train = np.concatenate([train.values[:,0:1]**2, train_src/(train.values[:,0:1]**2), train_dst, train_src/(train.values[:,0:1]**2) - train_dst,train_src/(train.values[:,0:1]**2)/(train_dst+small)], axis = 1)
# x_pred = np.concatenate([test.values[:,0:1]**2, test_src/(train.values[:,0:1]**2), test_dst, test_src/(train.values[:,0:1]**2) - test_dst,test_src/(train.values[:,0:1]**2)/(test_dst+small)], axis = 1)

train_df = pd.DataFrame(index=list(range(10000)))
train_gap = pd.DataFrame(train_src.values - train_dst.values,columns=list(range(650,1000,10))).add_suffix('_gap')

train_ratio = pd.DataFrame(train_src.values/(train_dst.values+small),columns=list(range(650,1000,10))).add_suffix('_ratio')
train_fft_real = pd.DataFrame(train_fu_real,columns = list(range(650,1000,10))).add_suffix('_real')
train_fft_img = pd.DataFrame(train_fu_imag,columns = list(range(650,1000,10))).add_suffix('_imag')

train_df = pd.concat([train["rho"]**2,train_dst,train_gap,train_ratio,train_fft_real,train_fft_img],axis=1)
# train_df = pd.concat([train_dst,train_gap,train_ratio,train_fft_real,train_fft_img,train.loc[:,"hhb":"na"]],axis=1)

train_df.index.name = "id"


test_df = pd.DataFrame()

test_gap = pd.DataFrame(test_src.values - test_dst.values,columns=list(range(650,1000,10))).add_suffix('_gap')

test_ratio = pd.DataFrame(test_dst.values/(test_src.values+small),columns=list(range(650,1000,10))).add_suffix('_ratio')
test_fft_real = pd.DataFrame(test_fu_real,columns = list(range(650,1000,10))).add_suffix('_real')
test_fft_img = pd.DataFrame(test_fu_imag,columns = list(range(650,1000,10))).add_suffix('_imag')

test_df = pd.concat([test["rho"]**2,test_dst,test_gap,test_ratio,test_fft_real,test_fft_img],axis=1)
test_df.index=list(range(10000,20000,1))
test_df.index.name = "id"
train_df.to_csv('./data/git_train.csv')
test_df.to_csv('./data/git_test.csv')

# x_train = np.concatenate([train.values[:,0:1]**2,train_dst, train_src-train_dst, train_src/(train_dst+small), train_fu_real, train_fu_imag] , axis = 1)
# x_pred = np.concatenate([test.values[:,0:1]**2,test_dst, test_src-test_dst, test_src/(test_dst+small), test_fu_real, test_fu_imag], axis = 1)

# x_train = np.concatenate([train.values[:,0:1]**2, train_src, train_dst, train_src - train_dst,train_src/(train_dst+small)], axis = 1)
# x_pred = np.concatenate([test.values[:,0:1]**2, test_src, test_dst, test_src - test_dst,test_src/(test_dst+small)], axis = 1)

# x_train = np.concatenate([train.values[:,0:1]**2, train_src, train_dst, train_src - train_dst,train_src/(train_dst+small), np.log10((train_src + small)/(train_dst+small))], axis = 1)
# x_pred = np.concatenate([test.values[:,0:1]**2, test_src, train_dst, test_src - test_dst,test_src/(test_dst+small), np.log10((test_src+small)/(test_dst+small))], axis = 1)
# print(pd.DataFrame(x_train).isnull().sum())
# print(pd.DataFrame(np.log10(train_src.values) - np.log10(train_dst.values)))
