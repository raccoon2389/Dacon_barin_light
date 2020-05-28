'''
id : 구분자
rho : 측정 거리 (단위: mm)
src : 광원 스펙트럼 (650 nm ~ 990 nm)
dst : 측정 스펙트럼 (650 nm ~ 990 nm)
hhb : 디옥시헤모글로빈 농도
hbo2 : 옥시헤모글로빈 농도
ca : 칼슘 농도
na : 나트륨 농도
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM,Input
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

train = pd.read_csv('data/train.csv',index_col=0)
test = pd.read_csv('data/test.csv', index_col=0)
submission = pd.read_csv('data/sample_submission.csv')

train_dst = train.filter(regex='_dst$', axis=1).replace(0, np.NaN) # dst 데이터만 따로 뺀다.
test_dst = test.filter(regex='_dst$', axis=1).replace(0, np.NaN) # 보간을 하기위해 결측값을 삭제한다.
test_dst.head(1)

# print(train_dst)

train_dst = train_dst.interpolate(methods='linear', axis=1)
test_dst = test_dst.interpolate(methods='linear', axis=1)

# 스팩트럼 데이터에서 보간이 되지 않은 값은 0으로 일괄 처리한다.

train_dst.fillna(0, inplace=True) 
test_dst.fillna(0, inplace=True)
print(test_dst.head())

train.update(train_dst) # 보간한 데이터를 기존 데이터프레임에 업데이트 한다.
test.update(test_dst)
# df = pd.DataFrame()
# d = train['rho']

# for i in range(650,1000,10):

#     train.loc[train[f"{i}_src"]<0.1,f"{i}_src"]=1e-4
#     df[f"{i}_dst"] = train[f"{i}_dst"]
#     df[f"{i}_src"] = train[f"{i}_src"]

# print(df.head())

sclar = StandardScaler()
sclar.fit(train)
df = sclar.transform(train)
# print(df.shape)
# col = range(650,1000,10)

# x_train = df

# train.filter(regex='_src$',axis=1).head().T.plot()
# plt.figure(figsize=(4,12))test.filter(regex='_src$',axis=1).head().T.plot()
# sns.heatmap(train.corr().loc['rho':'990_dst','hhb':].abs())
# sns.heatmap(train.corr().loc['rho':'990_src','650_dst':'990_dst'].abs())
# plt.show()


# print(train.head())
# test.head()
# msno.matrix(train)
# msno.matrix(test)
# print(train.isnull().sum()[train.isnull().sum().values > 0])
# print(train.isnull().sum()[train.isnull().sum().values > 0].index)

# train_dst.head().T.plot()
# test_dst.head().T.plot()
# plt.show()
# print(train_dst.head())

# print(test_dst.head())



x_train = train.loc[:,"650_dst":"990_dst"]
y_train = train.loc[:,'hhb':]

# print(x_train.shape, y_train.shape)
skf = KFold(n_splits=5,shuffle=True)
accuracy = []
for t, v in skf.split(x_train):
    print(t)
    x_t , y_t = x_train.iloc[t], y_train.iloc[t]

    input_rho = Input(shape=(1,))
    input_src = Input(shape=(35,))
    input_dst = Input(shape=(35,))

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

    
    # model.add(Dense(50,activation='relu'))
    model.add(Dense(4))

    
    model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])

    # 학습 데이터를 이용해서 학습
    model.fit(x_t, y_t, epochs=100, batch_size=20)

    # 테스트 데이터를 이용해서 검증
    k_accuracy = '%.4f' % (model.evaluate(x_train[v], y_train[v])[1])
    accuracy.append(k_accuracy)


print('\nK-fold cross validation Accuracy: {}'.format(accuracy))
