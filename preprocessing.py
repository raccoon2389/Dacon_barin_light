'''
id : 구분자
rho : 측정 거리 (단위: mm)
src : 광원 스펙트럼 (650 nm ~ 990 nm)
dst : 측정 스펙트럼 (650 nm ~ 990 nm)
hhb : 디옥시헤모글로빈 농도
hbo2 : 옥시헤모글로빈 농도
ca : 칼슘 농도
na : 나트륨 농도


na 빠진거 모아서 모델링 하고 predict

'''

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout,LSTM,Input
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.decomposition import PCA
from keras.layers.merge import concatenate
from keras.losses import mean_absolute_error
from sklearn.model_selection import train_test_split

#데이터 추출

train = pd.read_csv('data/train.csv',index_col=0)
test = pd.read_csv('data/test.csv', index_col=0)
submission = pd.read_csv('data/sample_submission.csv')



# print(train.dropna(axis= 0).head())

# print(train.loc[:,'rho'])

# train.dropna(axis=0).filter(regex='_dst$',axis=1).head().T.plot()
# plt.show()
'''
q=0
for i in range(700,950,10):
    if i == 700: 
        tmp = train.loc[train[f"{i-50}_dst"].notnull()&train[f"{i-40}_dst"].notnull()&train[f"{i-30}_dst"].notnull()& train[f"{i-20}_dst"].notnull()& train[f"{i-10}_dst"].notnull() & train[f"{i}_dst"].notnull()& train[f"{i+10}_dst"].notnull()&train[f"{i+20}_dst"].notnull()&train[f"{i+30}_dst"].notnull() &train[f"{i+40}_dst"].notnull()&train[f"{i+50}_dst"].notnull(), f"{i-50}_dst":f"{i+50}_dst"].values
        print(tmp.shape)
    else:
        tmp = np.r_[tmp,train.loc[train[f"{i-50}_dst"].notnull()&train[f"{i-40}_dst"].notnull()&train[f"{i-30}_dst"].notnull()& train[f"{i-20}_dst"].notnull()& train[f"{i-10}_dst"].notnull() & train[f"{i}_dst"].notnull()& train[f"{i+10}_dst"].notnull()&train[f"{i+20}_dst"].notnull()&train[f"{i+30}_dst"].notnull() &train[f"{i+40}_dst"].notnull()&train[f"{i+50}_dst"].notnull(), f"{i-50}_dst":f"{i+50}_dst"].values]

temp=np.zeros(11,)

for i in tmp:
    if i.sum() !=0:
        np.array(i).reshape(11,)
        temp = np.r_[temp,i]
        print(i)

print(temp)

np.save('./data/temp.npy',arr= temp)
'''

tmp = np.load('./data/temp.npy')
tmp = tmp.reshape(-1,11)
print(tmp.shape)

scaler = MinMaxScaler()


pre_x = tmp[:,(0,1,2,3,4,6,7,8,9,10)]
pre = pre_x[1]
scaler.fit(pre_x)
pre_x = scaler.transform(pre_x)
pre_y = tmp[:,5]

x_train,x_test , y_train, y_test = train_test_split(pre_x,pre_y, test_size=0.2)

print(x_train.shape)

model = Sequential()
model.add(Dense(30,activation='relu',input_dim=x_train.shape[1]))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

model.add(Dense(1))

model.compile(optimizer='adam',loss=mean_absolute_error,metrics=['acc'])
model.fit(x_train,y_train,batch_size=100,epochs=10,validation_split=0.25)
loss = model.evaluate(x_test,y_test,batch_size=100)
print(loss)
pred = model.predict(x_test,batch_size=100)
print(pre,tmp[1],pred[1])

'''
#dst와 src 분할

train_dst = train.filter(regex='_dst$', axis=1)
test_dst = test.filter(regex='_dst$', axis=1)

#거리를 제곱해준다.

dst_list = list(train_dst)

for i in dst_list:
    train[i] = train[i]* (train['rho']**2)


#dst와 src 업데이트

train_dst = train.filter(regex='_dst$', axis=1)
test_dst = test.filter(regex='_dst$', axis=1)




# print(train_dst)

train_dst = train_dst.interpolate(methods='linear', axis=1)
test_dst = test_dst.interpolate(methods='linear', axis=1)
# train_dst.loc[train_dst[f"650_dst"].isnull(),'650_dst']
print(test_dst.loc[:,'650_dst':].isnull().sum())

# print(train_dst.isnull().sum())


print(train_dst.isnull().sum())

print(test_dst.isnull().sum())

# 스팩트럼 데이터에서 보간이 되지 않은 값은 앞뒤 값의 평균으로 일괄 처리한다.

for i in range(980,640,-10):
    train_dst.loc[train_dst[f"{i}_dst"].isnull(),f"{i}_dst"]=train_dst.loc[train_dst[f"{i}_dst"].isnull(),f"{i+10}_dst"] 
    test_dst.loc[test_dst[f"{i}_dst"].isnull(),f"{i}_dst"]=test_dst.loc[test_dst[f"{i}_dst"].isnull(),f"{i+10}_dst"]



# print(test_dst.head())
# 
train.update(train_dst) # 보간한 데이터를 기존 데이터프레임에 업데이트 한다.
test.update(test_dst)
# 



# print(df.shape)
# col = range(650,1000,10)

# x_train = df

#데이터 분석

# train.filter(regex='_src$',axis=1)[16:20].T.plot()
# train.filter(regex='_dst$',axis=1)[0:5].T.plot()
# plt.figure(figsize=(4,12))

# sns.heatmap(train.corr().loc['rho':'990_dst','hhb':].abs())
# sns.heatmap(train.corr().loc['650_dst':'990_dst','650_src':'990_src'].abs())
# sns.heatmap(train.corr().loc['650_src':'990_src','650_dst':'990_dst'].abs().plot())
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
'''