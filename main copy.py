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
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout,LSTM,Input
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.decomposition import PCA
from keras.layers.merge import concatenate
from keras.losses import mean_absolute_error

e_stop = EarlyStopping(monitor='loss',patience=30,mode='auto')

m_check = ModelCheckpoint(filepath=".\model\dacon--{epoch:02d}--{val_loss:.4f}.hdf5", monitor = 'val_loss',save_best_only=True)

#데이터 추출

train = pd.read_csv('data/train.csv',index_col=0)
test = pd.read_csv('data/test.csv', index_col=0)
submission = pd.read_csv('data/sample_submission.csv')

#dst와 src 분할

train_dst = train.filter(regex='_dst$', axis=1)
test_dst = test.filter(regex='_dst$', axis=1)
train_src = train.filter(regex='_src$', axis=1)

#거리를 제곱해준다.

dst_list = list(train_dst)
src_list = list(train_src)

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

#src -dst gap
gap_feature = []

for i in range(650,1000,10):
    gap_feature.append(str(i)+'_gap')

alpha=pd.DataFrame(np.array(train[src_list]) - np.array(train[dst_list]), columns=gap_feature, index=train.index)
beta =pd.DataFrame(np.array(test[src_list])-np.array(test[dst_list]), columns=gap_feature, index = test.index)

train = pd.concat((train,alpha), axis=1)
test = pd.concat((test, beta),axis=1)

eps = 1e-10

for dst_col, src_col in zip(dst_list,src_list):
    dst_val = train[dst_col]
    src_val = train[src_col]+ eps
    delta_ratio = dst_val / src_val
    train[f"{dst_col}_{src_col}_ratio"]= delta_ratio
    dst_val = test[dst_col]
    src_val = test[src_col]+ eps
    delta_ratio = dst_val / src_val
    test[f"{dst_col}_{src_col}_ratio"]= delta_ratio


alpha_real = train[dst_list]
alpha_img = train[dst_list]

beta_real = test[dst_list]
beta_img = test[dst_list]

for i in alpha.index:
    alpha_real.loc[i] = alpha_real.loc[i] - alpha_real.loc[i].mean()
    alpha_img.loc[i] = alpha_img.loc[i] - alpha_img.loc[i].mean()

    alpha_real.loc[i] = np.fft.fft(alpha_real.loc[i], norm='ortho').real
    alpha_img.loc[i] = np.fft.fft(alpha_real.loc[i], norm='ortho').imag

for i in beta_real.index:
    beta_real.loc[i] = beta_real.loc[i] - beta_real.loc[i].mean()
    beta_img.loc[i] = beta_img.loc[i] - beta_img.loc[i].mean()

    beta_real.loc[i] = np.fft.fft(beta_real.loc[i], norm='ortho').real
    beta_img.loc[i] = np.fft.fft(beta_real.loc[i], norm='ortho').imag

real_part =[]
img_part = []

for col in dst_list:
    real_part.append(col + '_fft_real')
    img_part.append(col + '_fft_img')

alpha_real.columms = real_part
alpha_img.columns = img_part
alpha = pd.concat((alpha_real, alpha_img),axis=1)

beta_real.columms = real_part
beta_img.columns = img_part
beta = pd.concat((beta_real, beta_img),axis=1)

train = pd.concat((train,alpha), axis=1)
test = pd.concat((test,beta),axis=1)

train = train.drop(columns=src_list)
test = test.drop(columns=src_list)

print(train)

train.to_csv('./data/pre_train.csv')
test.to_csv('./data/pre_test.csv')


'''
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


x_train = train.loc[:,'rho':'990_dst']
x_col = list(x_train)
print(x_col)
y_train = train.loc[:,'hhb':]
y_col = list(y_train)
sclar = StandardScaler()
sclar.fit(x_train)
x_train = sclar.transform(x_train)
test = test.loc[:,'rho':]
test =sclar.transform(test)
x_train = pd.DataFrame(x_train,columns=x_col)
test = pd.DataFrame(test, columns=x_col)

test_src = test.loc[:,'650_src':'990_src']
test_rho = test.loc[:,'rho']
test_dst = test.loc[:,'650_dst':'990_dst']
'''


'''
# print(x_train.shape, y_train.shape)
skf = KFold(n_splits=5,shuffle=True)
accuracy = []
for t, v in skf.split(x_train):
    x_t , y_t = x_train.iloc[t], y_train.iloc[t]
    x_v = x_train.iloc[v]

    # print(x_t.head())
    # train_rho = x_t.loc[:,'rho']
    train_src = x_t.loc[:,'650_src':'990_src']
    train_dst = x_t.loc[:,'650_dst':'990_dst']
    # valid_rho = x_v.loc[:,'rho']
    valid_src = x_v.loc[:,'650_src':'990_src']
    valid_dst = x_v.loc[:,'650_dst':'990_dst']

    # rho = Input(shape = (1,))
    dst = Input(shape= (35,))
    src = Input(shape= (35,))

    dense1 = Dense(1000,activation='relu')(dst)
    dense1 = Dropout(0.4)(dense1)
    dense2 = Dense(1000,activation='relu')(src)
    dense2 = Dropout(0.4)(dense2)



    # merge = concatenate([rho,dense1,dense2])
    merge = concatenate([dense1,dense2])

    output1 = Dense(500,activation='relu')(merge)
    output1 = Dropout(0.4)(output1)

    output1 = Dense(4)(output1)

    # model = Model(inputs=[rho,src,dst], outputs=[output1])
    model = Model(inputs=[src,dst], outputs=[output1])

    



    model.compile(loss=mean_absolute_error, optimizer='adam',metrics=['accuracy'])

    # 학습 데이터를 이용해서 학습
    # model.fit([train_rho,train_src,train_dst], y_t, epochs=200, batch_size=20)
    hist = model.fit([train_src,train_dst], y_t, epochs=500, batch_size=100, callbacks=[m_check], validation_data=[[valid_src,valid_dst],y_train.iloc[v]])

    # plt.plot(hist.history['loss'])
    # plt.show()
    # 테스트 데이터를 이용해서 검증
    # k_accuracy = '%.4f' % (model.evaluate([valid_rho,valid_src,valid_dst], y_train.iloc[v])[1])
    k_accuracy = '%.4f' % (model.evaluate([valid_src,valid_dst], y_train.iloc[v])[1])
    accuracy.append(k_accuracy)


print('\nK-fold cross validation Accuracy: {}'.format(accuracy))
'''