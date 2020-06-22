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
from numpy.random import randint
# from keras.models import Sequential, Model,load_model
# from keras.layers import Dense, Dropout,LSTM,Input
from sklearn.model_selection import KFold, cross_val_score, train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
# from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.decomposition import PCA
# from keras.layers.merge import concatenate
# from keras.losses import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn import metrics
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
# e_stop = EarlyStopping(monitor='loss',patience=30,mode='auto')

# m_check = ModelCheckpoint(filepath=".\model\dacon--{epoch:02d}--{val_loss:.4f}.hdf5", monitor = 'val_loss',save_best_only=True)
'''
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




train = pd.read_csv('./data/pre_train.csv',index_col=0,header=0)
test = pd.read_csv('./data/pre_test.csv',index_col=0,header=0)
submission = pd.read_csv('./data/sample_submission.csv',index_col=0,header=0)



y_train = train.loc[:,'hhb':'na']
x_train = train.drop(['hhb','hbo2','ca','na'],axis=1)
print(x_train)
x_pred = train.drop(['hhb','hbo2','ca','na'],axis=1)


# print(df.shape)
# col = range(650,1000,10)

# x_train = df

#데이터 분석

# train.filter(regex='_src$',axis=1)[16:20].T.plot()
# train.filter(regex='_dst$',axis=1)[0:5].T.plot()
# plt.figure(figsize=(4,12))

sns.heatmap(train.corr().loc['hhb':'na','990_gap':].abs())
# sns.heatmap(train.corr().loc['650_dst':'990_dst','650_src':'990_src'].abs())
# sns.heatmap(train.corr().loc[:,'rho':].abs().plot())
plt.show()
# na = 730 910


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



# pca = PCA(n_components=20)
# pca.fit(x_train)
# x_train = pca.transform(x_train)
# x_pred = pca.transform(x_pred)

# scal = MinMaxScaler()
# scal.fit(x_train)
# x_train =scal.transform(x_train)
# x_pred = scal.transform(x_pred)


# ml = RandomForestRegressor()

# ml.fit(x_train,y_train)
#############################################
'''
kf = KFold(shuffle=True)

x_tr , x_te, y_tr, y_te = train_test_split(x_train, y_train, test_size= 0.25, shuffle = True)

multi_ran_fo = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
ran_fo = RandomForestRegressor(random_state=66)
# scores = cross_val_score(multi_ran_fo,x_train,y_train,cv=kf,n_jobs=-1)
multi_ran_fo.fit(x_tr,y_tr)
scores = multi_ran_fo.score(x_te,y_te)
y = multi_ran_fo.predict(x_te)
print(scores)

'''

#############################################
# def train_model(x_data, y_data, k=5):
#     models = []
    
#     k_fold = KFold(n_splits=k, shuffle=True, random_state=123)
    
#     for train_idx, val_idx in k_fold.split(x_data):
#         x_train, y_train = x_data.iloc[train_idx], y_data.values[train_idx] # 훈련 데이터를 kfold로 자른다
#         x_val, y_val = x_data.iloc[val_idx], y_data[val_idx] # 검증용 데이터도 자름
    
#         d_train = xgb.DMatrix(data = x_train, label = y_train) # 훈련 데이터를 xgb가 이용하기 쉬운 DMatrix로 변환해준다
#         d_val = xgb.DMatrix(data = x_val, label = y_val)
        
#         wlist = [(d_train, 'train'), (d_val, 'eval')]
        
#         params = {                                          #파라미터
#             'objective': 'reg:squarederror',
#             'eval_metric': 'mae',
#             'seed':777,
#             'num_boost_round':randint(100,5000,30).tolist(),
#             'n_estimators' : randint(500,10000,30).tolist(),
#             'max_depth' : randint(2,40,30).tolist(),
#             'learning_rate' : np.linspace(0.001,2,100).tolist()

#             }
#         # model = xgb.train(params=params, dtrain=d_train, num_boost_round=1000, verbose_eval=1000, evals=wlist) # 모델을 짜준다
#         model =  RandomizedSearchCV(xgb.XGBRFRegressor(),param_distributions=params,n_iter=41,n_jobs=-1)
#         model.fit(d_train)
#         models.append(model)
    
#     return models

# models = {}
# for label in y_train.columns:
#     print('train column : ', label)
#     models[label] = train_model(x_train, y_train[label])
#     print('\n\n\n')

# for col in models:
#     preds = []
#     for model in models[col]:
#         preds.append(model.predict(xgb.DMatrix(test)))
#     pred = np.mean(preds, axis=0)

#     submission[col] = pred
# submission.to_csv('Dacon.csv',index=True)

'''
# print(x_train.shape, y_train.shape)
skf = KFold(n_splits=5,shuffle=True)
accuracy = []
for t, v in skf.split(x_train):
    x_t , y_t = x_train[t], y_train[t]
    x_v , y_v = x_train[v], y_train[v]

    # rho = Input(shape = (1,))
    dst = Input(shape= (x_train.shape[1],))

    dense1 = Dense(1000,activation='relu')(dst)
    dense1 = Dropout(0.4)(dense1)

    output1 = Dense(500,activation='relu')(dense1)
    output1 = Dropout(0.4)(output1)

    output1 = Dense(4)(output1)

    # model = Model(inputs=[rho,src,dst], outputs=[output1])
    model = Model(inputs=[dst], outputs=[output1])

    



    model.compile(loss=mean_absolute_error, optimizer='adam',metrics=['accuracy'])

    # 학습 데이터를 이용해서 학습
    # model.fit([train_rho,train_src,train_dst], y_t, epochs=200, batch_size=20)
    hist = model.fit(x_t, y_t, epochs=500, batch_size=100, callbacks=[m_check], validation_data=[x_t,y_v])

    # plt.plot(hist.history['loss'])
    # plt.show()
    # 테스트 데이터를 이용해서 검증
    # k_accuracy = '%.4f' % (model.evaluate([valid_rho,valid_src,valid_dst], y_train.iloc[v])[1])
    k_accuracy = '%.4f' % (model.evaluate(x_v, y_v)[1])
    accuracy.append(k_accuracy)


print('\nK-fold cross validation Accuracy: {}'.format(accuracy))
'''