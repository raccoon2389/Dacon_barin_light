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
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.preprocessing import StandardScaler,RobustScaler,Normalizer,MinMaxScaler #standard : 1.01, Robust,1.5, Normalizer1.7, Minmax: 1.47
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm

#데이터 추출

# train = pd.read_csv('data/train.csv',index_col=0,header=0)
# test = pd.read_csv('data/test.csv', index_col=0,header=0)
submission = pd.read_csv('data/sample_submission.csv')

'''
#dst와 src 분할

train_dst = train.filter(regex='_dst$', axis=1)
test_dst = test.filter(regex='_dst$', axis=1)

#거리를 제곱해준다.

dst_list = list(train_dst)

for i in dst_list:
    train[i] = train[i]* (train['rho']**2)

'''

#dst와 src 업데이트

# train_dst = train.filter(regex='_dst$', axis=1)
# test_dst = test.filter(regex='_dst$', axis=1)



# train_dst = train_dst.interpolate(methods='linear', axis=1)
# test_dst = test_dst.interpolate(methods='linear', axis=1)

# # 스팩트럼 데이터에서 보간이 되지 않은 값은 앞뒤 값의 평균으로 일괄 처리한다.

# for i in range(980,640,-10):
#     train_dst.loc[train_dst[f"{i}_dst"].isnull(),f"{i}_dst"]=train_dst.loc[train_dst[f"{i}_dst"].isnull(),f"{i+10}_dst"] 
#     test_dst.loc[test_dst[f"{i}_dst"].isnull(),f"{i}_dst"]=test_dst.loc[test_dst[f"{i}_dst"].isnull(),f"{i+10}_dst"]


# train.update(train_dst) # 보간한 데이터를 기존 데이터프레임에 업데이트 한다.
# test.update(test_dst)
x = pd.read_csv('./data/git_train.csv',index_col=0,header=0)
y = np.load('./y_train.npy')
test = pd.read_csv('./data/git_test.csv',index_col=0,header=0)
x_h_train = np.load('./x_train.npy')
x_h_test = np.load('./x_pred.npy')

x_h_test = x_h_test[:,1:]
x_h_train = x_h_train[:,1:]

def train_model(x_data, y_data, k=5):
    models = []
    
    k_fold = KFold(n_splits=k, shuffle=True, random_state=123)
    
    for train_idx, val_idx in k_fold.split(x_data):        
        x_train, y_train = x_data[train_idx], y_data[train_idx] # 훈련 데이터를 kfold로 자른다
        x_val, y_val = x_data[val_idx], y_data[val_idx] # 검증용 데이터도 자름
    
        d_train = xgb.DMatrix(data = x_train, label = y_train) # 훈련 데이터를 xgb가 이용하기 쉬운 DMatrix로 변환해준다
        d_val = xgb.DMatrix(data = x_val, label = y_val)
        
        wlist = [(d_train, 'train'), (d_val, 'eval')]
        
        params = {                                          #파라미터
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'seed':777,
            'gpu_id':0,
            'tree_method':'gpu_hist'
                        }


        pa = {
            
        }   
        
        model = xgb.train(params=params, dtrain=d_train, num_boost_round=10000, verbose_eval=10000, evals=wlist,early_stopping_rounds=300) # 모델을 짜준다 
        # GridSearchCV(model
        models.append(model)
    
    return models

models={}
y_col=["hhb",'hbo2','ca','na']

print('train column : ', y_col[0])
models[y_col[0]] = train_model(x_h_train, y[:,0],10)

for label in range(1,4):
    print('train column : ', y_col[label])
    models[y_col[label]] = train_model(x.values, y[:,label],10)
    print('\n\n\n')

for col in models:
    preds = []
    for model in models[col]:
        if col == 'hhb':
            preds.append(model.predict(xgb.DMatrix(x_h_test)))
        else:
            preds.append(model.predict(xgb.DMatrix(test.values)))
    pred = np.mean(preds, axis=0)

    submission[col] = pred
submission.to_csv('Dacon_light.csv',index=False)      
