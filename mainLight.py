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
train = pd.read_csv('./data/pre_train.csv',index_col=0)
test = pd.read_csv('./data/pre_test.csv',index_col=0)
x_train = train.drop(columns=['hhb','hbo2','ca','na'])
print(x_train)
x_train = train.loc[:,"650_dst_650_src_ratio":]
x_col = list(x_train)
y_train = train.loc[:,'hhb':'na']
y_col = list(y_train)
test = test.loc[:,"650_dst_650_src_ratio":]


kf=KFold(n_splits=7)
def train_model(x_data, y_data, k=5):
    model_zip = []
    i=0
    for train_idx,val_idx in kf.split(x_data):
        x_train, y_train = x_data.iloc[train_idx],y_data[train_idx]
        x_val,y_val = x_data.iloc[val_idx],y_data[val_idx]
        train_set = lightgbm.Dataset(data = x_train, label = y_train)
        val_set = lightgbm.Dataset(data=x_val,label=y_val)
        param = {
            'objective' : 'regression_l1',
            'num_iterations' : 1000,
            'learning_rate' : 0.07,
            'num_leaves' : 100,
            'min_data_in_leaf' : 20,
            'tree_learner' : 'serial',
            'num_thread': 6,
            'max_depth': 100,
            'max_bin' : 1000
            # 'device_type' : 'gpu'
            }
        model = lightgbm.train(params=param,train_set=train_set,num_boost_round=1000,valid_sets=val_set,valid_names=f"{i}번째 CV", verbose_eval=1000)

        model_zip.append(model)
        i+=1
    
    return model_zip

models = {}
for label in y_train.columns:
    print('train column : ', label)
    models[label] = train_model(x_train, y_train[label])
    print('\n\n\n')

for col in models:
    preds = []
    for model in models[col]:
        print(model)
        preds.append(model.predict(lightgbm.Dataset(test)))
    pred = np.mean(preds, axis=0)

    submission[col] = pred
submission.to_csv('Dacon.csv',index=False)      
