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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler,RobustScaler,Normalizer,MinMaxScaler #standard : 1.01, Robust,1.5, Normalizer1.7, Minmax: 1.47
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA


#데이터 추출

train = pd.read_csv('data/train.csv',index_col=0,header=0)
test = pd.read_csv('data/test.csv', index_col=0,header=0)
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

train_dst = train.filter(regex='_dst$', axis=1)
test_dst = test.filter(regex='_dst$', axis=1)




# print(train_dst)

train_dst = train_dst.interpolate(methods='linear', axis=1)
test_dst = test_dst.interpolate(methods='linear', axis=1)
# train_dst.loc[train_dst[f"650_dst"].isnull(),'650_dst']
# print(test_dst.loc[:,'650_dst':].isnull().sum())

# print(train_dst.isnull().sum())


# print(train_dst.isnull().sum())

# print(test_dst.isnull().sum())

# 스팩트럼 데이터에서 보간이 되지 않은 값은 앞뒤 값의 평균으로 일괄 처리한다.

for i in range(980,640,-10):
    train_dst.loc[train_dst[f"{i}_dst"].isnull(),f"{i}_dst"]=train_dst.loc[train_dst[f"{i}_dst"].isnull(),f"{i+10}_dst"] 
    test_dst.loc[test_dst[f"{i}_dst"].isnull(),f"{i}_dst"]=test_dst.loc[test_dst[f"{i}_dst"].isnull(),f"{i+10}_dst"]


# print(test_dst.head())
# 
train.update(train_dst) # 보간한 데이터를 기존 데이터프레임에 업데이트 한다.
test.update(test_dst)
# drop_col = train[train.loc[:,'650_dst':'990_dst'].sum()].index
# train = train.drop(drop_col)
# print(train.head())
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
x_train = train.loc[:,'rho':'990_dst']


def plot_feature_importance(model,target_data,col):
    n_features = target_data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features),col)
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

x_col = list(x_train)
y_train = train.loc[:,'hhb':]
y_col = list(y_train)
test = test.loc[:,'rho':'990_dst'].values

x_train, x_test, y_train, y_test = train_test_split(x_train.values,y_train.values,test_size=0.2,random_state=66)

# ###################### Decision Tree ##############################
# dc = DecisionTreeRegressor(max_depth=10)
# dc.fit(x_train,y_train)
# score = dc.score(x_test,y_test)
# print("Decision Tree : ", score)
# y1 = dc.predict(test.values)
# df = pd.DataFrame(y1,index=range(10000,20000,1),columns=['hhb','hbo2','ca','na'])
# df.to_csv('./DC.csv')

# #Decision Tree :  -0.0653147515470097


# ######################   Random Forest ############################################
# ranfo = RandomForestRegressor(max_depth=10,n_estimators=200,n_jobs=-1)


# ranfo.fit(x_train,y_train)


# score = ranfo.score(x_test,y_test)
# print("Random Forest : ",score)
# plot_feature_importance(ranfo,x_train,x_col)

# plt.show()

# y2 = ranfo.predict(test.values)
# df = pd.DataFrame(y2,index=range(10000,20000,1),columns=['hhb','hbo2','ca','na'])

# df.to_csv('./Ranfor.csv')

# #Random Forest :  -0.006048530535375656

# ###################################################################################

#################### Gradient Boost   ##################
'''
# gb = GradientBoostingRegressor(max_depth=10,n_estimators=200)

gb = MultiOutputRegressor(GradientBoostingRegressor(max_depth=10,n_estimators=200),n_jobs=-1)
gb.fit(x_train,y_train)

score = gb.score(x_test,y_test)

print("Gradient Boosting : ", score)

y3 = gb.predict(test.values)
df = pd.DataFrame(y3,index=range(10000,20000,1),columns=['hhb','hbo2','ca','na'])

df.to_csv('./DC.csv')
'''
########################################################

################### XGB #################
# model = MultiOutputRegressor(xgb.XGBRFRegressor())
# model.fit(x_train,y_train)
# score = model.score(x_test,y_test)
# print(score)
# y4 = model.predict(test.values)
# df = pd.DataFrame(y4,index=range(10000,20000,1),columns=['hhb','hbo2','ca','na'])

# df.to_csv('./DC.csv')
##################################################

################### PCA ###############

model = xgb.XGBRFRegressor()
pca = PCA(n_components=1)
pca.fit(y_train)
y_train = pca.transform(y_train)
y_test = pca.transform(y_test)
print(test.shape,x_train.shape)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("score : ",score)

pred = model.predict(test).reshape(-1,1)

pred =pca.inverse_transform(pred)


#######################################



# def train_model(x_data, y_data, k=5):
#     models = []
    
#     k_fold = KFold(n_splits=k, shuffle=True, random_state=123)
    
#     for train_idx, val_idx in k_fold.split(x_data):
#         x_train, y_train = x_data.iloc[train_idx], y_data.values[train_idx]
#         x_val, y_val = x_data.iloc[val_idx], y_data[val_idx]
    
#         d_train = xgb.DMatrix(data = x_train, label = y_train)
#         d_val = xgb.DMatrix(data = x_val, label = y_val)
        
#         wlist = [(d_train, 'train'), (d_val, 'eval')]
        
#         params = {
#             'objective': 'reg:squarederror',
#             'eval_metric': 'mae',
#             'seed':777
#             }

#         model = xgb.train(params=params, dtrain=d_train, num_boost_round=500, verbose_eval=500, evals=wlist)
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
# submission.to_csv('Dacon.csv',index=False)