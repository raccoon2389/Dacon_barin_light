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
from sklearn.feature_selection import SelectFromModel
import pickle
from sklearn.metrics import mean_absolute_error
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

params = {                                          #파라미터
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'seed':777,
            'gpu_id':0,
            'tree_method':'gpu_hist'
                        }
def train_model(x_data, y_data, k=2):
    kf=KFold(n_splits=k)
    b_model = xgb.XGBRegressor()
    b_model.fit(x_data,y_data)
    thresholds = np.sort(b_model.feature_importances_)
    print(thresholds)

    thresholds = thresholds[thresholds>0]
    thresholds = thresholds[:60]
    print(thresholds)
    model_in_thres = []
    score_in_thres =[]
    f_imp=[]
    i=0

    for thres in thresholds:
        selects = SelectFromModel(b_model,threshold=thres,prefit=True)
        select_x_train = selects.transform(x_data)
        model_in_kf = []
        score_in_kf = []
        for train_idx,val_idx in kf.split(select_x_train):
            
            x_train, y_train = select_x_train[train_idx],y_data[train_idx]
            x_val,y_val = select_x_train[val_idx],y_data[val_idx]
            train_set =xgb.DMatrix(data = x_train, label = y_train)
            val_set = xgb.DMatrix(data=x_val,label=y_val)
            wlist = [(train_set, 'train'), (val_set, 'eval')]
            model = xgb.train(params=params,dtrain=train_set,num_boost_round=1000,evals=wlist, verbose_eval=1000,early_stopping_rounds=100)
            a = model.predict(xgb.DMatrix(x_val))
            score = mean_absolute_error(y_val,a)
            t = selects.transform(test.values)
            b = model.predict(xgb.DMatrix(t))

            # print(a)
            score_in_kf.append(score)
            model_in_kf.append(model)
            print(b)
            print(i,score_in_kf)
        score_in_thres.append(np.mean(score_in_kf))
        model_in_thres.append(model_in_kf)
        i+=1

        f_imp.append(selects)
    score_in_thres = np.array(score_in_thres)
    best_idx = np.argmin(score_in_kf)
    best_models = model_in_thres[best_idx]
    # best_models= [0,1]
    # best_selects= f_imp[0]
    best_selects= f_imp[best_idx]
    out = [best_models,best_selects]
    
    return out

models={}
y_col=["hhb",'hbo2','ca','na']


for label in range(0,4):
    print('train column : ', y_col[label])
    models[y_col[label]] = train_model(x.values, y[:,label],3)
    print('\n\n\n')

print(models["hhb"])

pickle.dump(models,open("./emergency.pickle","wb"))


sel = pickle.load(open('./emergency.pickle','rb'))
models = list(sel.values())

def train_model(x_data, y_data,selects ,k=2, idx=0):
    kf = KFold(n_splits=k)
    models = []
    i=0
    selects = selects[1]
    print(selects)

    select_x_train = selects.transform(x_data)
    
        
    for train_idx,val_idx in kf.split(select_x_train):
        
        x_train, y_train = select_x_train[train_idx],y_data[train_idx]
        x_val,y_val = select_x_train[val_idx],y_data[val_idx]
        train_set = xgb.DMatrix(data = x_train, label = y_train)
        val_set = xgb.DMatrix(data=x_val,label=y_val)
        print(select_x_train.shape[1])
        model = xgb.train(params=params,dtrain=train_set,num_boost_round=1000,evals=wlist, verbose_eval=10000,early_stopping_rounds=30)
        
        models.append(model)
        i+=1
    
    
    return models

models={}
y_col=["hhb",'hbo2','ca','na']


for label in range(0,4):
    print(y_col[label])
    print('train column : ', y_col[label])
    models[y_col[label]] = train_model(x.values, y[:,label],sel[y_col[label]],5,idx = label)
    print('\n\n\n')


pickle.dump(models,open('save_models.pickle','wb'))

idx=0

models = pickle.load(open('./save_models.pickle','rb'))
print(models)

for col in models:
    preds = []
    select = sel[y_col[idx]][1]
    select_test = select.transform(test.values)
    for model in models[col]:
        pre = model.predict(select_test)
        print(pre)
        preds.append(pre)
    pred = np.mean(preds, axis=0)
    idx +=1

    submission[col] = pred
submission.to_csv('Dacon_model_sel.csv',index=False)      
