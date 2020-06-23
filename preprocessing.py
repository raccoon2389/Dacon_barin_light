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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import RobustScaler



#데이터 추출

train = pd.read_csv('data/train.csv',index_col=0)
test = pd.read_csv('data/test.csv', index_col=0)
submission = pd.read_csv('data/sample_submission.csv')
# na1 = train[train.loc[:,'650_src':'990_src'].idxmax(axis=1)=='710_src']
# na2 = train[train.loc[:,'650_src':'990_src'].idxmax(axis=1)=='910_src']
# na_train_idx = pd.concat([na1,na2],axis=0).index
# print(na_train_idx)
#dst와 src 분할
# train = train.loc[train['rho']==25]
# test = test.loc[test['rho']==25]

train_dst = train.filter(regex='_dst$', axis=1)
test_dst = test.filter(regex='_dst$', axis=1)
train_src = train.filter(regex='_src$', axis=1)

# print(train_src[train_src.sum(axis=1)<5])
#거리를 제곱해준다.

dst_list = list(train_dst)
src_list = list(train_src)

for i in dst_list:
    train[i] = train[i]* (train['rho']**2)


#dst와 src 업데이트

train_dst = train.filter(regex='_dst$', axis=1)
test_dst = test.filter(regex='_dst$', axis=1)
# print(train_dst)

train_dst = train_dst.interpolate(methods='cubic', axis=1)
test_dst = test_dst.interpolate(methods='cubic', axis=1)
 
train_dst.loc[train_dst[f"650_dst"].isnull(),'650_dst']




# 스팩트럼 데이터에서 보간이 되지 않은 값은 앞뒤 값의 평균으로 일괄 처리한다.

for i in range(980,640,-10):
    train_dst.loc[train_dst[f"{i}_dst"].isnull(),f"{i}_dst"]=train_dst.loc[train_dst[f"{i}_dst"].isnull(),f"{i+10}_dst"] 
    test_dst.loc[test_dst[f"{i}_dst"].isnull(),f"{i}_dst"]=test_dst.loc[test_dst[f"{i}_dst"].isnull(),f"{i+10}_dst"]




# print(test_dst.head())
# 
train.update(train_dst) # 보간한 데이터를 기존 데이터프레임에 업데이트 한다.
test.update(test_dst)
# print(test_dst.loc[:,'650_dst':].isnull().sum())



# print(train_dst.isnull().sum())


# print(train_dst.isnull().sum())
# 
# print(test_dst.isnull().sum())

def outliers(data_out):
    out = []
    count = 0
    if str(type(data_out))== str("<class 'numpy.ndarray'>"):
        for col in range(data_out.shape[1]):
            data = data_out[:,col]
            print(data)

            quartile_1, quartile_3 = np.percentile(data,[25,75])
            print("1사분위 : ",quartile_1)
            print("3사분위 : ",quartile_3)
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr*1.5)
            upper_bound = quartile_3 + (iqr*1.5)
            out_col = np.where((data>upper_bound)|(data<lower_bound))
            print(out_col)
            data = data[out_col]
            print(f"{col+1}번째 행렬의 이상치 값: ", data)
            out.append(out_col)
            count += len(out_col)

    if str(type(data_out))== str("<class 'pandas.core.frame.DataFrame'>"):
        i=0
        print(data_out.columns)
        for col in data_out.columns:
            data = data_out[col].values
            # print(data)
            quartile_1, quartile_3 = np.percentile(data,[25,75])
            # print("1사분위 : ",quartile_1)
            # print("3사분위 : ",quartile_3)
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr*1.5)
            upper_bound = quartile_3 + (iqr*1.5)
            out_col = np.where((data>upper_bound)|(data<lower_bound))
            # print(out_col)
            print(out_col)
            data_out.iloc[out_col[0],i]='Nan'
            data = data[out_col]
            print(f"{col}의 이상치값: ", data)
            out.append(out_col)
            # print()
            i+=1
    return data_out


a = outliers(train)
print(a)

'''
#src -dst gap
gap_feature = []    

for i in range(650,1000,10):
    gap_feature.append(str(i)+'_gap')

alpha=pd.DataFrame(np.array(train[src_list]) - np.array(train[dst_list]), columns=gap_feature, index=train.index)
beta =pd.DataFrame(np.array(test[src_list])-np.array(test[dst_list]), columns=gap_feature, index = test.index)

train = pd.concat((train,alpha), axis=1)
test = pd.concat((test, beta),axis=1)

alpha=pd.DataFrame()
beta =pd.DataFrame()

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

for i in train.index:
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
# train.iloc[na_train_idx,:].to_csv('./data/pre_na.csv')
train.to_csv('./data/pre_train.csv')
test.to_csv('./data/pre_test.csv')
'''


