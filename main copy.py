import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import KFold
import pandas as pd

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
submission = pd.read_csv('data/sample_submission.csv')

# print(train.head())
# test.head()

x_dst = train.loc[:, '650_dst':'990_dst']
x_src = train.loc[:,'650_src':'990_src']
y_train = train.loc[:, 'hhb':'na']
col = np.arange(650,1000,10)

x_dst.columns = col
x_src.columns = col

# x_dst = x_dst.fillna(x_dst.mean())

# x_src = x_src.fillna(x_src.mean())

# print(x_src.head())


x_dst_src = x_dst / x_src
# print(x_dst.head())

x_dst_src = x_dst_src.replace([np.inf, -np.inf], np.nan).fillna(x_dst_src.mean())

y_train = y_train.replace([np.inf, -np.inf], np.nan).fillna(y_train.mean())

print(y_train)
'''

x_train = x_dst_src

print(x_train.shape, y_train.shape)
skf = KFold()
accuracy = []
for train, validation in skf.split(x_train):
    x_t , y_t = x_train.iloc[train], y_train.iloc[train]

    model = Sequential()
    model.add(Dense(1000,input_dim=(35),activation='relu'))
    # model.add(Dense(50,activation='relu'))
    model.add(Dense(4))

    
    model.compile(loss='mae', optimizer='adam',
                metrics=['accuracy'])

    # 학습 데이터를 이용해서 학습
    model.fit(x_t, y_t, epochs=100, batch_size=5)

    # 테스트 데이터를 이용해서 검증
    k_accuracy = '%.4f' % (model.evaluate(x_train[validation], y_train[validation])[1])
    accuracy.append(k_accuracy)


print('\nK-fold cross validation Accuracy: {}'.format(accuracy))
'''