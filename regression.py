import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest


pd.set_option('display.max_columns', 60)
np.set_printoptions(threshold=np.nan)
rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
isf = IsolationForest(max_samples=0.25, random_state=11, contamination=0.15, n_estimators=1000, n_jobs=-1)

dataTrain = pd.read_excel('data/Train111.xlsx')
dataTrain = dataTrain.drop('ID', 1)
dataTest = pd.read_excel('data/Test.xlsx')
dataTest = dataTest.drop('ID', 1)
# print(dataTrain.corr())
# MinMaxScaler().fit_transform(dataTrain)
first_quartile = dataTrain['TARGET'].describe()['25%']
second_quartile = dataTrain['TARGET'].describe()['50%']
third_quartile = dataTrain['TARGET'].describe()['75%']
#
# # Interquartile range
# iqr = third_quartile - first_quartile
#
# # Remove outliers

dataTrain = dataTrain[(dataTrain['TARGET'] < second_quartile)]

# print(dataTrain.describe())
y_train = dataTrain[['TARGET']]
# X_train = dataTrain.drop(['TARGET'], axis=1)
X_train = dataTrain[['NUM_6', 'NUM_10', 'CAT_2', 'NUM_17', 'NUM_21', 'NUM_22', 'NUM_13', 'CAT_3',
                     'NUM_11', 'CAT_9', 'CAT_20', 'CAT_23', 'CAT_7', 'NUM_9']]
y_test = dataTest[['TARGET']]

X_test = dataTest[['NUM_6', 'NUM_10', 'CAT_2', 'NUM_17', 'NUM_21', 'NUM_22', 'NUM_13', 'CAT_3',
                   'NUM_11', 'CAT_9', 'CAT_20', 'CAT_23', 'CAT_7', 'NUM_9']]


# Среднее значение
# mean = X_train.mean(axis=0)
# # Стандартное отклонение
# std = X_train.std(axis=0)
# X_train -= mean
# X_train /= std
# X_test -= mean
# X_test /= std
# print(X_train.corr())

# isf.fit(y_train)

# dataTrain1 = isf.predict(y_train)
# print(np.shape(dataTrain1), dataTrain1)

# kk = 0
#
# # print(dataTrain)
# for i in range(len(dataTrain1)):
#     if dataTrain1[i] == -1:
#         print(i + 1)
#         dataTrain = dataTrain.drop([i], 0, errors='ignore')
#         kk += 1
# dataTrain.boxplot(column=['NUM_6'])
# plt.show()
rf.fit(X_train, y_train)
y_test_arr = np.array(dataTest['TARGET'])
y_predicted = rf.predict(X_test)

k = 0
metric_arr = y_test_arr / y_predicted * 100
for i in metric_arr:
    if 120 > i > 80:
        k += 1

print(np.shape(metric_arr), "\n", k, "\n", k * 100 / len(metric_arr))
