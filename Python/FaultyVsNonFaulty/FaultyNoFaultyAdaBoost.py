import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.ensemble import AdaBoostClassifier
from joblib import dump, load

## Preprocessing ############################

# Load training data
train_data1 = pd.read_csv(sys.path[0] + "/../../TrainingData/neodata/14d_setpoints_1200.csv")
test_data = pd.read_csv(sys.path[0] + "/../../TestData/neodata/14d_setpoints_100.csv")

# Setting all fault targets to 1
train_data1.loc[train_data1['target']!=0,'target'] = 1
test_data.loc[test_data['target']!=0,'target'] = 1

# Set the amount of faulty and non-faulty data equal
train_data_fault = train_data1.loc[train_data1['target']!=0]
train_data_fault2 = train_data_fault.sample(1200)                   #Taking 1200 sampels of the faulty data
train_data_fault2['target'] = 1                                     #Setting target to 1

train_data_NoFault = train_data1.loc[train_data1['target']==0]      #Taking 1200 sampels of non-faulty
train_data_NoFault2 = train_data_NoFault.copy()

train_data = pd.concat([train_data_fault2,train_data_NoFault2])

#Nameing and standardization
X_trn1 = train_data.drop(['target','setpoint'],axis =1).to_numpy()
X_tst1 = test_data.drop(['target','setpoint'],axis=1).to_numpy()

y_trn=train_data['target'].to_numpy()
y_tst=test_data['target'].to_numpy()

g_trn = train_data['setpoint'].to_numpy()
g_tst = test_data['setpoint'].to_numpy()

scale = StandardScaler()
X_trn = scale.fit_transform(X_trn1)
X_tst = scale.transform(X_tst1)

## Training ######################

clf_base = svm.SVC(probability = True,kernel="rbf", decision_function_shape="ovo", C=15264.179671, gamma=0.002947051702)
clf = AdaBoostClassifier(n_estimators=16, base_estimator=clf_base, learning_rate=1)
clf.fit(X_trn,y_trn)

dump(clf,'AdaBoost.joblib')

## Test ##

import confusion_matrix2 as confusionMatrix

clf_load = load('AdaBoost.joblib')
y_trn_predict = clf_load.predict(X_trn)
confusionMatrix.confusion_matrix(y_trn,y_trn_predict,save_fig_name='AdaBoost_trn.pdf')

y_tst_predict = clf_load.predict(X_tst)
confusionMatrix.confusion_matrix(y_tst,y_tst_predict,save_fig_name='AdaBoost_tst.pdf')

