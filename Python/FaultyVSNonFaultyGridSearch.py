import pandas as pd
import numpy as np
from sklearn import svm
import confusion_matrix2 as confusion_matrix
from sklearn.preprocessing import StandardScaler
import sys


# Load training data
train_data1 = pd.read_csv(sys.path[0] + "/../TrainingData/neodata/11d_setpoints_1200.csv")
test_data = pd.read_csv(sys.path[0] + "/../TestData/neodata/11d_setpoints_100.csv")

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
X_trn1 = train_data.drop(['target'],axis =1 )
X_tst1 = test_data.drop(['target'],axis=1)

y_trn=train_data['target']
y_tst=test_data['target']

g_trn = train_data['setpoint']
g_tst = test_data['setpoint']

scale = StandardScaler()
X_trn = scale.fit_transform(X_trn1)
X_tst = scale.transform(X_tst1)

print(g_trn)