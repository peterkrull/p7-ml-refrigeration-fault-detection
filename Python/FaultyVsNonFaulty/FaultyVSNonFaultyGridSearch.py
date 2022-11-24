import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV,GroupKFold
from datetime import datetime
from joblib import dump, load



## Grid search


def false_positives(true_target : pd.DataFrame, predicted_target : pd.DataFrame):
    #Find the false positive rate
    zero_label_index = true_target.loc[true_target['target']==0]
    pred_non_fault = predicted_target[predicted_target.index.isin(zero_label_index.index)]
    false_positive = len(pred_non_fault.loc[pred_non_fault[0] != 0])/(len(pred_non_fault)+0.0001)
    return false_positive

def gridsearch_scoring(y_true : np.array, y_pred : np.array):
    #Calculate a score weighing false positives and false negatives
    sum = 0
    fp_ratio = .4       # High = False positives is unaccepteble
    fp = false_positives(pd.DataFrame(y_true, columns = ['target']), pd.DataFrame(y_pred))
    sum += (1-fp)*fp_ratio
    y_pred = pd.DataFrame(y_pred)
    accuracy = (len(y_pred) - len( y_pred[0].compare(pd.DataFrame(y_true)[0]) ) )  / (len(y_pred) + 0.0001)
    sum += (1-fp_ratio)*(accuracy)
    return sum



# Load training data
train_data1 = pd.read_csv(sys.path[0] + "/../../TrainingData/neodata/11d_setpoints_1200.csv")
test_data = pd.read_csv(sys.path[0] + "/../../TestData/neodata/11d_setpoints_100.csv")

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
X_trn1 = train_data.drop(['target'],axis =1).to_numpy()
X_tst1 = test_data.drop(['target'],axis=1).to_numpy()

y_trn=train_data['target'].to_numpy()
y_tst=test_data['target'].to_numpy()

g_trn = train_data['setpoint'].to_numpy()
g_tst = test_data['setpoint'].to_numpy()

scale = StandardScaler()
X_trn = scale.fit_transform(X_trn1)
X_tst = scale.transform(X_tst1)


#Parameters to search
C_params = [10**x for x in np.linspace(1,5, 50)]           #Logrithmic svaling of parameters
gamma_params = [10**x for x in np.linspace(-4,0, 50)]

svc = svm.SVC()

score = make_scorer(gridsearch_scoring, greater_is_better= True)

clf = GridSearchCV(svc,{'kernel':['rbf'], 'decision_function_shape':['ovo'], 'C' : C_params, 'gamma' : gamma_params}, n_jobs=-1, verbose=3, scoring = score,cv =GroupKFold(n_splits=5))
clf.fit(X_trn,y_trn,groups = g_trn)

f = open(sys.path[0] + "/gridSearchResult_lowerAlpha.txt", 'w')
f.write(str(datetime.now()) + "\n\n")
f.write(str(clf.cv_results_))
f.write('\n')
f.write(str(clf.best_estimator_))
f.close()

cv_log = pd.DataFrame.from_dict(clf.cv_results_)
f = open(sys.path[0] + "/GridSearchLog_lowerAlpha.json", 'w')
f.write(cv_log.to_json())
f.close()

dump(clf,'GridSearchBest_lowerAlpha.joblib')

"""
results = pd.read_json('GridSearchLog.json')
print(results)

import confusion_matrix2 as confusionMatrix

## Test if it works
clf_load = load('GridSearchBest.joblib')

y_tst_predict = clf_load.predict(X_tst)
confusionMatrix.confusion_matrix(y_tst,y_tst_predict,save_fig_name='confMatrix_tst.pdf')
print("Hallo")
y_trn_predict = clf_load.predict(X_trn)
confusionMatrix.confusion_matrix(y_trn,y_trn_predict,save_fig_name='confMatrix_trn.pdf')
"""

