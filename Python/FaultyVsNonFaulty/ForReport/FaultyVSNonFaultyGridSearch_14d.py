import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV,GroupKFold
from datetime import datetime
from joblib import dump, load
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Set correct working directory
import sys
sys.path.append(sys.path[0] + "/../../../Python")

import confusion_matrix2 as confusionMatrix

####    Load and scale data    ####
train_data1 = pd.read_csv(sys.path[0] + "/../../../TrainingData/neodata/14d_setpoints_1200.csv")
test_data = pd.read_csv(sys.path[0] + "/../../../TestData/neodata/14d_setpoints_100.csv")

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


####    Score functions and grid search settings    ####
#Calculate share of false positives
def false_positives(true_target : pd.DataFrame, predicted_target : pd.DataFrame):
    #Find the false positive rate
    zero_label_index = true_target.loc[true_target['target']==0]
    pred_non_fault = predicted_target[predicted_target.index.isin(zero_label_index.index)]
    false_positive = len(pred_non_fault.loc[pred_non_fault[0] != 0])/(len(pred_non_fault)+0.0001)
    return false_positive

#Calculate score
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
#Grid search parameters
C_params = [10**x for x in np.linspace(1,5, 1)]           #Logrithmic svaling of parameters
gamma_params = [10**x for x in np.linspace(-4,0, 1)]

####    Grid Search    ####
for dim in range(1,15):

    #PCA dim reduction
    reducer = PCA(n_components=dim)
    X_trn_pca = reducer.fit_transform(X_trn)
    X_tst_pca = reducer.fit_transform(X_tst)

    #Preform gridSearch
    svc = svm.SVC()
    score = make_scorer(gridsearch_scoring, greater_is_better= True)
    clf = GridSearchCV(svc,{'kernel':['rbf'], 'decision_function_shape':['ovo'], 'C' : C_params, 'gamma' : gamma_params}, n_jobs=-1, verbose=3, scoring = score,cv =GroupKFold(n_splits=5))
    clf.fit(X_trn,y_trn,groups = g_trn)

    f = open(sys.path[0] +"/GridSearchResult/" + f"/{dim}_14dGrid.txt", 'w')
    f.write(str(datetime.now()) + "\n\n")
    f.write(str(clf.cv_results_))
    f.write('\n')
    f.write(str(clf.best_estimator_))
    f.close()

    cv_log = pd.DataFrame.from_dict(clf.cv_results_)
    f = open(sys.path[0] +"/GridSearchResult/" +f"/{dim}_14dGrid.json", 'w')
    f.write(cv_log.to_json())
    f.close()

    dump(clf,sys.path[0] +"/GridSearchResult/"+f'{dim}_14Grid.joblib')


    ## Print result
    clf_load = load(sys.path[0] +"/GridSearchResult/"+f'{dim}_14Grid.joblib')

    y_trn_predict = clf_load.predict(X_trn)
    confusionMatrix.confusion_matrix(y_trn,y_trn_predict,save_fig_name=sys.path[0] +"/GridSearchResult/"+f'{dim}_conf_trn_14Grid.pdf')

    y_tst_predict = clf_load.predict(X_tst)
    confusionMatrix.confusion_matrix(y_tst,y_tst_predict,save_fig_name=sys.path[0] +"/GridSearchResult/"+f'{dim}_conf_tst_14Grid.pdf')
    print(f'dim:{dim} finished')
    plt.close()