import numpy as np
import pandas as pd

import sys
sys.path.append(sys.path[0] + "/../../../../Python")

from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from datetime import datetime

'''
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "cmr"
})
'''

from joblib import dump, load


Print_figs=True
Grid_search=True


# Load and standard scaling
trn_data = pd.read_csv(sys.path[0] + "/../../../../TrainingData/neodata/soltani_14d_nonoise_1200.csv")
vld_data = pd.read_csv(sys.path[0] + "/../../../../ValidationData/neodata/soltani_14d_nonoise_1200.csv")
tst_data = pd.read_csv(sys.path[0] + "/../../../../TestData/neodata/soltani_14d_nonoise_100.csv")


feature_names = trn_data.drop(['target'],axis=1).columns.values
feature_drop =['Psuc','Density','CprPower','target']
#feature_drop =['Pdis','Psuc','T0','Tsh','CondFan','CprPower','Tamb','Tset','target','setpoint']
print(trn_data.drop(feature_drop,axis=1))


scale = StandardScaler()
X_trn = scale.fit_transform(trn_data.drop(feature_drop,axis=1))
X_vld = scale.transform(vld_data.drop(feature_drop,axis=1))
X_tst = scale.transform(tst_data.drop(feature_drop,axis=1))

y_trn=trn_data['target']
y_vld=vld_data['target']
y_tst=tst_data['target']

##Grid search
if(Grid_search==True):
    clf = svm.SVC(kernel='rbf',decision_function_shape='ovo',C=1000,gamma=0.01)
    clf.fit(X_trn,y_trn)



    dump(clf,sys.path[0]+'/SVM_NoGrid.joblib')

if(Print_figs==True):
    print("Saving figures")
    import confusion_matrix2 as confusionMatrix
    from gridSearch_scoreplot import plot_gridsearch_log
    clf_load = load(sys.path[0] +'/SVM_NoGrid.joblib')

    '''
    gridSearchLog = pd.read_json(sys.path[0] +"/SVM_GridSearchLog.json")
    plot_gridsearch_log(gridSearchLog,plot_max=True,save_figure='SVM-GridResult.pdf')
    print('Score plot saved')
    '''

    y_trn_predict = clf.predict(X_trn)
    confusionMatrix.confusion_matrix(y_trn,y_trn_predict,save_fig_name="SVM_trn_NoGrid.pdf",eval_labels = False,title='SVM training_NoGrid')
    print("Traning figure saved")

    y_vld_predict = clf.predict(X_vld)
    confusionMatrix.confusion_matrix(y_vld,y_vld_predict,save_fig_name="SVM_vld_NoGrid.pdf", eval_labels = False,title='SVM validation_NoGrid')
    print("Validation figure saved")

    y_tst_predict = clf.predict(X_tst)
    confusionMatrix.confusion_matrix(y_tst,y_tst_predict,save_fig_name="SVM_tst_NoGrid.pdf", eval_labels = False,title='SVM test_NoGrid')
    print("Test figure saved")

    #plt.close()
