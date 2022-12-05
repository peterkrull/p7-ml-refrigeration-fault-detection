import numpy as np
import pandas as pd

import sys
sys.path.append(sys.path[0] + "/../../../../Python")

from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import GridSearchCV,GroupKFold
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
trn_data = pd.read_csv(sys.path[0] + "/../../../../TrainingData/neodata/14d_setpoints_1200.csv")
vld_data = pd.read_csv(sys.path[0] + "/../../../../ValidationData/neodata/14d_setpoints_1200.csv")
tst_data = pd.read_csv(sys.path[0] + "/../../../../TestData/neodata/14d_setpoints_100.csv")


feature_names = trn_data.drop(['target'],axis=1).columns.values
feature_drop =['Pdis','Psuc','T0','Tsh','CondFan','CprPower','Tamb','Tset','target','setpoint']
print(trn_data.drop(feature_drop,axis=1))


g_trn = trn_data['setpoint']
g_vld = vld_data['setpoint']
g_tst = tst_data['setpoint']

scale = StandardScaler()
X_trn = scale.fit_transform(trn_data.drop(feature_drop,axis=1))
X_vld = scale.transform(vld_data.drop(feature_drop,axis=1))
X_tst = scale.transform(tst_data.drop(feature_drop,axis=1))

y_trn=trn_data['target']
y_vld=vld_data['target']
y_tst=tst_data['target']

##Grid search
if(Grid_search==True):
    svc = svm.SVC(kernel='rbf',decision_function_shape='ovo')
    C_params = [10**x for x in np.linspace(2,5,42)]           #Logrithmic svaling of parameters
    gamma_params = [10**x for x in np.linspace(-3,-1,21)]

    clf = GridSearchCV(svc,{'C':C_params,'gamma':gamma_params},n_jobs=-1,verbose =3,cv=GroupKFold(n_splits=5))
    clf.fit(X_trn,y_trn,groups=g_trn)

    f = open(sys.path[0] +  "/SVM_GridsearchResult.txt", 'w')
    f.write(str(datetime.now()) + "\n\n")
    f.write(str(clf.cv_results_))
    f.write('\n')
    f.write(str(clf.best_estimator_))
    f.close()

    cv_log = pd.DataFrame.from_dict(clf.cv_results_)
    f = open(sys.path[0] +"/SVM_GridSearchLog.json", 'w')
    f.write(cv_log.to_json())
    f.close()

    dump(clf,sys.path[0]+'/SVM.joblib')

if(Print_figs==True):
    print("Saving figures")
    import confusion_matrix2 as confusionMatrix
    from gridSearch_scoreplot import plot_gridsearch_log
    clf_load = load(sys.path[0] +'/SVM.joblib')

    gridSearchLog = pd.read_json(sys.path[0] +"/SVM_GridSearchLog.json")
    plot_gridsearch_log(gridSearchLog,plot_max=True,save_figure='SVM-GridResult.pdf')
    print('Score plot saved')

    y_trn_predict = clf_load.predict(X_trn)
    confusionMatrix.confusion_matrix(y_trn,y_trn_predict,save_fig_name="SVM_trn.pdf",eval_labels = False,title='SVM training')
    print("Traning figure saved")

    y_vld_predict = clf_load.predict(X_vld)
    confusionMatrix.confusion_matrix(y_vld,y_vld_predict,save_fig_name="SVM_vld.pdf", eval_labels = False,title='SVM validation')
    print("Validation figure saved")

    y_tst_predict = clf_load.predict(X_tst)
    confusionMatrix.confusion_matrix(y_tst,y_tst_predict,save_fig_name="SVM_tst.pdf", eval_labels = False,title='SVM test')
    print("Test figure saved")

    #plt.close()
