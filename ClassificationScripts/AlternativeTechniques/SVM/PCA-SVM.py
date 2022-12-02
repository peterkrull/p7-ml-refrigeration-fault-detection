import numpy as np
import pandas as pd

import sys
sys.path.append(sys.path[0] + "/../../../Python")
import pca

from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "cmr"
})

from joblib import dump, load


Print_figs=False
Grid_search=True


# Load and standard scaling
trn_data = pd.read_csv(sys.path[0] + "/../../../TrainingData/neodata/14d_setpoints_1200.csv")
vld_data = pd.read_csv(sys.path[0] + "/../../../ValidationData/neodata/14d_setpoints_1200.csv")
tst_data = pd.read_csv(sys.path[0] + "/../../../TestData/neodata/14d_setpoints_100.csv")


feature_names = trn_data.drop(['target','setpoint'],axis=1).columns.values
feature_drop =['Pdis','Psuc','Tsup','Tsuc','Tsup','CondFan','CprPower','Density','Tamb','target','setpoint']

scale = StandardScaler()
X_trn = scale.fit_transform(trn_data.drop(feature_drop,axis=1))
X_vld = scale.transform(vld_data.drop(feature_drop,axis=1))
X_tst = scale.transform(tst_data.drop(feature_drop,axis=1))

y_trn=trn_data['target']
y_vld=vld_data['target']
y_tst=tst_data['target']


pcaRed =  pca.PCA_reducer(pd.DataFrame(X_trn),3, scree_plot=True)       #Preserve 91% of eigenvalues
X_trn_red = pcaRed.transform(pd.DataFrame(X_trn))
X_vld_red = pcaRed.transform(pd.DataFrame(X_vld))
X_tst_red = pcaRed.transform(pd.DataFrame(X_tst))

##Grid search
if(Grid_search==True):
    svc = svm.SVC(kernel='rbf',decision_function_shape='ovo')
    C_params = [10**x for x in np.linspace(1,5, 100)]           #Logrithmic svaling of parameters
    gamma_params = [10**x for x in np.linspace(-4,0, 100)]

    clf = RandomizedSearchCV(svc,{'C':C_params,'gamma':gamma_params},n_jobs=-1,verbose =3,n_iter=1)
    clf.fit(X_trn_red,y_trn)

    f = open(sys.path[0] +  "PCA-SVM_GridsearchResul.txt", 'w')
    f.write(str(datetime.now()) + "\n\n")
    f.write(str(clf.cv_results_))
    f.write('\n')
    f.write(str(clf.best_estimator_))
    f.close()

    cv_log = pd.DataFrame.from_dict(clf.cv_results_)
    f = open(sys.path[0] +"PCA-SVM_GridSearchLog.json", 'w')
    f.write(cv_log.to_json())
    f.close()

    dump(clf,'PCA-SVM.joblib')

if(Print_figs==True):
    print("Saving figures")
    import confusion_matrix2 as confusionMatrix
    clf_load = load(sys.path[0] +"PCA-SVM.joblib")

    y_trn_predict = clf_load.predict(X_trn_red)
    confusionMatrix.confusion_matrix(y_trn,y_trn_predict,save_fig_name="PCA-SVM_vld.pdf")
    print("Traning figure saved")

    y_vld_predict = clf_load.predict(X_vld_red)
    confusionMatrix.confusion_matrix(y_vld,y_vld_predict,save_fig_name="PCA-SVM_vld.pdf")
    print("Validation figure saved")

    y_tst_predict = clf_load.predict(X_tst_red)
    confusionMatrix.confusion_matrix(y_tst,y_tst_predict,save_fig_name="PCA-SVM_tst.pdf")
    print("Test figure saved")

    plt.close()
