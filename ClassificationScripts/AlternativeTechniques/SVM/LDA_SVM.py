import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV, PredefinedSplit,GroupKFold
import sys
sys.path.append(sys.path[0] + "/../../../Python")
import confusion_matrix2 as confusion_matrix
import standardization 
import lda
from datetime import datetime
from joblib import dump, load


#Load data
trn_data = pd.read_csv(sys.path[0] + "/../../../TrainingData/neodata/14d_setpoints_1200.csv")
vld_data = pd.read_csv(sys.path[0] + "/../../../ValidationData/neodata/14d_setpoints_1200.csv")
tst_data = pd.read_csv(sys.path[0] + "/../../../TestData/neodata/14d_setpoints_100.csv")

#Reduce amount of data, for convenience and time
trn_data = trn_data
vld_data = vld_data.iloc[::10,:]
trn_set = trn_data['setpoint']

#Remove setpoint column
trn_data = trn_data.drop('setpoint', axis = 1)
vld_data = vld_data.drop('setpoint', axis = 1)
tst_data = tst_data.drop('setpoint', axis = 1)

#Remove features dropped in feature selection
dropped = ['Pdis', 'Psuc', 'T0', 'Tsh', 'CondFan', 'CprPower', 'Tamb', 'Tset']
trn_data = trn_data.drop(dropped, axis = 1)
vld_data = vld_data.drop(dropped, axis = 1)
test_data = tst_data.drop(dropped, axis = 1)

#Standardize data
std = standardization.standardization(trn_data, 'target')
trn_std = std.transform(trn_data)
vld_std = std.transform(vld_data)
tst_std = std.transform(test_data)

#Make LDA transformation
LDA = lda.LDA_reducer(trn_std, 3, 'target')
plt.savefig(sys.path[0] + "/LDA_sceeplot.pdf")
plt.show()
trn_red = LDA.transform(trn_std)
vld_red = LDA.transform(vld_std)
tst_red = LDA.transform(tst_std)

#Create SVM classifier
svc = svm.SVC(kernel= 'rbf', decision_function_shape= 'ovo')
#Define parameters for gridsearch
gamma_params = [10**x for x in np.linspace(-3, -1, 21)]
C_params = [10**x for x in np.linspace(2, 4, 21)]
gamma_params = [10**x for x in np.linspace(-3, -1, 3)]
C_params = [10**x for x in np.linspace(2, 4, 3)]
#Execute gridsearch
clf = GridSearchCV(svc, {'C' : C_params, 'gamma' : gamma_params}, n_jobs= -1, verbose=3.1)#, cv =GroupKFold(n_splits=5))
clf.fit(trn_red.drop('target', axis = 1), trn_red['target'])

print(clf.best_estimator_)

f = open(sys.path[0] + "/ldasvm_grid_search.txt", 'w')
f.write(str(datetime.now()) + "\n\n")
f.write(str(clf.cv_results_))
f.write("\n\n" + str(clf.best_estimator_))
f.close()

cv_log = pd.DataFrame.from_dict(clf.cv_results_)
f = open(sys.path[0] + "/ldasvm_grid_search_log.json", 'w')
f.write(cv_log.to_json())
f.close()

#Evaluate classifier using validation data
vld_pred = clf.predict(vld_red.drop('target', axis = 1))
confusion_matrix.confusion_matrix(vld_red['target'], vld_pred, save_fig_name = sys.path[0] + "/LDA_SVM_confmat_vld.pdf", title = "LDA-SVM, validation data")

#Evaluate classifier using testing data
tst_pred = clf.predict(tst_red.drop('target', axis = 1))
confusion_matrix.confusion_matrix(tst_red['target'], tst_pred, save_fig_name = sys.path[0] + "/LDA_SVM_confmat_tst.pdf", title = "LDA-SVM, test data")


#Make LDA transformation
LDA = lda.LDA_reducer(trn_std, 5, 'target')
trn_red = LDA.transform(trn_std)
vld_red = LDA.transform(vld_std)
tst_red = LDA.transform(tst_std)

#Create SVM classifier
svc = svm.SVC(kernel= 'rbf', decision_function_shape= 'ovo')
#Define parameters for gridsearch
gamma_params = [10**x for x in np.linspace(-3, -1, 21)]
C_params = [10**x for x in np.linspace(2, 4, 21)]
gamma_params = [10**x for x in np.linspace(-3, -1, 3)]
C_params = [10**x for x in np.linspace(2, 4, 3)]
#Execute gridsearch
clf = GridSearchCV(svc, {'C' : C_params, 'gamma' : gamma_params}, n_jobs= -1, verbose=3.1)#, cv =GroupKFold(n_splits=5))
clf.fit(trn_red.drop('target', axis = 1), trn_red['target'])

print(clf.best_estimator_)

f = open(sys.path[0] + "/lda5_svm_grid_search.txt", 'w')
f.write(str(datetime.now()) + "\n\n")
f.write(str(clf.cv_results_))
f.write("\n\n" + str(clf.best_estimator_))
f.close()

cv_log = pd.DataFrame.from_dict(clf.cv_results_)
f = open(sys.path[0] + "/lda5_svm_grid_search_log.json", 'w')
f.write(cv_log.to_json())
f.close()

#Evaluate classifier using validation data
vld_pred = clf.predict(vld_red.drop('target', axis = 1))
confusion_matrix.confusion_matrix(vld_red['target'], vld_pred, save_fig_name = sys.path[0] + "/LDA5_SVM_confmat_vld.pdf", title = "LDA5-SVM, validation data")

#Evaluate classifier using testing data
tst_pred = clf.predict(tst_red.drop('target', axis = 1))
confusion_matrix.confusion_matrix(tst_red['target'], tst_pred, save_fig_name = sys.path[0] + "/LDA5_SVM_confmat_tst.pdf", title = "LDA5-SVM, test data")
