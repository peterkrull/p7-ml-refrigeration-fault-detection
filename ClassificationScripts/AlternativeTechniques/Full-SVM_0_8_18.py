from sklearn.metrics import confusion_matrix as skconfmatrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(sys.path[0] + "/../../Python")

# Set correct working directory
# if os.getcwd() != os.path.abspath("../Python"):
#     os.chdir("../../Python")

# Import files from /Python directory
from confusion_matrix import confusion_matrix,false_info

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "cmr"
})

trn_data = pd.read_csv(sys.path[0] + f"/../../TrainingData/neodata/soltani_14d_nonoise_1200.csv")
vld_data = pd.read_csv(sys.path[0] + f"/../..//ValidationData/neodata/soltani_14d_nonoise_1200.csv")
tst_data = pd.read_csv(sys.path[0] + f"/../..//TestData/neodata/soltani_14d_nonoise_100.csv")



# Separate into data and targets
y_trn = trn_data.pop('target')
X_trn = trn_data

y_vld = vld_data.pop('target')
X_vld = vld_data

y_tst = tst_data.pop('target')
X_tst = tst_data

X_trn = X_trn[(y_trn == 0) | (y_trn == 8) | (y_trn == 18)]
y_trn = y_trn[(y_trn == 0) | (y_trn == 8) | (y_trn == 18)]

X_tst = X_tst[(y_tst == 0) | (y_tst == 8) | (y_tst == 18)]
y_tst = y_tst[(y_tst == 0) | (y_tst == 8) | (y_tst == 18)]

X_vld = X_vld[(y_vld == 0) | (y_vld == 8) | (y_vld == 18)]
y_vld = y_vld[(y_vld == 0) | (y_vld == 8) | (y_vld == 18)]

X_trn = X_trn.drop(columns=['Density','CprPower','Psuc'])
X_tst = X_tst.drop(columns=['Density','CprPower','Psuc'])
X_vld = X_vld.drop(columns=['Density','CprPower','Psuc'])
X_trn.head()


# Use standard scaler for scaling
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()

# Use PCA sphering/whitening for scaling
# from sklearn.decomposition import PCA
# standardizer = PCA(whiten=True)

X_trn = standardizer.fit_transform(X_trn)
X_vld = standardizer.transform(X_vld)
X_tst = standardizer.transform(X_tst)

# Setup axis ticks with classes removed
ticks = [[x for x in range(len(y_trn.unique()))],y_trn.unique()]

from sklearn import svm
from sklearn.model_selection import GridSearchCV, PredefinedSplit,GroupKFold

svc = svm.SVC()
C_params = [10**x for x  in np.linspace(2,4, 31)]
gamma_params = [10**x for x  in np.linspace(-2,0, 31)]
clf = GridSearchCV(svc, {'decision_function_shape':['ovo'], 'C' : C_params, 'gamma' : gamma_params}, n_jobs = -1, verbose = 3.5)

clf.fit(X_trn,y_trn)
print(clf.best_estimator_)