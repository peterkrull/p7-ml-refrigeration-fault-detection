import numpy as np
from scipy import linalg
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import sys

import matplotlib.pyplot as plt

import pandas as pd

import standardization as sd                                #Self written, Scikit-learn do have standardScaler which does the same.
import confusion_matrix

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "cmr"
})

# Load training data
train_data = pd.read_csv(sys.path[0] + "/../TrainingData/neodata/fault_all_noise_67.csv")
test_data = pd.read_csv(sys.path[0] + '/../TestData/neodata/fault_all_noise_67.csv')
validation_data = pd.read_csv(sys.path[0] + "/../ValidationData/neodata/fault_all_noise_67.csv")

#Normalization / conditioning
standardizer = sd.standardization(train_data,'target')

trn = standardizer.transform(train_data)
tst = standardizer.transform(test_data)

#trn = train_data
#tst = test_data

targets = train_data['target'].unique().tolist()        # List of classes

labels_tst = test_data['target']


C_max_acc = 10000
gamma_max_acc = 0.001
C_params = [x for x in np.linspace(C_max_acc/2,C_max_acc, num = 5)]  + [x for x in np.linspace(C_max_acc, 2*C_max_acc, num = 5)]
gamma_params = [10**x for x in range(-6,0)]
parameters = {'kernel':['rbf'], 'decision_function_shape':['ovo'], 'C' : C_params, 'gamma': gamma_params}
print(parameters)
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, verbose = 2, n_jobs=8)
clf.fit(trn.drop('target', axis = 1).to_numpy(), trn['target'].to_numpy())
print(clf.best_estimator_)

f = open("svm_grid_search.txt", 'w')
f.write(str(clf.best_estimator_))
f.close()