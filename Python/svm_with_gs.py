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
val = standardizer.transform(validation_data)
tst = standardizer.transform(test_data)

#trn = train_data
#tst = test_data

targets = train_data['target'].unique().tolist()        # List of classes

labels_tst = test_data['target']


print_str : str = ''
C_max_acc = 1000
gamma_max_acc = 0.01
n_vals = 5
for i in range(0, 7):
    C_params = [x for x in np.linspace(C_max_acc/2,C_max_acc, num = n_vals)]  + [x for x in np.linspace(C_max_acc, 2*C_max_acc, num = n_vals)]
    C_params = list(dict.fromkeys(C_params))

    gamma_params = [x for x in np.linspace(gamma_max_acc/2,gamma_max_acc, num = n_vals)]  + [x for x in np.linspace(gamma_max_acc, 2*gamma_max_acc, num = n_vals)]
    gamma_params = list(dict.fromkeys(gamma_params))

    parameters = {'kernel':['rbf'], 'decision_function_shape':['ovo'], 'C' : C_params, 'gamma': gamma_params}
    print(parameters)
    svc = svm.SVC(cache_size= 500)
    clf = GridSearchCV(svc, parameters, verbose = 3, n_jobs=8, )
    clf.fit(trn.drop('target', axis = 1), trn['target'])
    print_str += str(i) + ' max accuracy: ' + str(clf.score(val.drop('target', axis = 1), val['target'])) + '\n'
    print(clf.best_estimator_)
    C_max_acc = clf.best_params_['C']
    gamma_max_acc = clf.best_params_['gamma']

f = open("svm_grid_search.txt", 'w')
f.write(print_str + 'best parameters:' + str(clf.best_params_))
f.close()