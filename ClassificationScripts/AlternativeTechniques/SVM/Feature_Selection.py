import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.feature_selection import SequentialFeatureSelector
import sys
sys.path.append(sys.path[0] + "/../../../Python")
import confusion_matrix2 as confusion_matrix
import standardization 
from datetime import datetime
from joblib import dump, load

train_data = pd.read_csv(sys.path[0] + "/../../../TrainingData/neodata/14d_setpoints_1200.csv")
val_data = pd.read_csv(sys.path[0] + "/../../../ValidationData/neodata/14d_setpoints_1200.csv")

trn_set = train_data['setpoint']

train_data = train_data.drop('setpoint', axis = 1)
val_data = val_data.drop('setpoint', axis = 1)
train_data = train_data.iloc[::10,:]
val_data = val_data.iloc[::10,:]

std = standardization.standardization(train_data, 'target')
trn_std = std.transform(train_data)
val_std = std.transform(val_data)

svc = svm.SVC(kernel= 'rbf', gamma = 0.01, C = 1000, decision_function_shape= 'ovo')

f = open(sys.path[0] + "/feature_extraction_results.txt", 'w')
f.write(str(datetime.now()) + "\n\n")
f.close()

sets = []
scores = []

for i in range(1,len(trn_std.drop('target', axis = 1).columns)-10):
    to_print = ''
    print(i)
    bw_selector = SequentialFeatureSelector(svc, direction = 'backward', n_jobs = -1, n_features_to_select = i)
    bw_selector.fit(trn_std.drop('target', axis = 1), trn_std['target'])

    labels = trn_std.drop('target',axis=1).columns
    dropped = labels[np.invert(bw_selector.get_support())]
    sets.append([x for x in labels[bw_selector.get_support()]])
    print(labels[bw_selector.get_support()])
    to_print += "Features kept: " + str(labels[bw_selector.get_support()]) + "\n"

    trn_red = trn_std.drop(dropped, axis = 1)
    val_red = val_std.drop(dropped, axis = 1)

    svc_int = svm.SVC(kernel= 'rbf', gamma = 0.01, C = 1000, decision_function_shape= 'ovo')
    svc_int.fit(trn_red.drop('target', axis = 1), trn_red['target'])
    score = svc_int.score(val_red.drop('target', axis = 1), val_red['target'])
    print(score)
    scores.append(score)
    to_print += "Score: " + str(score) + "\n\n"

    f = open(sys.path[0] + "/feature_extraction_results.txt", 'a')
    f.write(to_print)
    f.close()

print(sets)


plot_ticks = []
for i in sets:
    tick = ""
    for j in i:
        tick += str(j) + "\n"
    plot_ticks.append(tick)

print(plot_ticks)

plt.bar( [x for x in range(len(scores))],scores)
plt.ylabel("Accuracy")
plt.show()

