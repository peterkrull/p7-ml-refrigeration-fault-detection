from sklearn import svm
import pandas as pd
import sys
import standardization as std
from confusion_matrix2 import confusion_matrix as confmat
from sklearn.model_selection import GridSearchCV
import numpy as np

train_data = pd.read_csv(sys.path[0] + "/../TrainingData/neodata/fault_all_nonoise_67.csv")
test_data = pd.read_csv(sys.path[0] + '/../TestData/neodata/fault_all_nonoise_67.csv')
validation_data = pd.read_csv(sys.path[0] + "/../ValidationData/neodata/fault_all_nonoise_67.csv")

train_data = train_data
test_data = test_data
validation_data = validation_data


print("Training data")
print(train_data)
to_swap = np.c_[train_data['Tamb'].loc[train_data['target'] == 0].to_numpy(), train_data['Tset'].loc[train_data['target']==0].to_numpy()]
to_swap = pd.DataFrame(to_swap, columns = ['Tamb', 'Tset'])
fault_zero_rows = train_data[train_data['target'] == 0]
fault_zero_rows = fault_zero_rows.drop(['Tamb', 'Tset'], axis = 1)
fault_zero_rows.insert(9, 'Tamb', to_swap['Tset'])
fault_zero_rows.insert(10, 'Tset', to_swap['Tamb'])

train_data = train_data[train_data['target']!= 0]
train_data = pd.concat([fault_zero_rows, train_data]).reset_index().drop('index', axis  =1)


print(test_data)
to_swap = np.c_[test_data['Tamb'].loc[test_data['target'] == 0].to_numpy(), test_data['Tset'].loc[test_data['target']==0].to_numpy()]
to_swap = pd.DataFrame(to_swap, columns = ['Tamb', 'Tset'])
print(to_swap)
fault_zero_rows = test_data[test_data['target'] == 0]
print(fault_zero_rows)
fault_zero_rows = fault_zero_rows.drop(['Tamb', 'Tset'], axis = 1)
print(fault_zero_rows)
fault_zero_rows.insert(9, 'Tamb', to_swap['Tset'])
print(fault_zero_rows)
fault_zero_rows.insert(10, 'Tset', to_swap['Tamb'])

test_data = test_data[test_data['target']!= 0]
test_data = pd.concat([fault_zero_rows, test_data]).reset_index().drop('index', axis  =1)
print(test_data)




to_swap = np.c_[validation_data['Tamb'].loc[validation_data['target'] == 0].to_numpy(), validation_data['Tset'].loc[validation_data['target']==0].to_numpy()]
to_swap = pd.DataFrame(to_swap, columns = ['Tamb', 'Tset'])
fault_zero_rows = validation_data[validation_data['target'] == 0]
fault_zero_rows = fault_zero_rows.drop(['Tamb', 'Tset'], axis = 1)
fault_zero_rows.insert(9, 'Tamb', to_swap['Tset'])
fault_zero_rows.insert(10, 'Tset', to_swap['Tamb'])

validation_data = validation_data[validation_data['target']!= 0]
validation_data = pd.concat([fault_zero_rows, validation_data]).reset_index().drop('index', axis  =1)



standardizer = std.standardization(train_data, target = 'target')

trn = standardizer.transform(train_data)
tst = standardizer.transform(test_data)
val = standardizer.transform(validation_data)


C_params = [10**x for x in np.linspace(2,4,num = 30)]
gamma_params = [10**x for x in np.linspace(-3, -1, num = 30)]
print(gamma_params)

parameters = {'kernel':['rbf'], 'decision_function_shape':['ovo'], 'C' : C_params, 'gamma': gamma_params}
print(parameters)
svc = svm.SVC(cache_size= 500)
clf = GridSearchCV(svc, parameters, verbose = 3, n_jobs=8)
clf.fit(trn.drop('target', axis = 1), trn['target'])

print_str =' max accuracy: ' + str(clf.score(val.drop('target', axis = 1), val['target'])) + '\n'
print(clf.best_estimator_)

# SVM = svm.SVC(C = 1000, gamma = .01, kernel= 'rbf', decision_function_shape='ovo')
# SVM.fit(trn.drop('target', axis = 1), trn['target'])

pred_val = clf.predict(val.drop('target', axis = 1))

pred_tst = clf.predict(tst.drop('target', axis = 1))

f = open("svm_grid_search.txt", 'w')
f.write(print_str + 'best parameters:' + str(clf.best_params_))
f.close()

confmat(val['target'], pred_val, title = 'Validation data', save_fig_name='optimal_params_svm_wrong_col.pdf')

confmat(tst['target'], pred_tst, title = 'Test data', save_fig_name='optimal_params_svm_wrong_col_tst.pdf')