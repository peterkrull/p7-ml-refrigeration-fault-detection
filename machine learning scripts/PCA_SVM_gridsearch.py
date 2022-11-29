import sys
sys.path.append(sys.path[0] + "/../Python")
import pca
import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV, PredefinedSplit,GroupKFold
import standardization 
import plot_functions as pf
from sklearn.metrics import make_scorer
from datetime import datetime
from joblib import dump, load


def false_positives(true_target : pd.DataFrame, predicted_target : pd.DataFrame):
    zero_label_index = true_target.loc[true_target['target']==0]
    pred_non_fault = predicted_target[predicted_target.index.isin(zero_label_index.index)]
    false_positive = len(pred_non_fault.loc[pred_non_fault[0] != 0])/(len(pred_non_fault))
    return false_positive

def gridsearch_scoring(y_true : np.array, y_pred : np.array):
    sum = 0
    fp_ratio = .6
    fp = false_positives(pd.DataFrame(y_true, columns = ['target']), pd.DataFrame(y_pred))
    sum += fp_ratio*(1-fp)
    # if fp != 0:
    #     sum += fp_ratio*1/fp
    # else:
    #     sum += 1*fp_ratio

    y_pred = pd.DataFrame(y_pred)
    accuracy = (len(y_pred) - len( y_pred[0].compare(pd.DataFrame(y_true)[0]) ) )  / len(y_pred)

    sum += (1-fp_ratio)*accuracy
    return sum


def pca_svm_gridsearch(train_data: pd.DataFrame, test_data: pd.DataFrame, pca_n, dim):
    g_trn = train_data['setpoint'].to_numpy()
    train_data = train_data.drop('setpoint', axis = 1)
    test_data = test_data.drop('setpoint', axis = 1)
    pca_class = pca.PCA_reducer(train_data, pca_n,'target', scree_plot= True)
    trn = pca_class.transform(train_data)
    tst = pca_class.transform(test_data)


    C_params = [10**x for x in np.linspace(4,7, 31)]
    gamma_params = [10**x for x in np.linspace(-4,-1, 81)]
    C_params = [10**x for x in np.linspace(0,5, 1)]
    gamma_params = [10**x for x in np.linspace(-4,-1, 1)]
    score = make_scorer(gridsearch_scoring, greater_is_better= True)
    svc = svm.SVC()
    clf = GridSearchCV(svc, {'kernel':['rbf'], 'decision_function_shape':['ovo'], 'C' : C_params, 'gamma' : gamma_params}, n_jobs = -1, verbose = 3, scoring = score, cv =GroupKFold(n_splits=5))
    clf.fit(trn.drop('target', axis = 1).to_numpy(), trn['target'].to_numpy(), groups = g_trn)

    tst_pred = clf.predict(tst.drop('target', axis = 1).to_numpy())

    fp = false_positives(tst, pd.DataFrame(tst_pred))
    acc = (len(tst_pred) - len( pd.DataFrame(tst_pred)[0].compare(pd.DataFrame(tst_pred)[0]) ) )  / len(tst_pred)

    path = "/pca_svm_logs/dim" + str(dim) + "pca" + str(pca_n) + ".txt"
    f = open(sys.path[0] + path, 'w')
    f.write(str(datetime.now()) + "\n")
    f.write(str(clf.cv_results_))
    f.write("\nAccuracy = " + str(acc))
    f.write("\n False positive = " + str(fp))
    f.write("\nEigenvalue retention = " + str(pca_class.preserved_eigval))
    f.write("\n\n" + str(clf.best_estimator_))
    f.close()

    cv_log = pd.DataFrame.from_dict(clf.cv_results_)
    f = open(sys.path[0] + "/optimum_Svm/pca_svm_logs/svm_grid_search_log_kfold2_dim" + str(dim) + "_pca_" + str(pca_n) + ".json", 'w')
    f.write(cv_log.to_json())
    f.close()

    dump(clf, sys.path[0] + "/optimum_Svm/pca_svm_logs/svm_grid_search_log_kfold2_dim" + str(dim) + "_pca_" + str(pca_n) +".joblib")


if __name__ == "__main__":
    train_11 = pd.read_csv(sys.path[0] + "/../TrainingData/neodata/11d_setpoints_1200.csv")
    test_11 = pd.read_csv(sys.path[0] + "/../TestData/neodata/11d_setpoints_100.csv")

    train_14 = pd.read_csv(sys.path[0] + "/../TrainingData/neodata/14d_setpoints_1200.csv")
    test_14 = pd.read_csv(sys.path[0] + "/../TestData/neodata/14d_setpoints_100.csv")

    std_11 = standardization.standardization(train_11.drop('setpoint',axis = 1), target = 'target')
    trn_11 = std_11.transform(train_11.drop('setpoint', axis = 1))
    tst_11 = std_11.transform(test_11.drop('setpoint', axis = 1))
    trn_11.insert(len(trn_11.columns), 'setpoint', train_11['setpoint'])
    tst_11.insert(len(tst_11.columns), 'setpoint', test_11['setpoint'])
    print(trn_11)

    std_14 = standardization.standardization(train_14.drop('setpoint',axis = 1), target = 'target')
    trn_14 = std_14.transform(train_11.drop('setpoint', axis = 1))
    tst_14 = std_14.transform(test_11.drop('setpoint', axis = 1))
    trn_14.insert(len(trn_14.columns), 'setpoint', train_14['setpoint'])
    tst_14.insert(len(tst_14.columns), 'setpoint', test_14['setpoint'])    

    trn_11 = trn_11.iloc[::5,:]
    tst_11 = tst_11.iloc[::5,:]

    trn_14 = trn_14.iloc[::5,:]
    tst_14 = tst_14.iloc[::5,:]

    for i in range(2,4):# range(2, np.max([len(trn_14.columns), len(trn_11.columns)])):
        print("\n\nDimension" + str(i))
        
        if(i<len(trn_11.columns)):
            print("11 dimensions")
            pca_svm_gridsearch(trn_11, tst_11, i, 11)

        if(i<len(trn_14.columns)):
            print("\n14 dimensions")
            pca_svm_gridsearch(trn_14, tst_14, i, 14)
