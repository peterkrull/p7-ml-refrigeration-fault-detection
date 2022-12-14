from sklearn.metrics import make_scorer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV, PredefinedSplit,GroupKFold
import sys
sys.path.append(sys.path[0] + "/../../Python")
import confusion_matrix2 as confusion_matrix
import standardization 
from datetime import datetime
from joblib import dump, load


def false_positives(true_target : pd.DataFrame, predicted_target : pd.DataFrame):
    zero_label_index = true_target.loc[true_target['target']==0]
    pred_non_fault = predicted_target[predicted_target.index.isin(zero_label_index.index)]
    false_positive = len(pred_non_fault.loc[pred_non_fault[0] != 0])/(len(pred_non_fault)+0.0001)
    return false_positive

def gridsearch_scoring(y_true : np.array, y_pred : np.array):
    sum = 0
    fp_ratio = .6
    fp = false_positives(pd.DataFrame(y_true, columns = ['target']), pd.DataFrame(y_pred))
    sum += (1-fp)*fp_ratio
    y_pred = pd.DataFrame(y_pred)
    accuracy = (len(y_pred) - len( y_pred[0].compare(pd.DataFrame(y_true)[0]) ) )  / (len(y_pred) + 0.0001)
    sum += (1-fp_ratio)*(accuracy)
    return sum


if __name__ == "__main__":
    train_data = pd.read_csv(sys.path[0] + "/../../TrainingData/neodata/fault_all_nonoise_67.csv")
    test_data = pd.read_csv(sys.path[0] + "/../../TestData/neodata/fault_all_nonoise_67.csv")


    # Split training data into traning and validation data.
    val_data = train_data.loc[(train_data['Tamb']==10) & (train_data['Tset']==7)]
    train_data1 = train_data.loc[(train_data['Tamb']!=10) | (train_data['Tset']!=7)]

    std = standardization.standardization(train_data1, target = 'target')
    train_data1 = std.transform(train_data1)
    val_data = std.transform(val_data)
    test_data = std.transform(test_data)

    val_data = val_data
    train_data1 = train_data1

    val_fold = [-1 for _ in range(len(train_data1)) ] + [0 for _ in range(len(val_data))]    

    optimize_data = pd.concat([train_data1, val_data])

    # svc = svm.SVC()
    # C_params = [10**x for x in np.linspace(0,5, 121)]
    # gamma_params = [10**x for x in np.linspace(-4,-1, 81)]
    # #C_params = [10**x for x in np.linspace(0,5, 3)]
    # #gamma_params = [10**x for x in np.linspace(-4,-1, 3)]
    # score = make_scorer(gridsearch_scoring, greater_is_better= True)
    # ps = PredefinedSplit(val_fold)
    # clf = GridSearchCV(svc, {'kernel':['rbf'], 'decision_function_shape':['ovo'], 'C' : C_params, 'gamma' : gamma_params}, n_jobs = -1, verbose = 3, scoring = score, cv = ps)
    # #clf = GridSearchCV(svc, {'kernel':['rbf'], 'decision_function_shape':['ovo'], 'C' : C_params, 'gamma' : gamma_params}, n_jobs = -1, verbose = 3)
    # clf.fit(optimize_data.drop('target', axis = 1).to_numpy(), optimize_data['target'].to_numpy())

    # tst_pred = clf.predict(test_data.drop('target', axis = 1).to_numpy())

    # confusion_matrix.confusion_matrix(test_data['target'],tst_pred, save_fig_name = sys.path[0] + "/optimum_Svm/optimum_svm_tst.pdf")

    # print("Best estimator = " + str(clf.best_estimator_))

    # f = open(sys.path[0] + "/optimum_Svm/svm_grid_search.txt", 'w')
    # f.write(str(datetime.now()) + "\n\n")
    # f.write(str(clf.cv_results_))
    # f.write("\n\n" + str(clf.best_estimator_))
    # f.close()

    # cv_log = pd.DataFrame.from_dict(clf.cv_results_)
    # f = open(sys.path[0] + "/optimum_Svm/svm_grid_search_log.json", 'w')
    # f.write(cv_log.to_json())
    # f.close()
    # dump(clf,sys.path[0] + '/optimum_Svm/GridSearchBest2.joblib')


    ##KFOLD-CROSSVALIDATION##

    train_data2 = pd.read_csv(sys.path[0] + "/../../TrainingData/neodata/11d_setpoints_1200.csv")

    train_data2

    g_trn = train_data2['setpoint'].to_numpy()
    train_data2 = train_data2.drop('setpoint', axis = 1)

    std = standardization.standardization(train_data2, target = 'target')
    train_data2 = std.transform(train_data2)
    test_data = std.transform(test_data)

    
    svc = svm.SVC()
    C_params = [10**x for x in np.linspace(4,7, 31)]
    gamma_params = [10**x for x in np.linspace(-4,-1, 81)]
    #C_params = [10**x for x in np.linspace(0,5, 3)]
    #gamma_params = [10**x for x in np.linspace(-4,-1, 3)]
    score = make_scorer(gridsearch_scoring, greater_is_better= True)
    ps = PredefinedSplit(val_fold)
    clf = GridSearchCV(svc, {'kernel':['rbf'], 'decision_function_shape':['ovo'], 'C' : C_params, 'gamma' : gamma_params}, n_jobs = -1, verbose = 3, scoring = score, cv =GroupKFold(n_splits=5))
    clf.fit(train_data2.drop('target', axis = 1).to_numpy(), train_data2['target'].to_numpy(), groups = g_trn)

    tst_pred = clf.predict(test_data.drop('target', axis = 1).to_numpy())

    confusion_matrix.confusion_matrix(test_data['target'],tst_pred, save_fig_name = sys.path[0] + "/optimum_Svm/optimum_svm_tst_kfold2.pdf")

    print("Best estimator = " + str(clf.best_estimator_))

    f = open(sys.path[0] + "/optimum_Svm/svm_grid_search_kfold2.txt", 'w')
    f.write(str(datetime.now()) + "\n\n")
    f.write(str(clf.cv_results_))
    f.write("\n\n" + str(clf.best_estimator_))
    f.close()

    cv_log = pd.DataFrame.from_dict(clf.cv_results_)
    f = open(sys.path[0] + "/optimum_Svm/svm_grid_search_log_kfold2.json", 'w')
    f.write(cv_log.to_json())
    f.close()

    dump(clf,sys.path[0] + '/optimum_Svm/GridSearchBest_kfold2.joblib')