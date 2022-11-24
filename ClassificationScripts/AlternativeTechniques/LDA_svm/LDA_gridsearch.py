import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, PredefinedSplit,GroupKFold
from sklearn.metrics import make_scorer
import sys
sys.path.append(sys.path[0] + "/../../../Python")
import confusion_matrix2 as confusion_matrix
import standardization 
from lda import LDA_reducer
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


if __name__ == '__main__':
    train_11 = pd.read_csv(sys.path[0] + '/../../../TrainingData/neodata/fault_all_nonoise_67.csv')
    test_11 = pd.read_csv(sys.path[0] + '/../../../TestData/neodata/fault_all_nonoise_67.csv')
    train_14 = pd.read_csv(sys.path[0] + '/../../../TrainingData/neodata/14d_setpoints_1200.csv')
    test_14 = pd.read_csv(sys.path[0] + '/../../../TestData/neodata/14d_setpoints_100.csv')
    train_14 = train_14.drop('setpoint', axis = 1)
    test_14 = test_14.drop('setpoint', axis = 1)

    train_11 = train_11
    train_14 = train_14
    test_14 = test_14
    test_11 = test_11


    std11 = standardization.standardization(train_11, target = 'target')
    trn_11 = std11.transform(train_11)
    tst_11 = std11.transform(test_11)    


    print(train_14.columns)
    std14 = standardization.standardization(train_14, target = 'target')
    trn_14 = std14.transform(train_14)
    tst_14 = std14.transform(test_14)    



    lda_reduc_11 = LDA_reducer(trn_11,5, target_id = 'target', scree_plot = False)
    lda_reduc_14 = LDA_reducer(trn_14, 5, target_id = 'target', scree_plot = False)

    trn_11_r = lda_reduc_11.transform(trn_11)
    tst_11_r = lda_reduc_11.transform(tst_11)
    trn_14_r = lda_reduc_14.transform(trn_14)
    tst_14_r = lda_reduc_14.transform(tst_14)
    
    
    #C_params = [10**x for x in np.linspace(0,5, 121)]
    #gamma_params = [10**x for x in np.linspace(-4,-1, 81)]
    C_params = [10**x for x in np.linspace(0,5, 3)]
    gamma_params = [10**x for x in np.linspace(-4,-1, 3)]
    svc_11 = svm.SVC()
    score = make_scorer(gridsearch_scoring, greater_is_better= True)
    clf_11 = GridSearchCV(svc_11, {'kernel':['rbf'], 'decision_function_shape':['ovo'], 'C' : C_params, 'gamma' : gamma_params}, n_jobs = -1, verbose = 3, scoring = score)
    clf_11.fit(trn_11_r.drop('target', axis = 1).to_numpy(), trn_11_r['target'].to_numpy())
    tst_pred_11 = clf_11.predict(tst_11_r.drop('target', axis = 1).to_numpy())

    svc_14 = svm.SVC()
    clf_14 = GridSearchCV(svc_14, {'kernel':['rbf'], 'decision_function_shape':['ovo'], 'C' : C_params, 'gamma' : gamma_params}, n_jobs = -1, verbose = 3, scoring = score)
    clf_14.fit(trn_14_r.drop('target', axis = 1).to_numpy(), trn_14_r['target'].to_numpy())
    tst_pred_14 = clf_14.predict(tst_14_r.drop('target', axis = 1).to_numpy())

    
    confusion_matrix.confusion_matrix(tst_11_r['target'],tst_pred_11, save_fig_name = sys.path[0] + "/optimum_ldasvm11_tst_confmat.pdf")

    print("Best estimator = " + str(clf_11.best_estimator_))

    f = open(sys.path[0] + "/ldasvm11.txt", 'w')
    f.write(str(datetime.now()) + "\n\n")
    f.write(str(clf_11.cv_results_))
    f.write("\n\n" + str(clf_11.best_estimator_))
    f.close()

    cv_log = pd.DataFrame.from_dict(clf_11.cv_results_)
    f = open(sys.path[0] + "/lda_svm11_log.json", 'w')
    f.write(cv_log.to_json())
    f.close()
    dump(clf_11,sys.path[0] + '/model_ldasvm_11.joblib')




    confusion_matrix.confusion_matrix(tst_14_r['target'],tst_pred_14, save_fig_name = sys.path[0] + "/optimum_ldasvm14_tst_confmat.pdf")

    print("Best estimator = " + str(clf_14.best_estimator_))

    f = open(sys.path[0] + "/ldasvm14.txt", 'w')
    f.write(str(datetime.now()) + "\n\n")
    f.write(str(clf_14.cv_results_))
    f.write("\n\n" + str(clf_14.best_estimator_))
    f.close()

    cv_log = pd.DataFrame.from_dict(clf_14.cv_results_)
    f = open(sys.path[0] + "/lda_svm14_log.json", 'w')
    f.write(cv_log.to_json())
    f.close()
    dump(clf_14,sys.path[0] + '/model_ldasvm_14.joblib')

