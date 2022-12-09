import pandas as pd
import numpy as np
import sys
sys.path.append(sys.path[0] + "/../../../../Python")
from gridSearch_scoreplot import plot_gridsearch_log, plot_gridsearch_gradient_log
from joblib import dump, load
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV, PredefinedSplit,GroupKFold
import confusion_matrix2 as confusion_matrix
import standardization 
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
from matplotlib import rc
import lda




def do_stuff(dim_n, train, vali, test):
    search_log_fixed = pd.read_json(sys.path[0] + f"/lda{dim_n}_svm_grid_search_log.json")

    max_scores = search_log_fixed[search_log_fixed.mean_test_score == search_log_fixed.mean_test_score.max()].copy()
    max_scores = max_scores.drop(['std_fit_time', 'std_test_score', 'std_score_time', 'split0_test_score'],axis =1)


    print("C: ")
    print(max_scores['param_C'])
    print("\ngamma:")
    print(max_scores['param_gamma'])
    print("\nScore:")
    print(max_scores['mean_test_score'])



    orig = search_log_fixed.loc[(search_log_fixed['param_gamma'] == 0.01) & (search_log_fixed['param_C'] == 1000)]
    print(orig)


    plot_gridsearch_log(search_log_fixed, show_figure = True, save_figure = sys.path[0] + f"/lda{dim_n}_svm_scoreplot.pdf", plot_max = True, fig_size = (5,3))


    LDA = lda.LDA_reducer(train, dim_n, 'target')
    trn_red = LDA.transform(train)
    vld_red = LDA.transform(vali)
    tst_red = LDA.transform(test)

    # svc = svm.SVC(decision_function_shape= 'ovo', kernel = 'rbf', gamma = max_scores.param_gamma.item(), C = max_scores.param_C.item())
    # svc.fit(trn_red.drop('target', axis = 1), trn_red['target'])

    # val_pred = svc.predict(vld_red.drop('target', axis = 1).to_numpy())
    # tst_pred = svc.predict(tst_red.drop('target', axis = 1).to_numpy())

    # confusion_matrix.confusion_matrix(test['target'],tst_pred, save_fig_name = sys.path[0] + f"/LDA{dim_n}_SVM_confmat_tst.pdf", eval_labels = False, title = f"LDA{dim_n}-SVM, test data")
    # plt.show()
    # confusion_matrix.confusion_matrix(vali['target'],val_pred, save_fig_name = sys.path[0] + f"/LDA{dim_n}_SVM_confmat_vld.pdf", eval_labels = False, title = f"LDA{dim_n}-SVM, validation data")
    # plt.show()


    svc2 = svm.SVC(decision_function_shape= 'ovo', kernel = 'rbf', gamma = .01, C = 1000)
    svc2.fit(trn_red.drop('target', axis = 1), trn_red['target'])

    val_pred2 = svc2.predict(vld_red.drop('target', axis = 1).to_numpy())
    tst_pred2 = svc2.predict(tst_red.drop('target', axis = 1).to_numpy())

    confusion_matrix.confusion_matrix(test['target'],tst_pred2, save_fig_name = sys.path[0] + f"/LDA{dim_n}_SVM2_confmat_tst.pdf", eval_labels = False, title = f"LDA{dim_n}-SVM, test data")
    plt.show()
    confusion_matrix.confusion_matrix(vali['target'],val_pred2, save_fig_name = sys.path[0] + f"/LDA{dim_n}_SVM2_confmat_vld.pdf", eval_labels = False, title = f"LDA{dim_n}-SVM, validation data")
    plt.show()

if __name__ == "__main__":
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')



    dropped = ['Pdis', 'Psuc', 'T0', 'Tsh', 'CondFan', 'CprPower', 'Tamb', 'Tset']
    trn_data = pd.read_csv(sys.path[0] +"/../../../../TrainingData/neodata/soltani_14d_nonoise_1200.csv" )
    trn_data = trn_data.drop(dropped, axis = 1)
    std = standardization.standardization(trn_data, target = 'target')
    trn_data = std.transform(trn_data)

    tst_data = pd.read_csv(sys.path[0] + "/../../../../TestData/neodata/soltani_14d_nonoise_100.csv")
    tst_data = tst_data.drop(dropped, axis = 1)
    tst_data = std.transform(tst_data)

    vld_data = pd.read_csv(sys.path[0] + "/../../../../ValidationData/neodata/soltani_14d_nonoise_1200.csv")
    vld_data = vld_data.drop(dropped, axis = 1)
    vld_data = std.transform(vld_data)

    do_stuff(5, trn_data, vld_data, tst_data)
    