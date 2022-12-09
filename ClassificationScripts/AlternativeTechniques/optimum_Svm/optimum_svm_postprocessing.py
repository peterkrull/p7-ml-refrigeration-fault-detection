import pandas as pd
import numpy as np
import sys
sys.path.append(sys.path[0] + "/../../../Python")
from gridSearch_scoreplot import plot_gridsearch_log
from joblib import dump, load
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV, PredefinedSplit,GroupKFold
import confusion_matrix2 as confusion_matrix
import standardization 
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
from matplotlib import rc


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

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


search_log_fixed = pd.read_json(sys.path[0] + "/svm_grid_search_log.json")

max_scores = search_log_fixed[search_log_fixed.split0_test_score == search_log_fixed.split0_test_score.max()].copy()
max_scores = max_scores.drop(['std_fit_time', 'std_test_score', 'std_score_time', 'split0_test_score'],axis =1)

print("C: ")
print(max_scores['param_C'])
print("\ngamma:")
print(max_scores['param_gamma'])
print("\nScore:")
print(max_scores['mean_test_score'])


plot_gridsearch_log(search_log_fixed, show_figure = True, save_figure = sys.path[0] + "/fixed_scoreplot.pdf", plot_max = True)

search_log_kfold = pd.read_json(sys.path[0] + "/svm_grid_search_log_kfold.json")

max_scores_fold = search_log_kfold[search_log_kfold.split0_test_score == search_log_kfold.split0_test_score.max()].copy()
pd.concat([max_scores ,max_scores_fold.drop(['std_fit_time', 'std_test_score', 'std_score_time', 'split0_test_score'],axis =1)])

print("C: ")
print(max_scores_fold['param_C'])
print("\ngamma:")
print(max_scores_fold['param_gamma'])
print("\nScore:")
print(max_scores_fold['mean_test_score'])

plot_gridsearch_log(search_log_kfold, show_figure = True, save_figure = sys.path[0] + "/kfold_scoreplot.pdf", plot_max = True)

train_data = pd.read_csv(sys.path[0] +"/../../../TrainingData/neodata/fault_all_nonoise_67.csv" )
std = standardization.standardization(train_data, target = 'target')
train_data = std.transform(train_data)
test_data = pd.read_csv(sys.path[0] + "/../../../TestData/neodata/fault_all_nonoise_67.csv")
test_data = std.transform(test_data)
val_data = pd.read_csv(sys.path[0] + "/../../../ValidationData/neodata/fault_all_nonoise_67.csv")
val_data = std.transform(val_data)

clf_fixed = load(sys.path[0] + "/GridSearchBest.joblib")
clf_kfold = load(sys.path[0] + "/GridSearchBest_kfold.joblib")

val_pred = clf_fixed.predict(val_data.drop('target', axis = 1).to_numpy())
tst_pred = clf_fixed.predict(test_data.drop('target', axis = 1).to_numpy())

confusion_matrix.confusion_matrix(test_data['target'],tst_pred, save_fig_name = sys.path[0] + "/optimum_svm_confmat_fixed.pdf", eval_labels = False, title = "SVM non kfold grid search on test set")
plt.show()
confusion_matrix.confusion_matrix(val_data['target'],val_pred, save_fig_name = sys.path[0] + "/optimum_svm_confmat_fixed_validation.pdf", eval_labels = False, title = "SVM non kfold grid search on validation set")
plt.show()

clf_fixed2 = svm.SVC(decision_function_shape='ovo', kernel= 'rbf', gamma= max_scores['param_gamma'], C = max_scores['param_C'])

val_pred = clf_kfold.predict(val_data.drop('target', axis = 1).to_numpy())
tst_pred = clf_kfold.predict(test_data.drop('target', axis = 1).to_numpy())

confusion_matrix.confusion_matrix(test_data['target'],tst_pred, save_fig_name = sys.path[0] + "/optimum_svm_confmat_kfold.pdf", eval_labels = False, title = "SVM grid search on test set")
plt.show()
confusion_matrix.confusion_matrix(val_data['target'],val_pred, save_fig_name = sys.path[0] + "/optimum_svm_confmat_kfold_validation.pdf", eval_labels = False, title = "SVM grid search on validation set")
plt.show()
