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
from lda import LDA_reducer
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


gs_log = pd.read_json(sys.path[0] + "/lda_svm11_log.json")
print(gs_log)

max_scores = gs_log[gs_log.mean_test_score == gs_log.mean_test_score.max()].copy()
max_scores = max_scores.drop(['std_fit_time', 'std_test_score', 'std_score_time', 'split0_test_score'],axis =1)

print("C: ")
print(max_scores['param_C'])
print("\ngamma:")
print(max_scores['param_gamma'])
print("\nScore:")
print(max_scores['mean_test_score'])

plot_gridsearch_log(gs_log, show_figure = True, save_figure = sys.path[0] + "/lda_scoreplot.pdf", plot_max = True)


train_data = pd.read_csv(sys.path[0] +"/../../../TrainingData/neodata/fault_all_nonoise_67.csv" )
std = standardization.standardization(train_data, target = 'target')
train_data = std.transform(train_data)
test_data = pd.read_csv(sys.path[0] + "/../../../TestData/neodata/fault_all_nonoise_67.csv")
test_data = std.transform(test_data)
val_data = pd.read_csv(sys.path[0] + "/../../../ValidationData/neodata/fault_all_nonoise_67.csv")
val_data = std.transform(val_data)

lda_reduc_11 = LDA_reducer(train_data,5, target_id = 'target', scree_plot = False)
trn_3 = lda_reduc_11.transform(train_data)
tst_r = lda_reduc_11.transform(test_data)
vld_r = lda_reduc_11.transform(val_data)


clf = load(sys.path[0] + "/model_ldasvm_11.joblib")

vld_pred = clf.predict(vld_r.drop('target', axis = 1).to_numpy())
tst_pred = clf.predict(tst_r.drop('target', axis = 1).to_numpy())

confusion_matrix.confusion_matrix(tst_r['target'],tst_pred, save_fig_name = sys.path[0] + "/lda_11_confmat.pdf", eval_labels = False, title = "LDA-SVM grid search on test set")
plt.show()

confusion_matrix.confusion_matrix(vld_r['target'],vld_pred, save_fig_name = sys.path[0] + "/lda_11_confmat_validation.pdf", eval_labels = False,title = "LDA-SVM grid search on validation set")
plt.show()