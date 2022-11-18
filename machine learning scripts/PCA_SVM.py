import sys
sys.path.append(sys.path[0] + "/../Python")
import pca
import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
import json as js
import standardization 
import plot_functions as pf
import lda 
from sklearn.metrics import make_scorer

def get_valData(train_data: pd.DataFrame):
    validation_data = pd.DataFrame()
    validation_data = train_data.sample(int(train_data.shape[0]/20  ))
    train_data = train_data.drop(validation_data.index)
    validation_data = validation_data.reset_index()
    validation_data = validation_data.drop('index', axis = 1)
    validation_data = validation_data.sort_values(by = ['target'])

    return train_data, validation_data

def false_positives(true_target : pd.DataFrame, predicted_target : pd.DataFrame):
    zero_label_index = true_target.loc[true_target['target']==0]
    pred_non_fault = predicted_target[predicted_target.index.isin(zero_label_index.index)]
    false_positive = len(pred_non_fault.loc[pred_non_fault[0] != 0])/(len(pred_non_fault))
    return false_positive

def gridsearch_scoring(y_true : np.array, y_pred : np.array):
    sum = 0
    fp_ratio = .2
    fp = false_positives(pd.DataFrame(y_true, columns = ['target']), pd.DataFrame(y_pred))
    if fp != 0:
        sum += fp_ratio*1/fp
    else:
        sum += 1*fp_ratio

    y_pred = pd.DataFrame(y_pred)
    sum += (1-fp_ratio)*(len(y_pred) - len( y_pred[0].compare(pd.DataFrame(y_true)[0]) ) )  / len(y_pred)
    return sum


def PCA_SVM(train_data: pd.DataFrame, val_data: pd.DataFrame, classes: pd.DataFrame, conf_title: str = None, plt_title :str = 'pca_reduc', plt_show : bool = False, gamma = .01, c = 1000, test_data :pd.DataFrame= None, pca_n = 2):
    """
    train_data : Training data, pandas dataframe 
    val_data : Validation data, pandas dataframe
    classes : Class labels, pandas dataframe
    conf_title : Confusion matrix title, string
    plt_title : PCA reduction plot title, string
    plt_show : Show plots, Bool
    gamma : Gamma variable for SVM, standard = 0.01
    c : C variable for SVM, standard 1000
    test_data : Test data, pandas dataframe, if not defined won't predict on set
    pca_n : Dimension reduction for PCA, standard = 2
    """

    toReturn = "PCA_SVM, pca_n = " + str(pca_n) + ", dim dataset = " + str(len(train_data.columns)-1)

    #Initiate PCA-algo to find dim reduction matrix w (Stored in class)
    print("PCA-reduction to n = " + str(pca_n) + " dimensions")
    pca_class = pca.PCA_reducer(train_data, pca_n,'target', scree_plot= True)
    if pca_n == 2:
        plt.savefig(sys.path[0]+"/pca_svm_screeplot_"+ plt_title+ ".pdf", bbox_inches='tight')
        plt.clf()

    toReturn += "\n\tPreserved eigenvalues = " + str(pca_class.preserved_eigval)

    #Dim reduce training data & val data
    print("Transforming data")
    trans_data = pca_class.transform(train_data, 'target')
    val_red_data = pca_class.transform(val_data, 'target')


    if pca_n == 2:
        print("Plotting data")
        pf.plot_transformation(trans_data.iloc[::10, :], file_name =  sys.path[0] + "/"+ plt_title + ".pdf", ec_filepath = sys.path[0] + '/../Python/error_color_coding.json')
        pf.plot_transformation(val_red_data.iloc[::10, :], file_name= sys.path[0] + "/"+ plt_title + "_val_data.pdf", ec_filepath = sys.path[0] +'/../Python/error_color_coding.json')


    print("Fitting data")
    #Fit svm model to dim red data
    svc = svm.SVC(kernel = 'rbf', decision_function_shape='ovo')
    C_params = [10**x for x in np.linspace(-3,-1, 51)]
    gamma_params = [10**x for x in np.linspace(2,4, 51)]
    score = make_scorer(gridsearch_scoring, greater_is_better= True)
    clf = GridSearchCV(svc, {'C' : C_params, 'gamma' : gamma_params}, n_jobs = -1, verbose = 3, scoring = score)
    clf.fit(trans_data.drop('target', axis = 1).to_numpy(), trans_data['target'].to_numpy())

    toReturn += "\n\tEstimator = " +  str(clf.best_estimator_)


    print("Classifying validation data")
    #Use validation data

    pred_val = pd.DataFrame(clf.predict(val_red_data.drop('target', axis = 1).to_numpy()))

    val_accuracy = (len(pred_val) - len( pred_val[0].compare(val_red_data['target']) ) )  / len(pred_val)
    val_falsePositives = false_positives(val_red_data, pred_val)
    print( "Accuracy = " + str(val_accuracy))
    print("False positives = " + str(val_falsePositives))

    toReturn += "\n\tValidation Accuracy = " + str(val_accuracy)
    toReturn += "\n\tValidation false positives = " + str(val_falsePositives)

    #Create confusion matrix
    if conf_title:
        
        conf_matrix = np.zeros([classes.shape[0], classes.shape[0]])

        for x,y in zip(pred_val.to_numpy(), val_red_data['target'].to_numpy()):
            conf_matrix[int(y)][int(x)] +=1

        #Generate confusion matrix pdf
        confusion_matrix.confusion_matrix(conf_matrix, figsize = (10,10), save_fig_name = sys.path[0] +"/"+  conf_title + 'validation.pdf', title = 'Validation data')

    if not test_data.empty:
        print("Classifying test data")
        test_data = pca_class.transform(test_data, 'target')
        pf.plot_transformation(test_data.iloc[::10, :], file_name= sys.path[0] +"/"+ plt_title + "_actual_val_data.pdf", ec_filepath = 'Python/error_color_coding.json')


        pred_val = pd.DataFrame(clf.predict(test_data.drop('target', axis = 1).to_numpy()))
        test_accuracy = (len(pred_val) - len( pred_val[0].compare(test_data['target']) ) )  /   len(pred_val)
        test_falsePositives = false_positives(test_data.copy(), pred_val.copy())
        print( "Test accuracy = " + str(test_accuracy))
        print("Test false positives = "+ str(test_falsePositives))
        toReturn += "\n\tTest accuracy = " + str(test_accuracy)
        toReturn += "\n\tTest false positives = " + str(test_falsePositives)

        print(pred_val)
        if conf_title:
            #Create confusion matrix
            conf_matrix = np.zeros([classes.shape[0], classes.shape[0]])

            for x,y in zip(pred_val.to_numpy(), test_data['target'].to_numpy()):
                conf_matrix[int(y)][int(x)] +=1

            #Generate confusion matrix pdf
            confusion_matrix.confusion_matrix(conf_matrix, figsize = (10,10), save_fig_name = sys.path[0] +"/"+ conf_title + 'test.pdf', title = 'Test data')

    plt.clf()
    plt.close()
    return toReturn + "\n\n"


if __name__ == "__main__":

    plot_latex = False
    if plot_latex:
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

    #Read data and assign labels
    print("reading data")
    training_data = pd.read_csv(sys.path[0] + "/../TrainingData/neodata/fault_all_nonoise_67.csv")
    validation_data = pd.read_csv(sys.path[0] + "/../ValidationData/neodata/fault_all_nonoise_67.csv")
    test_data = pd.read_csv(sys.path[0] + "/../TestData/neodata/fault_all_nonoise_67.csv")
    class_labels = np.arange(0,20+1,1)
    print(sys.path[0])

    print("Standardizing data")
    std = standardization.standardization(training_data, target = 'target')
    trn_std = std.transform(training_data)
    val_std = std.transform(validation_data)
    tst_std = std.transform(test_data)
    
    print2file = ""

    print2file += PCA_SVM(trn_std, val_std, class_labels, conf_title = 'PCA_SVM', plt_title= 'pca_reduc', test_data= tst_std)


    training_data_14 = pd.read_csv(sys.path[0] + "/../TrainingData/neodata/soltani_14d_nonoise_1200.csv")
    validation_data_14 = pd.read_csv(sys.path[0] + "/../ValidationData/neodata/soltani_14d_nonoise_1200.csv")
    test_data_14 = pd.read_csv(sys.path[0] + "/../TestData/neodata/soltani_14d_nonoise_100.csv")
    std = standardization.standardization(training_data_14, target = 'target')
    trn_std_14 = std.transform(training_data_14)
    val_std_14 = std.transform(validation_data_14)
    tst_std_14 = std.transform(test_data_14)
    print(tst_std_14)

    print2file += PCA_SVM(trn_std_14, val_std_14, class_labels, conf_title = 'PCA_SVM_14', plt_title= 'pca_reduc_14', test_data= tst_std_14)


    for i in range(3, np.max([len(trn_std_14.columns), len(trn_std.columns)])):
        print("\n\nDimension" + str(i))
        
        if(i<len(trn_std.columns)):
            print("11 dimensions")
            print2file += PCA_SVM(trn_std, val_std, class_labels, test_data=tst_std, pca_n = i)

        if(i<len(trn_std_14.columns)):
            print("\n14 dimensions")
            print2file += PCA_SVM(trn_std_14, val_std_14, class_labels,test_data= tst_std_14, pca_n = i)
        
f = open(sys.path[0] + "/pcasvm_grid_search.txt", 'w')
f.write(print2file)
f.close()