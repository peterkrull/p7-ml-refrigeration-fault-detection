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
import json as js
import standardization 
import plot_functions as pf
import lda 


def get_valData(train_data: pd.DataFrame):
    validation_data = pd.DataFrame()
    validation_data = train_data.sample(int(train_data.shape[0]/20  ))
    train_data = train_data.drop(validation_data.index)
    validation_data = validation_data.reset_index()
    validation_data = validation_data.drop('index', axis = 1)
    validation_data = validation_data.sort_values(by = ['target'])

    return train_data, validation_data



def PCA_SVM(train_data: pd.DataFrame, val_data: pd.DataFrame, classes: pd.DataFrame, conf_title: str = 'confusion_matrix_pcasvm', plt_title :str = 'pca_reduc', plt_show : bool = False, gamma = .01, c = 1000, act_valdata :pd.DataFrame= None):
    #Initiate PCA-algo to find dim reduction matrix w (Stored in class)
    pca_class = pca.PCA_reducer(train_data, 2,'target', scree_plot= True)
    plt.savefig('machine learning scripts/pca_svm_screeplot.pdf', bbox_inches='tight')

    #Dim reduce training data & val data
    print("Transforming data")
    trans_data = pca_class.transform(train_data, 'target')
    val_red_data = pca_class.transform(val_data, 'target')

    #Get colors for the errors
    error_colors = js.load(open(f'Python/error_color_coding.json'))

    print("Plotting data")
    pf.plot_transformation(trans_data.iloc[::10, :], file_name = "machine learning scripts/" + plt_title + ".pdf", ec_filepath = 'Python/error_color_coding.json')

    pf.plot_transformation(val_red_data.iloc[::10, :], file_name="machine learning scripts/" + plt_title + "_val_data.pdf", ec_filepath = 'Python/error_color_coding.json')


    print("Fitting data")
    #Fit svm model to dim red data
    parameters = {'kernel':['linear', 'rbf'], 'decision_function_shape':['ovo', 'ovr'], 'C' : [10**x for x in range(-1,6)], 'gamma': [10**x for x in range(-3, 3)]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters, verbose = 2, n_jobs=8)
    clf.fit(trans_data.drop('target', axis = 1).to_numpy(), trans_data['target'].to_numpy())
    print(clf.best_estimator_)

    f = open("machine learning scripts/pcasvm_grid_search.txt", 'w')
    f.write(str(clf.best_estimator_))
    f.close()


    clf.fit(trans_data.drop('target', axis = 1).to_numpy(), trans_data['target'].to_numpy())


    trans_labels = trans_data['target'].to_numpy()
    #val_noLabel = val_red_data.drop('target', axis = 1).to_numpy()

    print("Classifying training data")
    pred_train = clf.predict(trans_data.drop('target', axis=1).to_numpy())

    #Create confusion matrix
    conf_matrix = np.zeros([classes.shape[0], classes.shape[0]])

    for x,y in zip(pred_train,trans_labels):
        conf_matrix[int(y)][int(x)] +=1

    #Generate confusion matrix pdf
    confusion_matrix.confusion_matrix(conf_matrix, figsize = (10,10), save_fig_name = 'machine learning scripts/' + conf_title + '.pdf', title = 'Training data')

    print("Classifying validation data")
    #Use validation data
    pred_val = clf.predict(val_red_data.drop('target', axis = 1).to_numpy())
    print(clf.score(val_red_data.drop('target', axis = 1).to_numpy(), val_red_data['target']))

    #Create confusion matrix
    conf_matrix = np.zeros([classes.shape[0], classes.shape[0]])

    for x,y in zip(pred_val, val_red_data['target'].to_numpy()):
        conf_matrix[int(y)][int(x)] +=1

    #Generate confusion matrix pdf
    confusion_matrix.confusion_matrix(conf_matrix, figsize = (10,10), save_fig_name = 'machine learning scripts/' + conf_title + 'validation.pdf', title = 'Validation data (Subset of training data)')

    if not act_valdata.empty:
        act_valdata = pca_class.transform(act_valdata, 'target')
        pf.plot_transformation(act_valdata.iloc[::10, :], file_name="machine learning scripts/" + plt_title + "_actual_val_data.pdf", ec_filepath = 'Python/error_color_coding.json')

        pred_val = clf.predict(act_valdata.drop('target', axis = 1).to_numpy())
        print(clf.score(act_valdata.drop('target', axis = 1).to_numpy(), act_valdata['target']))

        #Create confusion matrix
        conf_matrix = np.zeros([classes.shape[0], classes.shape[0]])

        for x,y in zip(pred_val, act_valdata['target'].to_numpy()):
            conf_matrix[int(y)][int(x)] +=1

        #Generate confusion matrix pdf
        confusion_matrix.confusion_matrix(conf_matrix, figsize = (10,10), save_fig_name = 'machine learning scripts/' + conf_title + '_actual_validation.pdf', title = 'Validation data')

    plt.clf()





def add_noise (data: pd.DataFrame, target: str = 'target'):
    noise_des = js.load(open(f'Python/noise_description.json'))

    for i in data.drop(target, axis = 1):
        noise = np.random.normal(noise_des[i]['mean'], noise_des[i]['var'], size = data[i].shape[0])
        data[i] = data[i].to_numpy() + noise
    
    return data


if __name__ == "__main__":

    plot_latex = True
    if plot_latex:
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

    #Read data and assign labels
    training_data = pd.read_csv(sys.path[0] + "/../TrainingData/neodata/fault_all_nonoise_67.csv")
    test_data = pd.read_csv(sys.path[0] + "/../ValidationData/neodata/fault_all_nonoise_67.csv")
    training_data_noisy = pd.read_csv(sys.path[0] + "/../TrainingData/neodata/fault_all_noise_67.csv")
    test_data_noisy = pd.read_csv(sys.path[0] + "/../ValidationData/neodata/fault_all_noise_67.csv")
    class_labels = np.arange(0,20+1,1)


    non_noisy_stand = standardization.standardization(training_data, target = 'target')
    training_data_std = non_noisy_stand.transform(training_data)
    test_data_std = non_noisy_stand.transform(test_data)

    print("Doing std non noisy data")
    #PCA_SVM(training_data_std, test_data_std, class_labels, conf_title = 'confusion_matrix_pcasvm_actual_vali_non_noisy', plt_title= 'pca_reduc_actual_vali_non_noisy')


    std_newdata = standardization.standardization(training_data_noisy, target = 'target')
    train_std = std_newdata.transform(training_data_noisy)
    test_std = std_newdata.transform(test_data_noisy)
    print(test_std)
    print("Doing Lau noise own validation data")
    train_data, val_data = get_valData(train_std)
    PCA_SVM(train_data, val_data, class_labels, conf_title = 'confusion_matrix_pcasvm_noise_my_val', plt_title= 'pca_reduc_noise_my_vali', act_valdata= test_std)



    #print("Doing noisy data")
    #PCA_SVM(training_data_noisy, test_data_noisy, class_labels, conf_title = 'confusion_matrix_pcasvm_with_noise', plt_title= 'pca_reduc_noise')


