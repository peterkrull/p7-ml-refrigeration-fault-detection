import sys
from tkinter.tix import COLUMN
sys.path.append(sys.path[0] + "\..")
from Python import pca
from Python import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.svm as svm
import json as js
from Python import standardization 

#Load training data from file

def get_valData(train_data: pd.DataFrame):
    validation_data = pd.DataFrame()
    data_header = train_data.columns
    for i in range(int(train_data.shape[0]/10)):
        index = np.random.randint(0, train_data.shape[0])
        validation_data = validation_data.append(train_data.loc[index])
        train_data = train_data.drop(index)
        train_data = train_data.reset_index()
        train_data = train_data.drop('index', axis = 1)
    validation_data = validation_data.reset_index()
    validation_data = validation_data.drop('index', axis = 1)

    return train_data, validation_data

def PCA_SVM(train_data: pd.DataFrame, val_data: pd.DataFrame, classes: pd.DataFrame, conf_title: str = 'confusion_matrix_pcasvm', plt_title :str = 'pca_reduc', plt_show : bool = False, gamma = .01, c = 1000):
    #Initiate PCA-algo to find dim reduction matrix w (Stored in class)
    pca_class = pca.PCA_reducer(train_data, 2,'target')

    #Dim reduce training data & val data
    print("Transforming data")
    trans_data = pca_class.transform(train_data, 'target')
    val_red_data = pca_class.transform(val_data, 'target')

    #Get colors for the errors
    error_colors = js.load(open(f'Python/error_color_coding.json'))

    print("Plotting data")
    #Plot errors in scatter plot
    for target in trans_data['target'].unique():
        target = int(target)
        plt.scatter(trans_data.loc[trans_data["target"] == target][0],trans_data.loc[trans_data["target"] == target][1], label = "Fault " + str(target), color = error_colors[str(target)]['color'])

    lgd = plt.legend(bbox_to_anchor=(1, -0.125), loc="lower left")
    plt.savefig("machine learning scripts/" + plt_title + ".pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')

    if plt_show:
        plt.show()


    print("Fitting data")
    #Fit svm model to dim red data
    clf = svm.SVC(decision_function_shape='ovo', gamma = gamma, C = c)
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
    confusion_matrix.confusion_matrix(conf_matrix, figsize = (10,10), save_fig_name = 'machine learning scripts/' + conf_title + 'validation.pdf', title = 'Validation data')
    plt.clf()





def add_noise (data: pd.DataFrame, target: str = 'target'):
    noise_des = js.load(open(f'Python/noise_description.json'))

    for i in data.drop(target, axis = 1):
        noise = np.random.normal(noise_des[i]['mean'], noise_des[i]['var'], size = data[i].shape[0])
        data[i] = data[i].to_numpy() + noise
    
    return data


if __name__ == "__main__":
    #Read data and assign labels
    training_data = pd.read_csv(sys.path[0] + "\..\\TrainingData\\neodata\\fault_all_dropfeatures_67.csv")
    test_data = pd.read_csv(sys.path[0] + "\..\\ValidationData\\neodata\\fault_all_dropfeatures_67.csv")
    training_data_noisy = pd.read_csv(sys.path[0] + "\..\\TrainingData\\neodata\\fault_all_noise_dropfeatures_67.csv")
    test_data_noisy = pd.read_csv(sys.path[0] + "\..\\ValidationData\\neodata\\fault_all_noise_dropfeatures_67.csv")
    class_labels = np.arange(0,20+1,1)

    #training_data , validation_data = get_valData(training_data)

    #PCA_SVM(training_data, test_data, class_labels)
    
    non_noisy_stand = standardization.standardization(training_data, target = 'target')
    training_data_std = non_noisy_stand.transform(training_data)
    test_data_std = non_noisy_stand.transform(test_data)

    print("Doing std non noisy data")
    PCA_SVM(training_data_std.iloc[::10,:], test_data_std.iloc[::10,:], class_labels, conf_title = 'confusion_matrix_pcasvm_std', plt_title= 'pca_reduc_std')

    # print("Doing reduced dataset")
    # i = 0
    # for j in range(8):
    #     gamma = .01*pow(10,i)
    #     c = 0.01*pow(10,j)
    #     print("Gamma: {}, c: {}", gamma, c)
    #     PCA_SVM(training_data_noisy.iloc[::25, :], test_data.iloc[::15,:], class_labels, conf_title = 'confusion_matrix_pcasvm_red', plt_title = 'pca_reduc_red', gamma= gamma, c= c)

    # print("Doing old dataset")
    training_data_old =  pd.read_csv(sys.path[0] + "\..\\TrainingData\\neodata\\fault_all_dropFeatures_67.csv")
    test_data_old = pd.read_csv(sys.path[0] + "\..\\ValidationData\\neodata\\fault_all_10.csv")
    # PCA_SVM(add_noise(training_data_old), add_noise(test_data_old), class_labels, conf_title = 'confusion_matrix_pcasvm_old', plt_title= 'pca_reduc_old')

    training_data_old = add_noise(training_data_old)
    print("Doing Søren noise add validation data")
    train_data, val_data = get_valData(training_data_old)
    PCA_SVM(train_data, val_data, class_labels, conf_title = 'confusion_matrix_pcasvm_old_myval', plt_title= 'pca_reduc_old_myval')

    #print("Doing noisy data")
    #PCA_SVM(training_data_noisy, test_data_noisy, class_labels, conf_title = 'confusion_matrix_pcasvm_with_noise', plt_title= 'pca_reduc_noise')


