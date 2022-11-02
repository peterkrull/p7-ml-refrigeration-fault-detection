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
import itertools 

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

def no_noise(train_data: pd.DataFrame, val_data: pd.DataFrame, classes: pd.DataFrame):
    #Initiate PCA-algo to find dim reduction matrix w (Stored in class)
    pca_class = pca.PCA_reducer(train_data, 2,'target')

    #Dim reduce training data & val data
    trans_data = pca_class.transform(train_data, 'target')
    val_red_data = pca_class.transform(val_data, 'target')

    print(trans_data)
    error_colors = js.load(open(f'Python/error_color_coding.json'))
    # error_colors = iter([plt.cm.tab20(i) for i in range(20)])
    # error_colors = list(error_colors)
    # #itertools.chain(error_colors, ((0.0, .35, .7, .01)))
    # error_colors.append(tuple((0.0, .35, .85, 1)))
    # for i in error_colors:
    #     print("tuple" + str(i))

    print(error_colors)

    for target in trans_data['target'].unique():
        print(target)
        plt.scatter(trans_data.loc[trans_data["target"] == target][0],trans_data.loc[trans_data["target"] == target][1], label = "Fault " + str(target), color = error_colors[str(target)]['color'])

    lgd = plt.legend(bbox_to_anchor=(1, -0.125), loc="lower left")
    plt.savefig("machine learning scripts/pca_reduc.pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


    #Fit svm model to dim red data
    clf = svm.SVC(decision_function_shape='ovr', gamma = 0.01, C = 1000)
    clf.fit(trans_data.drop('target', axis = 1).to_numpy(), trans_data['target'].to_numpy())

    print(clf.score(val_red_data.drop('target',axis = 1).to_numpy(), val_red_data['target'].to_numpy()))

    val_labels = val_red_data['target'].to_numpy()

    val_noLabel = val_red_data.drop('target', axis = 1).to_numpy()


    pred = clf.predict(val_noLabel)

    conf_matrix = np.zeros([classes.shape[0], classes.shape[0]])

    for x,y in zip(pred,val_labels):
        conf_matrix[int(x)][int(y)] +=1

    confusion_matrix.confusion_matrix(conf_matrix, figsize = (10,10), save_fig_name = 'machine learning scripts/confusion_matrix_pcasvm.pdf', title = 'Validation data')



def with_noise(train_data: pd.DataFrame, val_data: pd.DataFrame, classes: pd.DataFrame):
    noise_des = js.load(open(f'Python/noise_description.json'))

    for i in train_data.drop('target', axis = 1):
        noise = np.random.normal(noise_des[i]['mean'], noise_des[i]['var'], size = train_data[i].shape[0])
        train_data[i] = train_data[i].to_numpy() + noise
        noise = np.random.normal(noise_des[i]['mean'], noise_des[i]['var'], size = val_data[i].shape[0])
        val_data[i] = val_data[i].to_numpy() + noise
    
    pca_class = pca.PCA_reducer(train_data, 2 ,'target')
    
    trans_data_red = pca_class.transform(train_data, 'target')
    val_data_red   = pca_class.transform(val_data, 'target')

    clf = svm.SVC(decision_function_shape='ovr', gamma = 0.01, C = 1000)
    clf.fit(trans_data_red.drop('target', axis = 1).to_numpy(), trans_data_red['target'].to_numpy())

    print(clf.score(val_data_red.drop('target',axis = 1).to_numpy(), val_data_red['target'].to_numpy()))
    
    val_labels = val_data_red['target'].to_numpy()

    val_noLabel = val_data_red.drop('target', axis = 1).to_numpy()


    pred = clf.predict(val_noLabel)

    conf_matrix = np.zeros([classes.shape[0], classes.shape[0]])

    for x,y in zip(pred,val_labels):
        conf_matrix[int(x)][int(y)] +=1

    confusion_matrix.confusion_matrix(conf_matrix, figsize = (10,10), save_fig_name = 'machine learning scripts/confusion_matrix_pcasvm_with_noise.pdf', title = 'Validation data')


if __name__ == "__main__":
    training_data = pd.read_csv(sys.path[0] + "\..\\TrainingData\\neodata\\fault_all_10.csv")
    class_labels = np.arange(0,20+1,1)

    training_data , validation_data = get_valData(training_data)

    no_noise(training_data, validation_data, class_labels)
    print(plt.cm.get_cmap('tab20c'))
    #with_noise(training_data, validation_data, class_labels)


