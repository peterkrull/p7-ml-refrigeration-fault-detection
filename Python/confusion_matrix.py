import matplotlib.pyplot as plt
import numpy as np

def confusion_matrix(conf_matrix : np.matrix, axis_ticks : tuple = None, title : str = "Confusion matrix", normalize = True,save_fig_name : str = None, figsize : tuple = (10,10)):
    plt.imshow(np.sqrt(conf_matrix),cmap="Greens",figsize=figsize)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(title)

    width, height = conf_matrix.shape

    if type(axis_ticks) != type(None):
        if len(axis_ticks) == 2:
            plt.xticks(axis_ticks[0],axis_ticks[1])
            plt.yticks(axis_ticks[0],axis_ticks[1])
        else:
            plt.xticks(axis_ticks,axis_ticks)
            plt.yticks(axis_ticks,axis_ticks)

    for x in range(width):
        for y in range(height):
            if normalize:
                plt.annotate(str(round(int(conf_matrix[x][y]) / sum(conf_matrix[x]),2)), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')
            else :
                plt.annotate(str(int(conf_matrix[x][y])), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')
                
    if save_fig_name:
        plt.savefig(save_fig_name,format='pdf')