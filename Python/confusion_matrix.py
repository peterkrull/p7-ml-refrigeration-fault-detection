import matplotlib.pyplot as plt
import numpy as np

def confusion_matrix(conf_matrix : np.matrix, axis_ticks : tuple = None, title : str = "Confusion matrix", normalize = True,save_fig_name : str = None, figsize : tuple = (10,10)):
    
    # Calculate accuracy of confusion matrix
    accuracy = np.sum(np.diag(conf_matrix))/np.sum(conf_matrix)
    
    # Setup figure
    fig , axs = plt.subplots(figsize=figsize)
    axs.imshow(np.sqrt(conf_matrix),cmap="Greens")
    axs.set_xlabel("Predicted class")
    axs.set_ylabel("True class")
    axs.set_title(f"{title} : Accuracy {round(accuracy*100,2)}%")

    width, height = conf_matrix.shape

    # Apply axis ticks
    if type(axis_ticks) != type(None):
        if len(axis_ticks) == 2:
            axs.set_xticks(axis_ticks[0],axis_ticks[1])
            axs.set_yticks(axis_ticks[0],axis_ticks[1])
        elif len(axis_ticks) == 1:
            axs.set_xticks(axis_ticks,axis_ticks)
            axs.set_yticks(axis_ticks,axis_ticks)
    else:
        axs.set_xticks([x for x in range(width)],[x for x in range(width)])
        axs.set_yticks([x for x in range(height)],[x for x in range(height)])

    # Apply numbers to matrix fields
    for x in range(width):
        for y in range(height):
            if normalize:
                axs.annotate(str(round(int(conf_matrix[x][y]) / sum(conf_matrix[x]),2)), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')
            else :
                axs.annotate(str(int(conf_matrix[x][y])), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')
                
    # Optional, export figure
    if save_fig_name:
        axs.get_figure().savefig(save_fig_name,format='pdf')
        
    return fig,axs

# For testing modifications
if __name__ == "__main__":
    
    mat = np.ones((20,20))
    fig,axs = confusion_matrix(mat)
    plt.show()