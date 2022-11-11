import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix as confMatrix



def confusion_matrix(true_Label : np.array,predicted_label : np.array ,axis_ticks : tuple = None, title : str = "Confusion matrix", normalize = True,save_fig_name : str = None, figsize : tuple = (10,10),eval_labels = True):
    
    conf_matrix = confMatrix(true_Label, predicted_label)             # Make confusion matrix


    # Calculate accuracy of confusion matrix
    accuracy = np.sum(np.diag(conf_matrix))/np.sum(conf_matrix)

    each_accuracy = np.zeros(conf_matrix.shape[0])
    for i, (num, denum) in enumerate(zip(np.diag(conf_matrix), np.sum(conf_matrix, axis = 1))):
        if denum != 0:
            each_accuracy[i] = num/denum

    # Setup figure
    fig , axs = plt.subplots(figsize=figsize)
   
   
    axs.imshow(np.sqrt(conf_matrix),cmap="Greens")
    axs.set_xlabel("Predicted class")
    axs.set_ylabel("True class")
    axs.set_title(f"{title} : Accuracy {round(accuracy*100,3)}%")

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
            
            if eval_labels:
                text = 'perfect' if each_accuracy[y] == 1.0 else (str(round(each_accuracy[y]*100,2)) + '%')
                axs.text(width,y,f"{text}",color="green" if each_accuracy[y] > 0.95 else "black")
            
            rounded = 0.0
            if sum(conf_matrix[x]) != 0:
                rounded = round(int(conf_matrix[x][y]) / sum(conf_matrix[x]),2)
            
            if normalize:
                text = rounded    
            else:
                text = int(conf_matrix[x][y])
                
            axs.annotate(str(text), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center',color= "white" if rounded > 0.7 else "black")
                
    # Optional, export figure
    if save_fig_name:
        axs.get_figure().savefig(save_fig_name,format='pdf',bbox_inches='tight',)
        
    return fig,axs

# For testing modifications
if __name__ == "__main__":
    
    mat = np.ones((20,20))
    fig,axs = confusion_matrix(mat)
    plt.show()