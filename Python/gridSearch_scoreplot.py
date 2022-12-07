import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rc

def plot_gridsearch_log(grid_search_log : pd.DataFrame, show_figure : bool = False, save_figure : str = None, x_label : str = 'param_C', y_label : str = 'param_gamma', score_label : str = 'mean_test_score', plot_max : bool = False,fig_size=(4,4)):
    """ function for plotting contour plot of grid search log
        
        grid_search_log : Pandas DataFrame, log from gridsearch 
        show_figure : bool standard false, show figure once it is plotted
        save_figure : string standard none, if not none, saves plot to file with name given by save_figure
        x_label : string standard 'param_C', grid_search_log variable name for x-axis of plot
        y_label : string standard 'param_gamma', grid_search_log variable name for y-axis of plot
        score_label : string standard 'mean_test_score', grid_search_log variable name for contours of plot
    """


    df = grid_search_log[[x_label, y_label, score_label]].copy()
    
    df=df.pivot(y_label,x_label, score_label)
    gamma = df.index.values
    C = df.columns.values
    score = df.values
    x,y=np.meshgrid(C, gamma)

    fig = plt.figure(figsize=fig_size)
    plt.contourf(x,y, score, levels = np.linspace(0,1,21))#vmin = 0, vmax = 1, alpha = 1)

    plt.xscale('log')
    plt.yscale('log')
    cbar = plt.colorbar()
    cbar.set_label('Score')
    cbar.set_ticks(np.linspace(0,1,11))

    plt.xlabel('C')
    plt.ylabel('$\gamma$') 

    if plot_max:
        max_scores = grid_search_log[grid_search_log[score_label] == grid_search_log[score_label].max()].copy()
        plt.scatter(max_scores[x_label], max_scores[y_label], marker= 'x', color = 'red')

    if save_figure:
        plt.tight_layout()
        plt.savefig(save_figure)
        print("Score plot saved")

    if show_figure:
        plt.show()