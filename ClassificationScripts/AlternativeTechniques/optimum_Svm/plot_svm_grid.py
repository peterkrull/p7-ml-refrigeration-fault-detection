import matplotlib.pyplot as plt
import pandas as pd
import json 
import sys
import numpy as np
from matplotlib import rc

def plot_gridsearch_log(grid_search_log : pd.DataFrame, show_figure : bool = False, save_figure : str = None):
    Z = np.zeros((len(grid_search_log['param_C'].unique()), len(grid_search_log['param_gamma'].unique())))

    x_unique = grid_search_log['param_C'].unique()
    y_unique = grid_search_log['param_gamma'].unique()
    x_unique.sort()
    y_unique.sort()

    df = grid_search_log[['param_C', 'param_gamma', 'mean_test_score']].copy()
    
    df=df.pivot('param_gamma','param_C', 'mean_test_score')
    gamma = df.index.values
    C = df.columns.values
    score = df.values
    x,y=np.meshgrid(C, gamma)

    fig = plt.figure(figsize=(6,4))
    plt.contourf(x,y, score)

    plt.xscale('log')
    plt.yscale('log')
    cbar = plt.colorbar()
    cbar.set_label('Score')

    plt.xlabel('C')
    plt.ylabel('$\gamma$') 

    if save_figure:
        print("Saving figure")
        plt.savefig(save_figure)

    if show_figure:
        plt.show()
    

if __name__ == "__main__":
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    data = pd.read_json(sys.path[0] + "/svm_grid_search_log.json")
    plot_gridsearch_log(data, save_figure=sys.path[0] + "/grid_score.pdf")