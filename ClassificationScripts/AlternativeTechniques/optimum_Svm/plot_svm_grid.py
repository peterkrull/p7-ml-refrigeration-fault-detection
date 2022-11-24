import matplotlib.pyplot as plt
import pandas as pd
import json 
import sys
import numpy as np
from matplotlib import rc

def plot_gridsearch_log(grid_search_log : pd.DataFrame, show_figure : bool = False, save_figure : str = None):
    Z = np.zeros((len(data['param_C'].unique()), len(data['param_gamma'].unique())))

    for x,i in enumerate(data['param_C'].unique()):
        for y,j in enumerate(data['param_gamma'].unique()):
            val = data.loc[(data['param_C'] == i) & (data['param_gamma'] == j)]
            Z[x][y] = val['mean_test_score']

    fig = plt.figure(figsize=(6,4))
    plt.contourf(data['param_C'].unique(), data['param_gamma'].unique(), Z)

    plt.xscale('log')
    plt.yscale('log')
    cbar = plt.colorbar()
    cbar.set_label('Score')

    plt.xlabel('C')
    plt.ylabel('$\gamma$') 

    if save_figure:
        plt.savefig(save_figure)

    if show_figure:
        plt.show()
    

if __name__ == "__main__":
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    data = pd.read_json(sys.path[0] + "/svm_grid_search_log.json")
    plot_gridsearch_log(data, save_figure=sys.path[0] + "/grid_score.pdf")