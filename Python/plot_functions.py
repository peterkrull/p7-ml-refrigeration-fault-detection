import matplotlib.pyplot as plt
import pandas as pd
import json as js

def plot_transformation(data :pd.DataFrame, file_name : str = None, target : str = 'target', plt_show : bool = False, ec_filepath : str = f'error_color_coding.json', figsize : tuple = (6,4), legend_pos : tuple = (1.25, .5)):
    """
        Generates plots of two dimensional scatter data, for each fault. Useful for PCA and LDA

        data : Pandas Dataframe containing two dimensional data and label
        file_name : Name of exported figure, wont export if not defined
        target : Column name containing labels
        plt_show : Show plot? std False
        ec_filepath : file path to error color file
        fig_size : size of figure, standard (10,7.5)
        legend_pos : Position of legend, right edge of plot = 1, upper edge of plot = 1
    """
    
    error_colors = js.load(open(ec_filepath))

    plt.figure(figsize = figsize)
    for fault in data[target].unique():
        fault = int(fault)
        plt.scatter(data.loc[data[target] == fault][0],data.loc[data[target] == fault][1], label = "Fault " + str(fault), color = error_colors[str(fault)]['color'])

    #lgd = plt.legend(bbox_to_anchor=(1, -0.125), loc="lower left")
    lgd = plt.legend(loc="right",bbox_to_anchor = legend_pos)
    if file_name:
        plt.savefig(file_name, bbox_extra_artists=(lgd,), bbox_inches='tight')

    if plt_show:
        plt.show()
    lgd.remove()
    plt.close()