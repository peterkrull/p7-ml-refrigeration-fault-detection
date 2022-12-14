import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
from lda import LDA_reducer

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

x1=[0.6, 0.75, 1.4, 1.6, 2, 2.3]
y1=[0.5, 0.4, 0.55, 0.75, 0.8, 0.95]
x2=[1.85, 2.15, 2.5, 2.8, 2.95, 3.25]
y2=[1.5, 1.75, 2.4, 2.65, 1.8, 2.85]

colors = ["", ""]
colors[0] = plt.scatter(x1, y1,zorder = 10, label = "Class 1").get_facecolor()
colors[1] = plt.scatter(x2, y2, zorder = 9, label = "Class 2").get_facecolor()

train_data = pd.DataFrame()
data_head = ["x","y","target"]

toAdd = np.array((np.array(x1).transpose(), np.array(y1).transpose(), np.zeros(np.array(x1).shape)))
train_data = train_data.append(pd.DataFrame(toAdd.transpose(), columns = data_head))
toAdd = np.array((np.array(x2).transpose(), np.array(y2).transpose(), np.ones(np.array(x1).shape)))
train_data = train_data.append(pd.DataFrame(toAdd.transpose(), columns = data_head))
train_data = train_data.reset_index()
train_data = train_data.drop('index', axis = 1)
print(train_data)


lda1 = LDA_reducer(train_data, 1, "target", scree_plot = False)

trans_data = lda1.transform(train_data, target_id = "target")
print(trans_data)


trans_point = np.matmul(lda1.W, trans_data.drop('target', axis = 1).to_numpy().transpose()).transpose()
trans_point = pd.DataFrame(trans_point, columns = ['x', 'y'])
trans_point['target'] = trans_data['target']

print(trans_point)

for i in range(trans_point.shape[0]):
    print(i)
    print(trans_point.iloc[0])
    plt.plot([trans_point.iloc[i]['x'],train_data.iloc[i]['x']], [trans_point.iloc[i]['y'],train_data.iloc[i]['y']], "--", color = 'grey')

for i in range (2):
    plot_points = trans_point.loc[trans_point["target"] == i]    
    print(plot_points)
    plt.scatter(plot_points['x'].to_numpy(), plot_points['y'].to_numpy(), color = colors[i], zorder = 8)

w_line = trans_point.drop('target', axis = 1)
plt.plot(w_line['x'].to_numpy(),w_line['y'].to_numpy(),  color = 'grey')

plt.legend()
plt.axis('equal')
plt.savefig("LDA_plot_example.pdf")

