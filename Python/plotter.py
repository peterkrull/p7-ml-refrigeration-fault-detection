import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import pandas as pd

data = pd.DataFrame(read_csv('Ftrain4.csv')[0:6000],dtype=float)
label = pd.DataFrame({'label':[0 for _ in range(len(data))]})
index = pd.DataFrame({'index':[x for x in range(len(data))]})


data['CprPower'] = data['CprPower']/1000


xdata = index.join(data).join(label)


ran = [0,6000]

for i in xdata.drop('index',axis=1).drop('label',axis=1):
    plt.plot(xdata['index'][ran[0]:ran[1]],xdata[i][ran[0]:ran[1]])
plt.legend([x for x in xdata.drop('index',axis=1).drop('label',axis=1)])
plt.show()
