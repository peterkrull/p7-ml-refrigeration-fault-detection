import matplotlib.pyplot as plt
from io import StringIO
import pandas as pd
import sys
import os

#
#   In this program you can view all datasets by pressing left and right arrows
#

#starting dataset
n = 17

def filepath(n):#get the filepaths from numbers
    if n == 0:
        return "TrainingData/Ntrain1.csv"
    else:
        return f"TrainingData/Ftrain{n}.csv"

def on_press(event):
    global n #cuz it only works like this

    sys.stdout.flush()
    if event.key == 'left':
        n-=1
        redraw_Figure(n)
        
    if event.key == 'right':
        n=n+1
        redraw_Figure(n)
        
def redraw_Figure(n):
    data = pd.DataFrame(pd.read_csv(filepath(n))[0:6000],dtype=float)
    data['CprPower'] = data['CprPower']/1000
    axs.clear()
    axs.plot(data)
    plt.legend(data.columns, bbox_to_anchor=(1, 1))
    plt.title(f'Fault {n}')
    plt.xlabel('Press left and right arrows to change datasets')
    fig.canvas.draw()



file_name = filepath(n)

data = pd.DataFrame(pd.read_csv(file_name)[0:6000],dtype=float)
data['CprPower'] = data['CprPower']/1000

fig, axs = plt.subplots(figsize=(10,10))
fig.canvas.mpl_connect('key_press_event', on_press)

axs.plot(data)
plt.legend(data.columns, bbox_to_anchor=(1, 1))#, loc="center left")
plt.title(f'Fault {n}')
plt.xlabel('Press left and right arrows to change datasets')

plt.show()