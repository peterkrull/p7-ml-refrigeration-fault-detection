import matplotlib.pyplot as plt
from io import StringIO
import pandas as pd
import sys
import os

#
#   In this program you can view all datasets by pressing left and right arrows
#


fault_description = ['Normal operation','Tsuc positive offset','Tsup positive offset','Tret positive offset','Tdis positive offset','Pdis positive offset','Psuc positive offset','Compr low perf','Loose exp valve','Evap fan low perf','Cond fan low perf','Tsuc negative offset','Tsup negative offset','Tret positive offset','Tdis negative offset','Pdis negative offset','Psuc negative offset','Broken compressor','Broken exp valve','Broken evap fan', 'Blocked condenser fan']

#starting dataset
n = 17

def filepath(n):#get the filepaths from numbers
    if n == 0:
        return "ValidationData/NewValid_n1.csv"
    else:
        return f"ValidationData/NewValid_f{n}.csv"

def on_press(event):
    global n #cuz it only works like this

    sys.stdout.flush()
    if event.key == 'left':
        n = n-1 if n>0 else n
        redraw_Figure(n)
        
    if event.key == 'right':
        n = n+1 if n<20 else n
        redraw_Figure(n)
        
def redraw_Figure(n):
    data = pd.DataFrame(pd.read_csv(filepath(n))[0:10000],dtype=float)
    data['CprPower'] = data['CprPower']/1000
    axs.clear()
    axs.plot(data)
    plt.legend(data.columns, bbox_to_anchor=(1, 1))
    plt.title(f'Fault {n}: {fault_description[n]}')
    plt.xlabel('Press left and right arrows to change datasets')
    fig.canvas.draw()



file_name = filepath(n)

data = pd.DataFrame(pd.read_csv(file_name)[0:10000],dtype=float)
data['CprPower'] = data['CprPower']/1000

fig, axs = plt.subplots(figsize=(10,10))
fig.canvas.mpl_connect('key_press_event', on_press)

axs.plot(data)
plt.legend(data.columns, bbox_to_anchor=(1, 1))#, loc="center left")
plt.title(f'Fault {n}: {fault_description[n]}')
plt.xlabel('Press left and right arrows to change datasets')

plt.show()