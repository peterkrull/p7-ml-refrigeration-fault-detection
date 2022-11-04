import matplotlib.pyplot as plt
from io import StringIO
import pandas as pd
import sys
import os

#
#   In this program you can view all datasets by pressing left and right arrows
#

dir_path = os.path.dirname(os.path.realpath(__file__))

#data file to read
file_name = 'fault_all_noise_67.csv'

#setpoints/fault
segments=18

fault_description = ['Normal operation','Tsuc positive offset','Tsup positive offset','Tret positive offset','Tdis positive offset','Pdis positive offset','Psuc positive offset','Compr low perf','Loose exp valve','Evap fan low perf','Cond fan low perf','Tsuc negative offset','Tsup negative offset','Tret positive offset','Tdis negative offset','Pdis negative offset','Psuc negative offset','Broken compressor','Broken exp valve','Broken evap fan', 'Blocked condenser fan']

fault = 0
setpoint = 0

def on_press(event):
    pass
    global fault #cuz it only works like this
    global setpoint
    print(event.key)

    sys.stdout.flush()
    if event.key == 'left':
         fault = fault-1 if fault>0 else fault
         redraw_Figure(fault,setpoint)
        
    if event.key == 'right':
        fault = fault+1 if fault<20 else fault
        redraw_Figure(fault,setpoint)

    if event.key == 'up':
        setpoint = setpoint+1 if setpoint<segments-1 else setpoint
        redraw_Figure(fault,setpoint)

    if event.key == 'down':
        setpoint = setpoint-1 if setpoint>0 else setpoint
        redraw_Figure(fault,setpoint)
        
def redraw_Figure(fault,setpoint):
    fault_segment = data[data['target']==fault]
    seg_size = len(fault_segment)/segments
    axs[0].clear()
    axs[1].clear()
    axs[0].plot(fault_segment)
    axs[0].set_title(f'fault {fault}\n {fault_description[fault]}')

    start = int(setpoint*seg_size)
    end = int(start+seg_size)

    axs[1].plot(fault_segment[start:end])
    axs[1].set_title(f"setpoint {setpoint+1}")

    fig.canvas.draw()





data = pd.DataFrame(pd.read_csv(dir_path+'/'+file_name),dtype=float)
if 'CprPower' in data.columns:
    data['CprPower'] = data['CprPower']/1000

fig, axs = plt.subplots(1,2)
fig.canvas.mpl_connect('key_press_event', on_press)
fig.suptitle(file_name)

fault_segment = data[data['target']==fault]
seg_size = len(fault_segment)/segments

axs[0].plot(fault_segment)
axs[0].set_title(f'fault {fault}\n {fault_description[fault]}')

start = int(setpoint*seg_size)
end = int(start+seg_size)

axs[1].plot(fault_segment[start:end])
axs[1].set_title(f'setpoint {setpoint+1}')
plt.legend(data.columns, bbox_to_anchor=(1, 1))


plt.show()