import json
import matplotlib.pyplot as plt
import os 

file_name = "Python/error_color_coding.json"

error_colors = iter([plt.cm.tab20(i) for i in range(20)])
error_colors = list(error_colors)
error_colors.append(tuple((0.0, .35, .85, 1)))

with open(file_name, 'r') as f:
    data = json.load(f)
    print(data)
    for i in data:
        print(data[i]['color'])
        data[i]['color'] = error_colors[int(i)]

print(data)

os.remove(file_name)
with open(file_name, 'w') as f:
    json.dump(data, f, indent = 4)