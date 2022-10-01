import numpy as np
from math import cos, pi, sin
from matplotlib.patches import Ellipse

# For overlaying standard deviation plots
def std_dev(mean, cov, stds,axs):
    vals, vecs = np.linalg.eig(cov[0:2,0:2])
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * stds * np.sqrt(vals)
    ell = Ellipse(xy=(mean[0], mean[1]),
                width=w, height=h,
                angle=theta, color='black',linewidth=3,linestyle='--',zorder=10)
    ell.set_facecolor('None')
    axs.add_artist(ell)