import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import tikzplotlib as tpl

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

#data = np.matrix('1.5,1;0.5,2;2,.5; 1,2;1,.5')
data = np.matrix('0.6, 0.75, 1.4, 1.6, 2, 2.3,1.85, 2.15, 2.5, 2.8, 2.95, 3.25; 0.5, 0.4, 0.55, 0.75, 0.8, 0.95, 1.5, 1.75, 2.4, 2.65, 1.8, 2.85').transpose()
data = data.T

plt.scatter(data[0,:].tolist(), data[1,:].tolist(), label = r"$\boldsymbol{x}$")
#plt.show()

sample_mean = data.mean(1)
sample_cov = np.cov(data)
print("sample_mean: \n" + str(sample_mean))

eig_val, eig_vec = np.linalg.eig(sample_cov)
print("\neig vectors: \n" + str(eig_vec))
print("\neig_val:\n" + str(eig_val))
w = eig_vec[:,1]
print("\nw: \n" + str(w))

print("\ndata: \n" + str(data))
print("\ndata-sample_mean: \n" + str(data-sample_mean))
transformed_data = np.matmul(np.matrix(w),data-sample_mean)
print("\ntransformed_data: \n" + str(transformed_data))

print("\n Size data: \n", data.shape)
print("\n Size w: \n", np.matrix(w).T.shape)
transformed_coordinate = np.matmul(np.matrix(w), np.matrix(data))
print("\ntransformed_coordinate: \n" + str(transformed_coordinate))
print("\n Size transformed_coordinate: \n", np.matrix(transformed_coordinate).shape)
coordinate_along_w = np.matmul(np.matrix(w).T, np.matrix(transformed_coordinate))
print("\n coordinate along w: \n", str(coordinate_along_w))
print("\n Size coordinate_along_w: \n", coordinate_along_w.shape)

plt.scatter(coordinate_along_w[0,:].tolist(), coordinate_along_w[1,:].tolist(), color = "orange",zorder = 10, label = r"$\hat x$")
plt.plot(coordinate_along_w[0,:].T, coordinate_along_w[1,:].T, color = "orange",zorder = 9)

len_trans = max(transformed_coordinate.shape)
for i in range(0, len_trans):
    plt.plot([coordinate_along_w[0,i], data[0,i]], [coordinate_along_w[1,i], data[1,i]], "--", color = "grey")
    print(np.dot([coordinate_along_w[0,i]-data[0,i] ,coordinate_along_w[1,i]-data[1,i]], w))

plt.axis('equal')
plt.legend()
plt.savefig("pcaChapter/PCA_example_py.pdf")

plt.show()
#tpl.save("PCA_example_py.tex")

