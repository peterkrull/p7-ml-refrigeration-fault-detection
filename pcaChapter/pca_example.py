import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib as tpl

data = np.matrix('1.5,1;0.5,2;2,.5; 1,2;1,.5')
data = data.T

plt.scatter(data[0,:].tolist(), data[1,:].tolist())
#plt.show()

sample_mean = data.mean(1)
sample_cov = np.cov(data)

eig_val, eig_vec = np.linalg.eig(sample_cov)
print(eig_vec)
w = eig_vec[:,1]
print(w)

transformed_data = np.matmul(np.matrix(w),data-sample_mean)
print(transformed_data)

transformed_coordinate = np.matmul(np.matrix(w).T, np.matrix(transformed_data))
print(transformed_coordinate)


plt.scatter(transformed_coordinate[0,:].tolist(), transformed_coordinate[1,:].tolist(), color = "orange")
plt.plot(transformed_coordinate[0,:].T, transformed_coordinate[1,:].T, color = "orange")

len_trans = max(transformed_coordinate.shape)
for i in range(0, len_trans):
    plt.plot([transformed_coordinate[0,i], data[0,i]], [transformed_coordinate[1,i], data[1,i]], "--", color = "black")
    print(np.dot([transformed_coordinate[0,i]-data[0,i] ,transformed_coordinate[1,i]-data[1,i]], w))

tpl.save("pcaChapter/PCA_example_py.tex")

