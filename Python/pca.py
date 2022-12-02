import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Transform entire data frame
class PCA_reducer:

    def __init__(self, data : pd.DataFrame, dims : int, target_id : str = None, scree_plot : bool = False) -> None:

        # Extract labels and drop
        if target_id:
            labels = data[target_id]
            data = data.drop(target_id, axis=1)
        
        # Calculate covariance matrix
        cov = data.cov()

        # Get eigen vectors
        eig_val , eig_vec = np.linalg.eig(cov)

        #Sort eigen values + vectors, largest -> smallest
        idx = eig_val.argsort()[::-1]
        eig_val = eig_val[idx]
        eig_vec = eig_vec[idx]

        #Create projection matrix
        self.W = eig_vec[:,0:dims]
        self.mean = data.mean(axis=0)
        self.target_id = target_id

        s = sum(np.real(eig_val))
        self.preserved_eigval = round(sum(np.real(eig_val[0:dims]))/s*100,2)
        print(f"Preserving {self.preserved_eigval}% of eigen value transformations",)

        if scree_plot:
            eig_val[::-1].sort()
            eig_total = eig_val.sum()
            plt.figure(figsize=(6,4))
            plt.bar([x for x in range(1,len(eig_vec)+1)], np.real(eig_val[0:len(eig_vec/eig_total)]))
            plt.xticks([x for x in range(1, len(eig_vec)+1)])
            plt.xlabel("Eigenvalue")
            plt.ylabel("Propotion of total eigenvalue")

    def transform(self, data : pd.DataFrame, target_id : str = None):

        # Determine appropriate label definition
        if self.target_id in data: target_id = self.target_id

        # Extract labels and drop
        if target_id:
            labels = data[target_id]
            data = data.drop(target_id, axis=1)

        # Do transform
        sub = data.transpose()#sub(self.mean).transpose()
        z = pd.DataFrame(np.dot(self.W.transpose(),sub.to_numpy()))

        # Add labels back to reduced data
        if target_id:
            z = pd.concat([
                z.transpose().reset_index(drop=True).reset_index(drop=True) ,
                pd.DataFrame({target_id:labels}).reset_index(drop=True)
                ],axis=1)
        else:
            z = z.transpose()
        
        # Return data
        return z