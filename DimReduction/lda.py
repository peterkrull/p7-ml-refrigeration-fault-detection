import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import threading

# Transform entire data frame
class LDA_reducer:

    def __init__(self, data : pd.DataFrame, dims : int, target_id : str, frac : int = 1) -> None:


        # Extract labels and drop
        self.target_id = target_id
        
        # Get number of unique classes
        g = len(data.drop(self.target_id, axis=1).any())

        # Get mean of data and (optionally) downsample data set
        if frac < 1:
            data = data.sample(frac=frac)

        # Calculate between-class covariance
        mu = data.drop(self.target_id, axis=1).mean()
        Sb = np.zeros((g,g))
        for k in data[self.target_id].unique():
            inner = data[data[self.target_id] == k].drop(self.target_id, axis=1).mean() - mu
            Sb += np.outer(inner,inner)


        # Calculate within-class covariance
        Ss = np.zeros((g,g))
        i = len(data[self.target_id].unique())
        for e,k in enumerate(data[self.target_id].unique()):
            print("Processing",e+1,"of",i,"targets")
            data_k = data[data[self.target_id] == k]
            muk = data_k.drop(self.target_id, axis=1).mean()
            for j in range(len(data_k)):
                inner = data_k.drop(self.target_id, axis=1).iloc[j] - muk
                Ss += np.outer(inner,inner)

        # Calculate omega
        omega = np.dot( np.linalg.inv( Ss ), Sb )

        # Get eigen vectors
        self.eig_val , self.eig_vec = np.linalg.eig(omega)

        # Save transformation matrix
        self.W = self.eig_vec[:,0:dims]

        # Plot largest eigen values
        plt.figure(figsize=(5,5))
        if len(self.eig_vec) > 20:
            plt.bar([x for x in range(20)],self.eig_val[0:20])
        else:
            plt.bar([x for x in range(len(self.eig_vec))],self.eig_val[0:len(self.eig_vec)])

        s = sum(np.real(self.eig_val))
        print(f"Preserving {round(sum(np.real(self.eig_val[0:dims]))/s*100,2)}% of eigen value transformations",)

    def set_dims(self,dims:int):
        self.W = self.eig_vec[:,0:dims]
        # s = sum(np.real(self.eig_val))
        # print(f"Preserving {round(sum(np.real(self.eig_val[0:dims]))/s*100,2)}% of eigen value transformations",)

    def transform(self, data : pd.DataFrame, target_id : str = None):

        # Determine appropriate label definition
        if self.target_id in data: target_id = self.target_id

        # Save target column
        target = data.pop(target_id) if target_id in data else None

        # Do transformation
        Z = pd.DataFrame(data.dot(self.W))

        # Reassign target if available
        if type(target) != type(None) :
            print("Adding target")
            Z = Z.assign(target=target)
        
        # Return data
        return Z