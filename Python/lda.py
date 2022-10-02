import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
        self.Sb = np.zeros((g,g))
        mu = data.drop(self.target_id, axis=1).mean()
        for k in data[self.target_id].unique():
            inner = data[data[self.target_id] == k].drop(self.target_id, axis=1).mean() - mu
            self.Sb += np.real(np.outer(inner,inner))

        # Calculate within-class covariance
        self.Sw = np.zeros((g,g))
        for k in data[self.target_id].unique():
            data_k = data[data[self.target_id] == k]
            muk = data_k.drop(self.target_id, axis=1).mean()
            inner = data_k.drop(self.target_id, axis=1) - muk            
            self.Sw += np.real(np.dot(inner.T,inner))

        # Calculate omega
        self.omega = np.dot( np.linalg.inv( self.Sw), self.Sb)

        # Get eigen vectors
        self.eig_val , self.eig_vec = np.linalg.eig(self.omega)

        # Save transformation matrix
        self.W = np.real(self.eig_vec[:,0:dims])

        # Plot largest eigen values
        plt.figure(figsize=(5,5))
        if len(self.eig_vec) > 20:
            plt.bar([x for x in range(1,20+1)],np.real(self.eig_val[0:20]))
        else:
            plt.bar([x for x in range(1,len(self.eig_vec)+1)],np.real(self.eig_val[0:len(self.eig_vec)]))

        s = sum(np.real(self.eig_val))
        print(f"Preserving {round(sum(np.real(self.eig_val[0:dims]))/s*100,2)}% of eigen value transformations",)

    def set_dims(self,dims:int):
        self.W = self.eig_vec[:,0:dims].astype('float')
       
    def transform(self, data : pd.DataFrame, target_id : str = None):

        # Determine appropriate label definition
        if self.target_id in data: target_id = self.target_id
        
        # Do transformation
        if target_id in data:
            Z = pd.DataFrame(data.drop(target_id,axis=1).astype('float').dot(self.W)).assign(target=data[target_id])
        else:
            Z = pd.DataFrame(data.astype('float').dot(self.W))
        
        # Return data
        return Z.astype('float')