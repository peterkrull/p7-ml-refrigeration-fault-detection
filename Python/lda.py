import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Reduce dimensionality using Fisher's LDA
class LDA_reducer:

    # Initialize the LDA by using a dataset
    def __init__(self, data : pd.DataFrame, dims : int, target_id : str, frac : int = 1, scree_plot : bool = True) -> None:

        # Save target id of data
        self.target_id = target_id
        
        # Get number features
        g = data.shape[1] - 1

        # Optional : downsample data set
        if frac < 1:
            data = data.sample(frac=frac)

        # Calculate between-class covariance
        self.Sb = np.zeros((g,g))
        mu = data.drop(target_id, axis=1).mean()
        for k in data[target_id].unique():
            inner = data[data[target_id] == k].drop(target_id, axis=1).mean() - mu
            self.Sb += np.outer(inner,inner)

        # Calculate within-class covariance
        self.Sw = np.zeros((g,g))
        for k in data[target_id].unique():
            data_k = data[data[target_id] == k]
            muk = data_k.drop(target_id, axis=1).mean()
            inner = data_k.drop(target_id, axis=1) - muk            
            self.Sw += np.dot(inner.T,inner)

        # Calculate omega
        self.omega = np.dot( np.linalg.pinv( self.Sw ), self.Sb)

        # Get eigen vectors
        self.eig_val , self.eig_vec = np.linalg.eig(self.omega)
        
        # Ensure eigen values are sorted
        idx = np.flip(np.argsort(np.abs(self.eig_val)))
        self.eig_val = self.eig_val[idx]
        self.eig_vec = self.eig_vec[:,idx]

        # Save transformation matrix
        self.W = np.real(self.eig_vec[:,0:dims])

        # Plot largest eigen values
        if scree_plot:
            plt.figure(figsize=(5,5))
            if len(self.eig_vec) > 20:
                plt.bar([x for x in range(1,20+1)],np.real(self.eig_val[0:20]))
            else:
                plt.bar([x for x in range(1,len(self.eig_vec)+1)],np.real(self.eig_val[0:len(self.eig_vec)]))

        s = sum(np.real(self.eig_val))
        print(f"Preserving {round(sum(np.real(self.eig_val[0:dims]))/s*100,2)}% of variance",)

    # Redefine the number of dimensions
    def set_dims(self,dims:int):
        self.W = self.eig_vec[:,0:dims].astype('float')
       
    # Transform a data set using the projection matrix
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

# Reduce dimensionality using Fisher's LDA
class reducer:

    # Initialize the LDA by using a dataset
    def __init__(self, X : 'np.ndarray', y : 'np.ndarray', dims : int, scree_plot : bool = False) -> None:
        
        # Get number features
        features = X.shape[1]

        if dims > features - 1:
            raise ValueError("Number of dimensions is too great.")

        # Calculate between-class covariance
        self.Sb = np.zeros((features,features))
        mu = X.mean(axis=0)
        for k in y.unique():
            inner = X[y == k].mean(axis=0) - mu
            self.Sb += np.outer(inner,inner)

        # Calculate within-class covariance
        self.Sw = np.zeros((features,features))
        for k in y.unique():
            data_k = X[y == k]
            inner = data_k - data_k.mean(axis=0)
            self.Sw += np.dot(inner.T,inner)

        # Calculate omega
        self.omega = np.dot( np.linalg.pinv( self.Sw ), self.Sb)

        # Get eigen vectors
        self.eig_val , self.eig_vec = np.linalg.eig(self.omega)
        
        # Ensure eigen values are sorted
        idx = np.flip(np.argsort(np.abs(self.eig_val)))
        self.eig_val = self.eig_val[idx]
        self.eig_vec = self.eig_vec[:,idx]

        # Save transformation matrix
        self.W = np.real(self.eig_vec[:,0:dims])

        # Plot largest eigen values
        if scree_plot:
            plt.figure(figsize=(5,5))
            if len(self.eig_vec) > 20:
                plt.bar([x for x in range(1,20+1)],np.real(self.eig_val[0:20]))
            else:
                plt.bar([x for x in range(1,len(self.eig_vec)+1)],np.real(self.eig_val[0:len(self.eig_vec)]))

        s = sum(np.real(self.eig_val))
        print(f"Preserving {round(sum(np.real(self.eig_val[0:dims]))/s*100,2)}% of variance",)

    # Redefine the number of dimensions
    def set_dims(self, dims:int):
        self.W = self.eig_vec[:,0:dims].astype('float')
       
    # Transform a data set using the projection matrix
    def transform(self, X : 'np.ndarray'):
      
        
        # Transform and return data
        return X.astype('float').dot(self.W)

class classifier:

    # Initialize the LDA by using a dataset
    def __init__(self, X : 'pd.ndarray', y : 'pd.ndarray', p: 'pd.ndarray' = None) -> None:

        """Linear Discriminant Classification constructor

        Based on pdf-page 128 from :
        https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf

        """

        # Enforce data types
        X = np.array(X)
        y = np.ravel(y)
        p = np.ravel(p)

        self.classes = np.unique(y)

        # Extract mean, covariance and priors from data
        self.M = np.ascontiguousarray( [np.mean(X[y == c],axis=0) for c in self.classes], dtype = float )
        self.S = np.ascontiguousarray( [np.cov(X[y == c],rowvar=False) for c in self.classes], dtype = float )
        self.P = np.ascontiguousarray( [len(X[y == c])/len(y) for c in self.classes], dtype = float ) if p == None else p 

        # Calculate within-class covariance of dimensionality reduced data
        self.Sw = np.zeros((X.shape[1],X.shape[1]))
        for c in np.unique(y):
            data_k = X[y == c]
            muk = data_k.mean()  
            inner = data_k - muk      
            self.Sw += np.dot(inner.T,inner)

    def predict(self,X):

            # Get 'average' covariance
            S = sum(self.S)

            X = np.array(X)
            return self.classes[np.argmax([ lda_discriminant_function(X,S,m,p) for m,p in zip(self.M,self.P) ], axis=0)]


def lda_discriminant_function(X,s,m,p):
    part1 = np.dot(X,np.dot(np.linalg.inv(s),m))
    part2 = -0.5*np.dot(m,np.dot(np.linalg.inv(s),m))
    return part1 + part2 + np.log(p)