import pandas as pd
import numpy as np

class classifier:

    # Initialize the QDA by using a dataset
    def __init__(self, X : 'pd.ndarray' = None, y : 'pd.ndarray' = None,**kwargs) -> None:

        """Linear Discriminant Classification constructor

        Based on pdf-page 128 from :
        https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf

        """

        if X and y:
            self.fit(X,y)

    def fit(self, X : 'pd.ndarray' = None, y : 'pd.ndarray' = None, p: 'pd.ndarray' = None):
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

    def get_params(self,deep : bool = False) : return {'priors': None, 'reg_param': 0.0, 'store_covariance': False, 'tol': 0.0001}

    def score(self,X,y):
        return sum(y == self.predict(X))/len(y)

    def predict(self,X):

            X = np.array(X)
            return self.classes[np.argmax([ qda_discriminant_function(X,np.matrix(s),m,p) for m,s,p in zip(self.M,self.S,self.P) ], axis=0)]

def qda_discriminant_function(X,s,m,p):
    part1 = -0.5*np.log(np.linalg.det(s))
    part2 = -0.5*np.einsum("ji,ij->j",(X-m),np.dot(np.linalg.pinv(s),(X-m).T))
    return part1 + part2 + np.log(p) 