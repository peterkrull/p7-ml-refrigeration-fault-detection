import pandas as pd
import numpy as np

# Transform entire data frame
class PCA_reducer:

    def __init__(self, data : pd.DataFrame, dims : int, target_id : str = None) -> None:

        # Extract labels and drop
        if target_id:
            labels = data[target_id]
            data = data.drop(target_id, axis=1)
        
        # Calculate covariance matrix
        cov = data.cov()

        # Get eigen vectors
        eig_val , eig_vec = np.linalg.eig(cov)

        self.W = eig_vec[:,0:dims]
        self.mean = data.mean(axis=0)
        self.target_id = target_id

        s = sum(np.real(eig_val))
        print(f"Preserving {round(sum(np.real(eig_val[0:dims]))/s*100,2)}% of eigen value transformations",)

    def transform(self, data : pd.DataFrame, target_id : str = None):

        # Determine appropriate label definition
        if self.target_id in data: target_id = self.target_id
        
        # Extract labels and drop
        if target_id:
            labels = data[target_id]
            data = data.drop(target_id, axis=1)

        # Do transform
        sub = data.sub(self.mean).transpose()
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