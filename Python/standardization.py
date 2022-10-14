import pandas as pd

class standardization:
    
    def __init__(self, data : pd.DataFrame, target = None) -> None:
        self.target = target
        if target:
            self.mean = data.drop(target,axis=1).mean()
            self.std = data.drop(target,axis=1).std()
        else:
            self.mean = data.mean()
            self.std = data.std()
        
    def transform(self, data : pd.DataFrame):
        if self.target:
            targets = data.get(self.target)
            return ((data.drop(self.target,axis=1)-self.mean)/self.std).assign(target = targets)
        else:
            return (data-self.mean)/self.std
