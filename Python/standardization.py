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
            for i in data.drop(self.target, axis = 1):
                if self.std[i] != 0:
                    data[i] = (data[i]-self.mean[i])/self.std[i]
                else:
                    data[i] = data[i]-self.mean[i]

            return data
        else:
            for i in data:
                if self.std[i] != 0:
                    data[i] = (data[i]-self.mean[i])/self.std[i]
                else:
                    data[i] = data[i]-self.mean[i]

            return data
