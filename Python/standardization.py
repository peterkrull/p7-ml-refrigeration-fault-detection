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
        data2 = data.copy()
        if self.target:
            targets = data2.get(self.target)
            for i in data2.drop(self.target, axis = 1):
                if self.std[i] != 0:
                    data2[i] = (data2[i]-self.mean[i])/self.std[i]
                else:
                    data2[i] = data2[i]-self.mean[i]

            return data2
        else:
            for i in data2:
                if self.std[i] != 0:
                    data2[i] = (data2[i]-self.mean[i])/self.std[i]
                else:
                    data2[i] = data2[i]-self.mean[i]

            return data
