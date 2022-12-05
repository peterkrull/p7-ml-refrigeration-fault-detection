import pandas as pd
import numpy as np

class dataloader:

    def __init__(self,path, target_col) -> None:
        self.__path = path
        self.__target_col = target_col

        # Load data from csv file
        if type(target_col) == type(path) == str:
            data = pd.read_csv(path)
            self.y = data.pop(target_col)
            self.X = data

        # If passed data is already data frames
        elif type(path) == type(pd.DataFrame()):
            self.y = target_col
            self.X = path

        else: raise(TypeError(f"Must pass path or DataFrame, got : {type(path)}"))

    def sample(self,**kwargs):
        
        """Sample the data collection either as a fraction, eg. `frac = 0.2` or as a specific number, eg. `n = 100`."""
        
        
        data = pd.concat([self.X,self.y],axis='columns').sample(**kwargs)
        y = data.pop('target')
        X = data
        return dataloader(X,y)

    def total(self):
        
        """Sample the data collection either as a fraction, eg. `frac = 0.2` or as a specific number, eg. `n = 100`."""
        return pd.concat([self.X,self.y],axis='columns')


    def get(self,args):

        """Get subset of data collection. Recommended to use a dictionary.
        `data_dubset = data.get({'target':[0,1,2],'setpoint' = 10 })`
        """

        data = pd.concat([self.X,self.y],axis='columns')

        if type(args) == dict:
            subdata = data

            for each in args:
                if type(args[each]) == list:
                    df = None
                    for each2 in args[each]:
                        df = pd.concat([df,subdata[subdata[each] == each2]])
                    subdata = df
                else:
                    subdata = subdata[subdata[each] == args[each]]

            y = subdata.pop('target')
            X = subdata
            return dataloader(X,y)

        elif type(args) == list or type(args) == type(np.array([])) or type(args) == type(pd.Series()):
            y = data.pop('target')
            X = data
            X = X[args]
            return dataloader(X,y)

        else:
            data = data[args]
            y = data.pop('target')
            X = data
            return dataloader(X,y)

    def apply(self,fn):
        cols = self.X.columns
        X_applied = fn(self.X)
        if X_applied.shape == self.X.shape:
            return dataloader(pd.DataFrame(X_applied, columns = cols),self.y)
        else:
            return dataloader(pd.DataFrame(X_applied),self.y)

from sklearn.feature_selection import SequentialFeatureSelector
def feature_selection(clf,trn : dataloader ,vld : dataloader):

    summary = pd.DataFrame({'n_features':[],'score':[], 'features' : [], 'direction' : []})
    for n_features in range( 2, np.shape(trn.X.columns)[0] ):
        for direction in ["forward","backward"]:
            
            bw = SequentialFeatureSelector(clf, direction=direction, n_jobs=-1, n_features_to_select=n_features)
            bw.fit(trn.X, trn.y)

            features = trn.X.columns[bw.get_support()]

            score_clf = clf
            score_clf.fit(trn.get(features.tolist()).X,trn.y)

            score = score_clf.score(vld.get(features.tolist()).X,vld.y)
            summary = pd.concat([summary, pd.DataFrame({'n_features':n_features,'score':score,'features': features, 'direction': direction})], )
    
    return summary