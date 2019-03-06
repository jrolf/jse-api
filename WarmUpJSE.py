
import pandas as pd
import numpy  as np
from copy import *
from scipy.optimize import curve_fit
from sklearn.metrics import *

from collections import defaultdict as defd 

import warnings
warnings.filterwarnings("ignore")

#################################################################
################################################################# 


class PolyFit:
    def __init__(self,poly=[2,3,4,5]):
        if 'int' in str(type(poly)): poly=[poly]
        self.poly = poly
        self.models = {}
        
    def fit(self,x_train,y_train,poly=[]):
        if poly: 
            if 'int' in str(type(poly)): poly=[poly]
            self.poly = poly
        x = np.array(x_train)
        y = np.array(y_train)
        if x.shape == (len(x),1): x = x.reshape([len(x),]) 
        if y.shape == (len(y),1): x = x.reshape([len(y),])  
        results = []
        for deg in self.poly:
            params = np.polyfit(x,y,deg)
            self.models[deg] = params    

    def predict(self,x_test): 
        x = np.array(x_test) 
        if x.shape == (len(x),1): x = x.reshape([len(x),])
        results = []
        for deg in self.poly:
            params = self.models[deg] 
            preds = np.polyval(params,x)
            results.append(preds)
        M = np.array(results)
        preds_final = M.mean(0) 
        return preds_final

class ExponentialModel:
    def __init__(self,deg=3,log_x=True,log_y=False,buf=1):
        self.deg = deg
        self.models = []
        self.log_x = log_x
        self.log_y = log_y  
        self.buf = buf
    
    def fit(self,x,y):
        x2,y2 = np.array(x),np.array(y)
        if self.log_x: x2 = np.log(x2+self.buf)
        if self.log_y: y2 = np.log(y2+self.buf)  
        model = PolyFit([self.deg])
        model.fit(x2,y2)
        self.models.append(model)
        
    def predict(self,x):
        x2 = np.array(x) 
        if self.log_x: x2 = np.log(x2+self.buf)  
        model = self.models[-1]
        y2 = model.predict(x2)
        y = y2
        if self.log_y: y = np.exp(y)-self.buf
        return np.around(y,2)  


class DecayModel:
    def __init__(self,start_params=(176,0.35,31.2,5.96,10.6)):
        self.fit_params = start_params
    
    def func(self,x,a,b,c,d,e):
        y = ((a/((x**b)+c))**d)+e
        return y 
    
    def fit(self,x,y):
        x2,y2 = np.array(x),np.array(y) 
        fit_params,pcov = curve_fit(self.func,x2,y2,p0=self.fit_params,maxfev=5000)
        self.fit_params = fit_params 
        
    def predict(self,x):
        x2 = np.array(x) 
        y2 = self.func(x2,*self.fit_params) 
        return np.clip(np.around(y2,2),0.0,9**10) 

#################################################################
################################################################# 

fn = 'DATA/JS_PULLED2.csv'
PULL = pd.read_csv(fn) 

ZeroRank = defd(lambda:999999) 
CATS = list(PULL.columns)[1:]
Models = {}
for cat in CATS:
    x,y = PULL['RANK'],PULL[cat]  
    model = DecayModel()
    try:
        model.fit(x,y) 
        Models[cat] = model 
    except:

        try:
            model = ExponentialModel()
            model.fit(x,y) 
            Models[cat] = model
            #print('Log-Fit:',cat)

            x0 = list(range(999999))
            y0 = list(np.clip(model.predict(x0),0,10*12).astype('int')) 
            try:    zero_rank = y0.index(0) 
            except: zero_rank = 999999
            ZeroRank[cat] = zero_rank  

        except:
            #print('Non-Fit:',cat)
            pass 


#################################################################
################################################################# 

# Returns a list of valid category names: 
def GetCategories():
    return sorted(Models)
    
# Individual Prediction for Rank->QSold 
# Batched Predictions for RankList->QSoldList  
def JungleScoutPredict(category,rank):
    if 'list' in str(type(category)): CAT_LIST = category
    else:                             CAT_LIST = [category]
    if 'list' in str(type(rank)):    RANK_LIST = rank
    else:                            RANK_LIST = [rank] 
    if   len(CAT_LIST) < len(RANK_LIST):
        CAT_LIST  = [CAT_LIST[-1] for _ in range(len(RANK_LIST))]
    elif len(CAT_LIST) > len(RANK_LIST):
        RANK_LIST = [RANK_LIST[-1] for _ in range(len(CAT_LIST))]
    results = []
    for cat,rank in zip(CAT_LIST,RANK_LIST):
        zero_rank = ZeroRank[cat]
        if rank>=zero_rank:
            results.append(0.0)
            continue
        qsold = Models[cat].predict([rank])[0]
        results.append(qsold)
    if len(results)>1:
        return results
    else:
        return results[0] 
        
#################################################################
################################################################# 

# [END]


