
import pandas as pd
import numpy  as np
from copy import *
from scipy.optimize import curve_fit
from sklearn.metrics import *

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

        except:
            #print('Non-Fit:',cat)
            pass 

#################################################################
################################################################# 

# Individual Prediction for Rank->QSold 
def JungleScoutPredict(category,rank):
    model = Models[category]
    return model.predict([rank])[0] 

# Batched Predictions for RankList->QSoldList  
def JungleScoutBatch(category,rank_list):
    model = Models[category]
    return list(model.predict(rank_list)) 

def GetCategories():
	return sorted(Models) 

#################################################################
################################################################# 


#qsold = 

Notes = '''

>>> category = 'Home & Garden'
>>> qsold = JungleScoutPredict(category,1000)
>>> qsold
332.36

>>> category = 'Home & Garden'
>>> ranks = [10,100,1000,10000]
>>> qsold_list = JungleScoutBatch(category,ranks)
[1456.19, 1004.18, 332.36, 47.06]

>>> cats = GetCategories()
>>> for cat in cats: print(cat)

Appliances
Arts, Crafts & Sewing
Automotive
Baby
Cell Phones & Accessories
Clothing & Accessories
Electronics
Health & Personal Care
Home & Garden
Home & Kitchen
Home Improvement
Industrial & Scientific
Musical Instruments
Office Product
Pet Supplies
Sports & Outdoors
Toys & Games


'''




