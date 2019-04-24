
import numpy  as np
import pandas as pd 

######################################################################
######################################################################

bb_params = np.array([-0.25804822,  0.92910462,  0.11108338])  
bb_params = np.array([-0.24440845,  0.87918093,  0.10563643]) 
bbr1 = lambda bb: bb*0.9 
bbr2 = lambda bb: ((bb_params[0]*(bb**2)) + (bb_params[1]*bb) + bb_params[2])  
bbr3 = lambda bb: (bbr1(bb)*0.4)+(bbr2(bb)*0.6)  

# Convert Spreetail QSold to Global Amazon QSold Estimate:
# st_qsold : Spreetail Quantity Sold
# bb_pct   : BuyBox Percent (i.e. 78.0)
def local2global(st_qsold,bb_pct):  
    bb_ratio = bbr3(bb_pct/100.0) 
    return st_qsold/bb_ratio

# Convert Global Amazon QSold to Spreetail QSold Estimate:
# gl_qsold : Global Quantity Sold
# bb_pct   : BuyBox Percent (i.e. 78.0) 
def global2local(gl_qsold,bb_pct):
    bb_ratio = bbr3(bb_pct/100.0) 
    return gl_qsold*bb_ratio 

# Converts a local val to equivalent if BB==100%: 
def local2local(st_qsold,bb_pct):
    bb = min(max(25,bb_pct),100) 
    glo = local2global(st_qsold,bb)
    loc = global2local(glo,100) 
    return loc

######################################################################
######################################################################



def OlsFromPoints(xvals,yvals):
    LEN = len(xvals)
    xvals = np.array(xvals).reshape(LEN,1) 
    model = OLS()
    model.fit(xvals,yvals) 
    return model  

def GetOlsParams(ols_model):
    m = ols_model.coef_[0] 
    b = ols_model.intercept_
    return m,b 

# Given an OLS Model, Return an Inverse Model
def GenInverseOLS(normal_ols_model):
    m,b = GetOlsParams(normal_ols_model)
    inv_func = lambda y: (y-b)/float(m) 
    
    xvals = np.linspace(-100,100,1000)
    yvals = inv_func(xvals) 
    inv_ols_model = OlsFromPoints(xvals,yvals) 
    return inv_ols_model 

def AdaptiveError(preds,actual):
    y = actual
    raw_err = np.abs(preds-y)
    prop_er = []
    max_val = []
    for p,a in zip(preds,y):
        
        MaxVal = min(999,max(p,a))
        MinVal = max(0.001,min(p,a))
    
        if min(p,a)==0: er = 0
        else:
            try: er = (1.0-(MinVal/MaxVal)) 
            except: er = 0
        prop_er.append(er)
    prop_er = np.array(prop_er) 
    
    mean_vals = (preds+y)/2.0 
    ratios  = np.clip(y/mean_vals,0.001,999) 
    log_err  = np.abs(np.log(ratios))
    comp_er = (log_err**2)+(10*raw_err*prop_er)
    return comp_er  

def CompositeError(preds,actual):
    y = actual
    raw_err = np.abs(preds-y)
    prop_er = []
    max_val = []
    for p,a in zip(preds,y): 
        MaxVal = min(999,max(p,a))
        MinVal = max(0.001,min(p,a))
        if min(p,a)==0: er = 0
        else:
            try: er = (1.0-(MinVal/MaxVal)) 
            except: er = 0
        prop_er.append(er)
    prop_er = np.array(prop_er) 
    comp_er = raw_err*prop_er 
    return comp_er  

def RawError(preds,actual): 
    return np.abs(preds-actual) 

######################################################################
######################################################################


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
        return np.clip(y2,0.0,9**10)    
    
    
class DecayModel2:
    def __init__(self,start_params=(176,-1.05,31.2,1.785,10.6)): 
        self.fit_params = start_params
    
    def func(self,x,a,b,c,d,e):
        y = ((a/((x**np.exp(b))+c))**np.exp(d))+e
        return y 
    
    def fit(self,x,y):
        x2,y2 = np.array(x),np.array(y) 
        fit_params,pcov = curve_fit(self.func,x2,y2,p0=self.fit_params,maxfev=5000)
        self.fit_params = fit_params 
        
    def predict(self,x):
        x2 = np.array(x) 
        y2 = self.func(x2,*self.fit_params) 
        return np.clip(y2,0.0,9**10) 
    



######################################################################
######################################################################

#---------------------------------------
# Conservative Model: (Tops-Out @~17.5) 
base_params1 = [
    30.521333151265473,
    -1.127887081600148,
    21.184330932144285,
    1.830559238999896,
    0.0052741309085363175
]
#MODEL1 = DecayModel2()
#MODEL1.fit_params = base_params1

#---------------------------------------
# Agressive Model: (Tops-Out @~7.4) 
base_params2 = [
    20.109506094359123,
    -1.1598304689364167,
    11.397077178832497,
    1.7804697922991446,
    0.0031169700674444694
] 
#MODEL2 = DecayModel2()
#MODEL2.fit_params = base_params2 

######################################################################
######################################################################

# BASE Conversion Model1
base_params1_t = tuple(base_params1) 
def BASE_MODEL1(x):
    a,b,c,d,e = base_params1_t 
    y = (((a/((x**np.exp(b))+c))**np.exp(d))*2008.4)+e 
    return y 

# BASE Conversion Model2
base_params2_t = tuple(base_params2) 
def BASE_MODEL2(x):
    a,b,c,d,e = base_params2_t 
    y = (((a/((x**np.exp(b))+c))**np.exp(d))*2005.8)+e 
    return y 

# HYBRID Conversion Model:
def BASE_MODEL3(x,ratio=0.5): 
    y1 = BASE_MODEL1(x) 
    y2 = BASE_MODEL2(x)
    y3 = (y2*ratio)+(y1*(1.0-ratio)) 
    return y3 

######################################################################
######################################################################


class RankModel3:
    def __init__(self,start_params=[0.0,0.0,0.0],fitted=''):
        self.fit_params = start_params
        self.fac_y  = 0
        self.fitted = fitted 
        
    def estimate_fac_y(self,x,y):
        y_hat = BASE_MODEL3(np.array(x),0.5) 
        ratio = y.mean()/y_hat.mean() 
        self.fac_y = np.log(ratio)  
        
    def funcA(self,x,fac_x,fac_y,exp_y):
        y = (BASE_MODEL3(x*np.exp(fac_x),0.5)*np.exp(fac_y))**np.exp(exp_y)
        return y 
    
    def funcB(self,x,fac_x,fac_y,exp_y):
        y = (BASE_MODEL3(x*np.exp(fac_x),0.1)*np.exp(fac_y))**np.exp(exp_y)
        return y
    
    def funcC(self,x,fac_x,fac_y,exp_y):
        y = (BASE_MODEL3(x*np.exp(fac_x),0.9)*np.exp(fac_y))**np.exp(exp_y)
        return y 
    
    def fit(self,x,y):
        x2,y2 = np.array(x),np.array(y) 
        #bnds = tuple([np.array([-3,-3,-3]),np.array([3,3,3])]) 
        if self.fac_y==0: 
            self.estimate_fac_y(x,y)
            self.fit_params[1] = self.fac_y  # Inserting the estimate for the y factor.
        params = tuple(self.fit_params) 
        self.params = params
    
        if not self.fitted:
            try: 
                fit_params,pcov = curve_fit(self.funcA,x2,y2,p0=params,maxfev=3000)
                self.fitted = 'ModelA'
            except: pass
            
        if not self.fitted:
            try: 
                fit_params,pcov = curve_fit(self.funcB,x2,y2,p0=params,maxfev=3000)
                self.fitted = 'ModelB'
            except: pass

        if not self.fitted:
            try: 
                fit_params,pcov = curve_fit(self.funcC,x2,y2,p0=params,maxfev=3000)
                self.fitted = 'ModelC'
            except: pass
            
        self.fit_params = fit_params 
        
    def predict(self,x):
        x2 = np.array(x) 
        if   self.fitted=='ModelA': y2 = self.funcA(x2,*self.fit_params)
        elif self.fitted=='ModelB': y2 = self.funcB(x2,*self.fit_params)
        elif self.fitted=='ModelC': y2 = self.funcC(x2,*self.fit_params)
        return y2

######################################################################
######################################################################


class DemandForecastModel:
    def __init__(self,rank_model='',forecast='',rmodel_beta=1.0,final_beta=1.0):
        if rank_model != '': 
            self.ingest(rank_model,forecast,rmodel_beta,final_beta) 
    
    def ingest(self,rank_model,forecast,rmodel_beta=1.0,final_beta=1.0): 
        self.rank_model = rank_model
        self.rmodel_beta = rmodel_beta 
        self.forecast = forecast 
        self.final_beta = final_beta  
        self.alldates = sorted(forecast.index) 
    
    def predict(self,rank=10000,date='2018-07-04',buybox=100):
        if 'str' not in str(type(date)): date = str(date)[:10] 
        pred1 = self.rank_model.predict([rank])[0] 
        pred2 = pred1*self.rmodel_beta 
        d = self.forecast.loc[date] 
        mid,lo,hi = d['yhat'],d['yhat_lower'],d['yhat_upper']  
        rdr_preds = np.array([lo,mid,hi]) 
        pred3 = pred2*rdr_preds 
        pred4 = pred3*self.final_beta 
        pred5 = global2local(pred4,buybox)
        return pred5

######################################################################
######################################################################



















