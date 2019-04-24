

import pandas as pd
import numpy  as np
from copy import *
from bisect import * 
from scipy.optimize import curve_fit
from sklearn.metrics import *
from collections import defaultdict as defd 
import datetime,pickle 

from DemandHelper import * 

import warnings
warnings.filterwarnings("ignore")

#################################################################
################################################################# 
#################################################################

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

#################################################################
################################################################# 



# Export a fitted model to text file:
# These filenames normally end in '.pkl'
def ExportModel(filename,model_object):
    pickle.dump(model_object, open(filename, 'wb'))
    print('Model Saved TO: '+filename)

# Import a fitted model from text file:
# These filenames normally end in '.pkl'
def ImportModel(filename):
    model_object = pickle.load(open(filename, 'rb'))
    print('Model Imported FROM: '+filename)
    return model_object 

def GetToday():
	today = datetime.datetime.today()
	return str(today)[:10]




#################################################################
################################################################# 
#################################################################

short2long = {
	'H&G'    : 'Home & Garden',
	'L&G'    : 'Lawn & Garden',
	'SPORTS' : 'Sports & Outdoors',
	'HI'     : 'Home Improvement',
	'TOY'    : 'Toys & Games',
	'KIT'    : 'Home & Kitchen', 
} 

long2short = {}
for short in sorted(short2long):
	long2short[short2long[short]] = short 
Shorts = sorted(short2long)
Longs  = sorted(long2short) 

def ConvertToShort(thing):
	if thing in long2short: return long2short[thing]
	return thing

Models2 = {}
for SH in Shorts: 
	fn = 'MODELS/'+SH+'/DFM2.pkl' 
	model =  ImportModel(fn) 
	Models2[SH] = model 

AllDates = sorted(set([str(a)[:10] for a in Models2['H&G'].alldates]))  

#################################################################
################################################################# 

# Returns a list of valid category names: 
def GetCategories2():
    return sorted(long2short) 


# SPREETAIL DEMAND PREDICTION:  
# cat      : Category (String or List)
# rank     : Sales Rank (Integer, 2-List, Long-List)
# date1    : First Date of Forecast ("2018-09-03")
# date2    : Final Date of Forecast OR # Days Forward ("2018-10-03" or 30) 
# bb_ratio : BuyBox Percent (100.0)
# md_ratio : Marketplace Distribution Percent
def SpreetailPredict(cat,rank,date1='today',date2=30,bb_ratio=1.0,md_ratio=0.62):
	if (not date1) or (str(date1).lower()=='today'): date1 = GetToday() 
	index1 = bisect_left(AllDates,date1)
	if len(str(date2)) >10: date2 = str(date2)[:10]
	if len(str(date2))==10: index2 = bisect_left(AllDates,date2)
	else: index2 = index1+int(date2) 
	index_dif = abs(index2-index1)
	index1 = min([index1,index2]) 
	index2 = index1+index_dif
	DateRange = AllDates[index1:index2+1] 
	LEN = len(DateRange) 
	#-------------------------------------- 
	tdf = pd.DataFrame() 
	tdf['DATE'] = DateRange 
	#-------------------------------------- 
	if 'list' in str(type(cat)):
		cat = [ConvertToShort(a) for a in cat]  
		if len(cat)==LEN: tdf['CAT'] = cat 
		else: tdf['CAT'] = cat[0] 
	else: tdf['CAT'] = ConvertToShort(cat) 
	#-------------------------------------- 
	if 'list' in str(type(rank)):
		if len(rank)==LEN: tdf['RANK'] = rank  
		elif len(rank)==2:
			r1,r2 = tuple(rank)
			tdf['RANK'] = np.linspace(r1,r2,LEN) 
		else: tdf['RANK'] = rank[0]  
	else: tdf['RANK'] = rank 
	#-------------------------------------- 
	md_ratio2 = max(0.3,min(md_ratio,0.99))
	other_ratio = (1.0-md_ratio2)/md_ratio2
	tdf['BBR'] = bb_ratio
	tdf['MDR'] = md_ratio2  
	#-------------------------------------- 

	M = tdf.values
	results = []
	for row in M:
		d,c,r = tuple(row[:3])  
		pred_100 = Models2[c].predict(r,d,100.0) 
		pred_bbr = Models2[c].predict(r,d,100.0*bb_ratio)  
		results.append([pred_100,pred_bbr]) 

	tdf['P_100']    = [r[0][1] for r in results]
	tdf['P_100_HI'] = [r[0][2] for r in results]
	tdf['P_100_LO'] = [r[0][0] for r in results]

	tdf['P_BBR']    = [r[1][1] for r in results]
	tdf['P_BBR_HI'] = [r[1][2] for r in results]
	tdf['P_BBR_LO'] = [r[1][0] for r in results]

	tdf['P_OTH']    = other_ratio * tdf['P_100']
	tdf['P_OTH_HI'] = other_ratio * tdf['P_100_HI']
	tdf['P_OTH_LO'] = other_ratio * tdf['P_100_LO'] 

	tdf['P_TOT']    = tdf['P_BBR']   +tdf['P_OTH']
	tdf['P_TOT_HI'] = tdf['P_BBR_HI']+tdf['P_OTH_HI'] 
	tdf['P_TOT_LO'] = tdf['P_BBR_LO']+tdf['P_OTH_LO'] 

	cols = list(tdf.columns)[5:]
	for col in cols:
		col2 = col+'_C'
		tdf[col2] = np.cumsum(tdf[col])

	Matrix = [list(tdf.columns)]
	for row in tdf.values:
		Matrix.append(list(row)) 
	MainPred = list(tdf['P_TOT_C'])[-1] 
	return [MainPred,Matrix] 

def SpreePred(cat,rank,date1='today',date2=30,bb_ratio=1.0,md_ratio=0.62):
	result = SpreetailPredict(cat,rank,date1,date2,bb_ratio,md_ratio)
	M = result[1]
	cols,m = M[0],M[1:] 
	return pd.DataFrame(m,columns=cols)


#################################################################
################################################################# 

# [END]

















































