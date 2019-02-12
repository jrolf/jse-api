# funcs.py

####################################################
####################################################

### !!!!! OMG !!!!!!
### Primary Finding:
###     For some reason, storing files deeper than 
###     one folder causes an error where the filename
###     cannot be found.
### Example:
###     BAD:   fn = 'DATA/MINDS/BASE_DATA/Words.csv'
###     GOOD:  fn = 'DATA/Words.csv'

####################################################
####################################################

from ApiCreds  import *
from WarmUpJSE import *

import sys, codecs, csv 
from random import shuffle
from datetime import datetime,timedelta 
import pandas as pd 

####################################################
####################################################

GoodFuncs = [
    'echo', 
    'time_now',  
    
    'JS_Predict',
    'JS_Batch',
    'JS_GetCats', 
]

####################################################
####################################################  

def master(key='',fname='',data={}):
    if str(key) != API_KEY: 
        return {"message":"invalid key"}
    if fname not in GoodFuncs: 
        return {"message":"unapproved func"}
    
    try:    f = eval(fname)
    except: return {"message":"eval failed"}

    try:    return f(data)
    except: return {"message":"processing failed"}

####################################################
####################################################

# Returns the Input:
def echo(x):
    return x 

def get_time(dif_hrs=0):
    dt = datetime.now()
    if dif_hrs!=0:
        dt = dt+timedelta(hours=dif_hrs)
    return dt  

def GetCST():
    return get_time(-6)

def time_now(x): 
    stamp = str(GetCST()) 
    return {'Time in CST':stamp}  

####################################################
####################################################

def JS_Predict(x):
    cat,rank = tuple(x)
    return JungleScoutPredict(cat,rank) 

def JS_Batch(x):
    cat,ranks = tuple(x)
    return JungleScoutBatch(cat,ranks)  

def JS_GetCats(x):
    return GetCategories()

####################################################
####################################################

# [END]





