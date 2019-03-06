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

def JS_GetCats(x=[]):
    return GetCategories()

def JS_Predict(x=[]):
    cat,rank = tuple(x)
    return JungleScoutPredict(cat,rank) 

####################################################
####################################################

Notes '''

USAGE: 

>>> cats = GetCategories()
>>> 
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


>>> X1 = ['Home Improvement',1234]
>>> JS_Predict(X1)
1238.26

>>> X1 = ['Home Improvement',[1234,23456,123456]]
>>> JS_Predict(X1)
[1238.26, 78.03, 0.0]

>>> X1 = [['Home Improvement','Home & Garden'],1000]
>>> JS_Predict(X1)
[1347.8, 332.36]

>>> X1 = [['Home Improvement','Home & Garden'],[1000,1000]]
>>> JS_Predict(X1)
[1347.8, 332.36]

>>> category = 'Home & Garden'
>>> ranks = [10,100,1000,10000] 
>>> JS_Predict([category,ranks])
[1456.19, 1004.18, 332.36, 47.06]


'''

####################################################
####################################################

# [END]





