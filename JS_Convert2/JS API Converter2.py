
#p = "c:\\users\\"
#import sys
#if p not in sys.path: sys.path = [p]+sys.path

import requests
from ApiCreds import *

#import numpy as np
#import pandas as pd

import json,os,codecs,csv,math
from collections import defaultdict as defd 


##############################################################
##############################################################

### FUNC FOR API TESTING ### 

def PingAPI(fname,data=[]): 
    global response
    Json = {
            "key"  :API_KEY,
            "fname":fname,
            "data" :data
            }  
    response = requests.post(API_URL,headers=HEADERS,json=Json)
    J = response.json() 
    return J


# Create a path if it doesn't exist:
def EnsurePath(path):
    if not os.path.exists(path):
        os.makedirs(path) 

# https://docs.python.org/2/library/json.html   
# https://docs.python.org/2/library/json.html#py-to-json-table  
# http://www.pythonforbeginners.com/files/reading-and-writing-files-in-python
def ReadJSON(filename): 
    File = open(filename,'r')
    return json.load(File) 
    
def WriteJSON(filename,obj,pretty=True):  
    path = GetPath(filename)  
    if path: EnsurePath(path) 
    File = open(filename,'w')
    if pretty: json.dump(obj,File,sort_keys=True,indent=2,separators=(',',': ')) 
    else:      json.dump(obj,File,sort_keys=True)
    
# 'Read()' takes all the lines of a text file,
# converts them to strings, and returns LoL of these strings.
def Read(FILE):
    rawfilelist = open(FILE,'rt').readlines()
    filelist = [i[:-1] for i in rawfilelist]
    return filelist

# 'Write()' overwrites a whole file with a list.
def Write(FILE,LIST):
    path = GetPath(FILE)
    if path: EnsurePath(path) 
    List = [str(i)+'\n' for i in LIST]
    open(FILE,'wt').writelines(List) 
    
def readcsv(filepath):
    data = []
    OF = codecs.open(filepath,'r', 'cp1252')
    for row in csv.reader(OF, dialect='excel', skipinitialspace=True):
        data.append(row)
    return data

def writecsv(filepath,data):
    WF = codecs.open(filepath,'w', 'cp1252')
    writer = csv.writer(WF, dialect='excel', skipinitialspace=True)
    for row in data:
        writer.writerow(row) 

##############################################################

fn = 'Data/Raw Data.csv'

print()
print('Reading Input From File:',fn)

rows = readcsv(fn)
cols = rows[0] 
rows = rows[1:] 

CatList = [r[0] for r in rows]
RnkList = [float(r[1]) for r in rows]

print('Sending data to API...') 

QSoldList = PingAPI('JS_Predict',[CatList,RnkList]) 
results = [r+[q] for r,q in zip(rows,QSoldList)]
results = [cols+['SalesPerMonth']]+results
print('Recieved response - Boy, that was quick!')

fn2 = 'Data/Results.csv'
print('Writing Output To File: ',fn2)
writecsv(fn2,results)

print()
print('Program Complete!')
print('Have an awesome day!!')
print()

##############################################################





    



