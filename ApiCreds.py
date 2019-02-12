
####################################################
####################################################

API_KEY  = "z3MzW5Si6oC4XveH8N4"
APP_NAME = "SuperAwesomeAPI"

BASE_URL = "https://"+APP_NAME+".herokuapp.com/"
FUNC     = 'master'
NumList  = [8,2,9,11,3.14159] # Temporary Object for testing.

API_URL =  BASE_URL+FUNC  
HEADERS = {'Content-Type': 'application/json'}
PAYLOAD = {"key":API_KEY,"fname":"echo","data":NumList}  

####################################################
####################################################

#import requests
#response = requests.post(API_URL,headers=HEADERS,json=PAYLOAD) 
#J = response.json()





