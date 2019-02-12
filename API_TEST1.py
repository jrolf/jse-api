import requests
from ApiCreds import * 

##############################################################

### GLOBAL PRE-REQS: ###

#AppName = 'magiclogic-dev'
API_KEY = "Y8JsJ7z3XvSi6oC4eH8N4MzW5"
Base = 'https://'+AppName+'.herokuapp.com/'
func = 'master' 
api_url  = Base+func
Headers = {'Content-Type': 'application/json'}

##############################################################

### LOCAL NON-FUNC TESTING ###

## Test data for echo:
#NumList = [8,3,9,1]

## Loading the Json Payload:
#Json = {"key":API_KEY,
#        "fname":"echo",
#        "data":NumList}  

## Executing the API Request:
#response = requests.post(api_url,headers=Headers,json=Json) 
#J1 = response.json()

##############################################################

### FUNC FOR API TESTING ### 

def PingAPI(fname,data):
    global response
    Json = {
            "key"  :API_KEY,
            "fname":fname,  #"rev_analysis",
            "data" :data
            }  
    response = requests.post(API_URL,headers=HEADERS,json=Json)
    J = response.json() 
    return J

##############################################################

### TESTING THE API WITH FUNC ### 
print('Testing API with PingAPI()...') 
print(api_url)
NumList2 = [8,3,9,1,3.14159] 
J2 = PingAPI('echo',NumList2)
print('Recieved response:')
print(J2) 

##############################################################













