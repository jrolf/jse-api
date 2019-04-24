import requests
from ApiCreds import * 

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

##############################################################

### TESTING THE API WITH FUNC ### 
NumList2 = [8,3,9,1,3.14159]
print() 
print('Testing API with PingAPI()...') 
print(API_URL)
print('Sending Payload For Echo:')
print(NumList2)
print() 
J2 = PingAPI('echo',NumList2)
print('Recieved response:')
print(J2)
print() 
print('Also Checking the Time!')
api_time = PingAPI('time_now')
print(api_time)
print()
print('Finally, we are estimating how many sales will occur in a')
print('month if a product in "Home & Garden" is ranked @5,000th:') 
qsold = PingAPI('JS_Predict',['Home & Garden',5000])
print('We expect',qsold,'units to be sold.')
print()
print('Go here to test it for yourself!!')
print("https://www.junglescout.com/estimator/")
print() 

##############################################################

# [END]  











