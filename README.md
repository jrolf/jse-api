
# README.md

```
##########################

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



```







