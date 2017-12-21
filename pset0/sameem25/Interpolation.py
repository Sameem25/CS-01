
# coding: utf-8

# ## CS01 Project (Data Science)

#importing Python libraries required for Project CS01
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#user input values for 'Year' and the corresponding 'Profit'
year_arr = []
profit_arr = []
user = (input('Enter the number of data to be presented: '))
for i in range(int(user)):
    year = float(input('Enter the year: '))
    year_arr.append(year)
    profit = float(input("Enter the profit for the year: "))
    profit_arr.append(profit)


#converting them to numpy arrays as column matrix
year = np.array(year_arr).reshape(int(user),1)
profit = np.array(profit_arr).reshape(int(user),1)



#This part has been designed to work on train-set using ML algorithm.
length = len(year)
for i in range(length):
    b = np.insert(year, 0, 1, axis = 1)
c = b[:,1]**2
c = np.insert(b,2,c, axis = 1)



#defining random values for theta
theta = np.array([[1],[1],[1]], dtype = float)
#defining a hypothesis function
hypo = np.dot(c,theta)



#Calculaing the cost function
error = np.subtract(hypo,profit)
errorsquare = np.sum(error**2)
cost = (1/(2*len(hypo)))*errorsquare


# Applying Normal Equation:
theta = np.dot(np.dot((np.linalg.inv(np.dot(np.transpose(c),c))),np.transpose(c)),profit)
#This is the value of theta we obtained using Machine Learning algorithm.



#Predicting test-set data using Machine Learning Algorithm
val = []
user = int(input('Enter the year: '))
val.append(user)
val.append(user**2)
user = np.array(val)
user = np.insert(user,0,1, axis = 0)
predict = np.dot(user,theta)

print ("The expected profit in year", val[0], "is expected to be: ", predict[0])

