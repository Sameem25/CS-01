
# coding: utf-8

# ## CS01 Project (Data Science)

# In[34]:


#importing Python libraries required for Project CS01

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')



#user input values for 'Year' and the corresponding 'Profit'

year_arr = []
profit_arr = []
user= (input('Enter the number of data to be presented: '))
for i in range(int(user)):
    year= input('Enter the year: ')
    year=float(year)
    year_arr.append(year)
    profit= input("Enter the profit for the year: ")
    profit= float(profit)
    profit_arr.append(profit)


#converting them to numpy arrays as column matrix

year= np.array(year_arr).reshape(int(user),1)
profit= np.array(profit_arr).reshape(int(user),1)
print(year)
print(profit)


#This part has been designed to work on train-set using ML algorithm.

length= len(year)
for i in range(length):
    b=np.insert(year, 0, 1, axis=1)
c=b[:,1]**2
d=b[:,1]**3
c= np.insert(b,2,c, axis=1)
print (c)


#defining random values for theta
theta= np.array([[1],[1],[1]], dtype=float)

#defining a hypothesis function
hypo=np.dot(c,theta)
print(hypo)


#plotting the hypothesis function with the output data

plt.scatter(year,profit, color='red')
plt.plot(year,hypo)
plt.xlabel('year')
plt.ylabel('profit')
plt.title('convexing of hypothesis function using random theta')


#Calculaing the cost function

error= np.subtract(hypo,profit)
print (error)
errorsquare=np.sum(error**2)
cost=(1/(2*len(hypo)))*errorsquare
print(cost)


# Applying Normal Equation:

theta= np.dot(np.dot((np.linalg.inv(np.dot(np.transpose(c),c))),np.transpose(c)),profit)
print (theta)
#This is the value of theta we obtained using Machine Learning algorithm.


#Applying the values of theta obtained from Normal Equation.
#We, here, redraw the hypothesis function using the theta obtained from Normal Equation.

hypo=np.dot(c,theta)
plt.scatter(year,profit, color='red')
plt.plot(year,hypo)
plt.xlabel('year')
plt.ylabel('profit')
plt.title('Graph of Hypothesis function') 


#recalculating cost function

hypo=np.dot(c,theta)
error= np.subtract(hypo,profit)
errorsquare=np.sum(error**2)
cost=(1/(2*len(hypo)))*errorsquare
print(cost)


#Predicting test-set data using Machine Learning Algorithm
val=[]
user=(input('Enter the year: '))
user= int(user)
val.append(user)
val.append(user**2)
user= np.array(val)
user = np.insert(user,0,1, axis=0)
predict= np.dot(user,theta)

print ("The expected profit in year", val[0], "is expected to be: ", predict[0])

