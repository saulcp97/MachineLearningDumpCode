import matplotlib.pyplot as plt
from utils import *
import datetime
import numpy as np

#Future Project
#https://towardsdatascience.com/step-by-step-guide-building-a-prediction-model-in-python-ac441e8b9e8b


#https://api.coindesk.com/v1/bpi/historical/close.json?start=2012-01-28&end=2018-05-25

#Extract the data from the file by "date"
dates = extractV('date')

dayZero = (2012,1,27)
firstDay = "2012-01-28"
lastDay = (2018,5,25)

initialCost = 5.292 #Date 2012-01-27.
prices = extractBitcoinDict()

def previousDate(Name):
    #Convert the date to a datetime object
    dateF = datetime.datetime.strptime(Name, '%Y-%m-%d')
    #Get the day previous to dateF.
    dateB = dateF - datetime.timedelta(days=1)
    #Convert the dateB to a string.
    dateB = dateB.strftime('%Y-%m-%d')
    return dateB

def nextDate(date):
    date = datetime.datetime.strptime(date, '%Y-%m-%d')
    date = date + datetime.timedelta(days=1)
    date = date.strftime('%Y-%m-%d')
    return date

def nextNDates(n, date):
    for i in range(n):
        date = nextDate(date[:])
    return date

#Calculate the difference between the cost of the day before and the current day, the first day compare to the initial cost.
def calculateDifference(prices):
    resultD = {}
    for date in prices:
        #Check if the date is the first day.
        if date == firstDay:
            resultD[date] = prices[date] - initialCost
        else:
            resultD[date] = prices[date] - prices[previousDate(date)]
    return resultD

def getNPreviousDates(n, date):
    resultD = []
    for i in range(n):
        date = previousDate(date)
        resultD.append(date)
    return resultD

def sort_dates_no_duplicates(dates):
    dateL = list(set(dates))
    dateL.sort()
    return dateL

def frequencyDates(dates):
    nDates = sort_dates_no_duplicates(dates)
    resultD = {}
    for date in nDates:
        resultD[date] = dates.count(date)
    return resultD

#Fr = frequencyDates(dates)
#Dual plot of Frequency of News and prices
#plt.plot(Fr.keys(),Fr.values())
#plt.plot(prices.keys(),prices.values())
#plt.show()

#print(Fr)
#Ocurr = list(Fr.values())
#print("Extreme Ocurrency of News/Day Min:", min(Ocurr), "Max:", max(Ocurr))

#Get the difference between the prices of the day before and the current day.
#difference = calculateDifference(prices)
