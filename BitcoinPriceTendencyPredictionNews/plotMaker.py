#Given the markov model, make a plot of the accuracy of the model, depending on the percentage of the data used.
#The Axis X is the percentage of the data used, and the axis Y is the accuracy of the model.
#In blue is the accuracy of the model with training data, and in red is the accuracy of the model with test data.

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from utils import *
from markov import Markov

#Fixing the seed of the random number generator.
np.random.seed(0)

#Create the lists that will be used to plot the graph.
aTrD = []
aTeD = []

#Define the minimum and maximum percentage of the data used.
minPerc = 1
maxPerc = 99

#Load all the data using the utils package.
categories = extractV('category')
news = clean_text(extractV('headline'))

#Define the Bag of Words.
bWords = list(set([x for phrase in news for x in phrase]))
#Define the Baf of Categories.
bCats = list(set(categories))
#Occurrences of each category.
catOcc = {x:categories.count(x) for x in bCats}

#Create the Markov model.
markov = Markov(bWords, bCats, 0.01)
markov.initializeClassifier()
markov.save('auxiliarF')

#Start the loop that will make the graph.
for i in range(minPerc, maxPerc + 1):
    #Create the training and test data. Define the length of the training data.
    trLen = int(len(news) * i / 100)
    trainDNews = news[:trLen]
    trainDCats = categories[:trLen]
    testDNews = news[trLen:]
    testDCats = categories[trLen:]

    #Load the Markov model
    markov = markov.load("auxiliarF")

    #Train the model.
    markov.train(trainDNews, trainDCats)
    #Apply the smoothing for ponderation of occurrences.
    markov.ponderate(catOcc)

    #Apply the e.
    markov.apply_e()

    #Get the test accuracy.
    errorTe = markov.calculateError(testDNews, testDCats)
    #Get the training accuracy.
    errorTr = markov.calculateError(trainDNews, trainDCats)
    
    #Print the error of the model and the percentage of the data used.
    print("Error of the model with training data: ", errorTr, " with ", i, "% of the data used.")
    print("Error of the model with test data: ", errorTe, " with ", i, "% of the data used.")
    
    #Append the accuracy to the lists.
    aTrD.append(errorTr)
    aTeD.append(errorTe)

#Plot the graph.
plt.plot(range(minPerc, maxPerc + 1), aTrD, 'b', label='Training Data')
plt.plot(range(minPerc, maxPerc + 1), aTeD, 'r', label='Test Data')
plt.xlabel('Percentage of Data')
plt.ylabel('Error')
plt.title('Error of the Model')
plt.legend()
plt.show()