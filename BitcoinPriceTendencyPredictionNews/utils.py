#Utility functions for the project
import json

#Read the data from the json file "News_Category_Dataset_v2.json" and train the classifier to classify the news articles into the categories.
#Read line per line the json file and store the value of the key "category" in a dictionary as a key, counting the number of times occurance of each category.

#A function to read the file 'News_Category_Dataset_v2.json' and return a list of the values of the key.
def extractV(key):
    values = []
    with open('News_Category_Dataset_v2.json') as json_file:
        for line in json_file:
            data = json.loads(line)
            values.append(data[key])
    return values

def extractBitcoinDict():
    dates = {}
    with open('bitcoinPriceDate.json') as json_file:
        print("Reading the file...")
        for line in json_file:
            data = json.loads(line)
            #print(data["bpi"])
            dates = data["bpi"]
    return dates


#A function to read the file 'News_Category_Dataset_v2.json' and return a list with the ocurrance of each category, each date and the total number of articles that day.
def extract_data():
    #Extract the categories.
    categories = extractV("category")
    catName = list(set(categories))
    catName.sort()
    #Create a dictionary with the categories as a key and the value their position in the list.
    catDict = {}
    for i in range(len(catName)):
        catDict[catName[i]] = i

    dictDate = {}
    #Explore the file.
    with open('News_Category_Dataset_v2.json') as json_file:
        print("Reading the file...")
        for line in json_file:
            data = json.loads(line)
            key = data["date"]
            cat = data["category"]
            if key in dictDate:
                index = catDict[cat]
                dictDate[key][index] += 1
            else:
                dictDate[key] = [0] * len(catName)
                index = catDict[cat]
                dictDate[key][index] += 1
    return dictDate, catName

import datetime
def previousDate(Name):
    #Convert the date to a datetime object
    dateF = datetime.datetime.strptime(Name, '%Y-%m-%d')
    #Get the day previous to dateF.
    dateB = dateF - datetime.timedelta(days=1)
    #Convert the dateB to a string.
    dateB = dateB.strftime('%Y-%m-%d')
    return dateB

#Group the vector from extract_data() with the next 5 days.
def extract_data_NDays(N):
    dictDate, catName = extract_data()
    dictDate_NDays = {}
    for key in dictDate:
        dictDate_NDays[key] = [dictDate[key]]
        actualDate = key
        for i in range(N):
            actualDate = previousDate(actualDate)
            if actualDate in dictDate:
                dictDate_NDays[key].append(dictDate[actualDate])
    return dictDate_NDays, catName


#A function to clean the text of the news articles.
def clean_text(news):
    news_clean = []
    for i in range(len(news)):
        news_clean.append(news[i].lower())
        news_clean[i] = news_clean[i].replace(".", "")
        news_clean[i] = news_clean[i].replace(",", "")
        news_clean[i] = news_clean[i].replace("!", "")
        news_clean[i] = news_clean[i].replace("?", "")
        news_clean[i] = news_clean[i].replace("'", "")
        news_clean[i] = news_clean[i].replace("\"", "")
        news_clean[i] = news_clean[i].replace("-", " ")
        news_clean[i] = news_clean[i].replace("/", " ")
        news_clean[i] = news_clean[i].replace("(", " ")
        news_clean[i] = news_clean[i].replace(")", " ")
        news_clean[i] = news_clean[i].replace("[", " ")
        news_clean[i] = news_clean[i].replace("]", " ")
        news_clean[i] = news_clean[i].replace("{", " ")
        news_clean[i] = news_clean[i].replace("}", " ")
        news_clean[i] = news_clean[i].replace("<", " ")
        news_clean[i] = news_clean[i].replace(">", " ")
        news_clean[i] = news_clean[i].replace("#", " ")
        news_clean[i] = news_clean[i].replace("$", " ")
        news_clean[i] = news_clean[i].replace("%", " ")
        news_clean[i] = news_clean[i].replace("^", " ")
        news_clean[i] = news_clean[i].replace("&", " ")
        #Split the cleaned text into a list of words.
        news_clean[i] = news_clean[i].split()
    return news_clean