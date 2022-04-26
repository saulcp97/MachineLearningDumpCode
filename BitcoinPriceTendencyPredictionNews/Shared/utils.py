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