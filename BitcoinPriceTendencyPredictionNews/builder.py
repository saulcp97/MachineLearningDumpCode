#Build The project Dataset

#Using the function calculateDifference from dateSaver.py
from numpy.core.defchararray import count
from dateSaver import *
from utils import *

prices = extractBitcoinDict()
difference = calculateDifference(prices)

print(difference["2018-01-27"])


dDate, dCat = extract_data()

d5Date, d5Cat = extract_data_NDays(5)
#Create a csv file called "dataTrain.csv"
#The first column is the date
#The second column is the difference.
#The third column is the sum of the category that date.
#The next 41 colums are the vector of the category that date.

#dDate is a dictionary with the date as a key and the vector of the category as a value.
#with open("dataTrain.csv", "w") as f:
#    for key in difference:
#        if key in dDate:
#            f.write(key + "," + str(difference[key]) + "," + str(sum(dDate[key])) + ",")
#            for i in range(len(dDate[key])):
#                f.write(str(dDate[key][i]) + ",")
#            f.write("\n")

#dictionary = {category:categories.count(category) for category in categoryK}
print(difference["2018-01-27"], dDate["2018-01-27"])


#Now create a csv file called "dataTest2_5.csv" whit the same structure of dataTrain.csv, but ignore the 5 firsts days.
count = 0
with open("dataTest2_5.csv", "w") as f:
    for key in difference:
        n2 = nextNDates(2, key)
        #print(difference[n2])
        print(n2)
        count += 1
        if count > 5:
            if key in d5Date and n2 in difference:
                f.write(n2 + "," + str(difference[n2]) + ",")

                
                prev5 = getNPreviousDates(5, key)
                for i in range(len(prev5)):
                    f.write(str(difference[prev5[i]]) + ",")
                    
                for i in range(len(d5Date[key])):
                    print(key)
                    sumCatD = sum(d5Date[key][i])
                    f.write(str(sumCatD) + ",")

                for i in range(len(d5Date[key])):
                    for j in range(len(d5Date[key][i])):
                        f.write(str(d5Date[key][i][j]) + ",")
                f.write("\n")
# "," + str(sum(d5Date[key])) + ","