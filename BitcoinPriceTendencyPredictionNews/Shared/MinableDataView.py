#Python
import csv

with open('dataTrain.csv', newline='') as csvfile:
  spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
  for row in spamreader:
    print(', '.join(row))

#For the dataTrain.csv file, get the maximum and minimum values for the second column.
with open('dataTrain.csv', newline='') as csvfile:
  spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
  max = 0
  min = 0
  for row in spamreader:
    if row[1] > max:
      max = row[1]
    if row[1] < min:
      min = row[1]
    print("Max: " + str(max))
    print("Min: " + str(min))

#Calculate the average of the second column.
with open('dataTrain.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    sum = 0
    for row in spamreader:
        sum += float(row[1])
    print("Average: " + str(sum/len(spamreader)))

#Calculate the standard deviation of the second column.
with open('dataTrain.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    sum = 0
    inst = 0
    for row in spamreader:
        sum += float(row[1])
        inst += 1
    avg = sum/inst
    print("Average: " + str(avg))
    sum = 0
    for row in spamreader:
        sum += (float(row[1]) - avg)**2
        if sum == 0:
            print("Standard goes zero: " + str(0))
    
    std = (sum/inst)**(1/2)
    print("Standard Deviation: " + std)
