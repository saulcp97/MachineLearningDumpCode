#Pytorch LSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt
# Read data from the file dataTrain.csv with the next format:
#The first column is the date
#The second column is the difference.
#The third column is the sum of the category that date.
#The next 41 colums are the vector of the category that date.
import numpy as np
#save in 'data' variable the content of the file dataTrain.csv with np
#The first column is a string
#The second column is a float
#The rest are Ints 41
#remove the last column
data = np.loadtxt('dataTrain.csv', delimiter=',', skiprows=0, usecols=range(1,44))
print(data[0])

difs = data[:,0]
#Min and max of the difference
difs_min = np.min(difs)
difs_max = np.max(difs)
print(difs_min, difs_max)

def normalize(difs):
    return (difs - difs_min) / (difs_max - difs_min)

def denormalize(difs):
    return difs * (difs_max - difs_min) + difs_min

print(normalize(-2237))
print(denormalize(1))
#Create the LSTM model with torch
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

#The input vector have 42 elements, the output vector have 1 element.
input_dim = 42
hidden_dim = 10
num_layers = 2
output_dim = 1
model = LSTM(input_dim, hidden_dim, num_layers, output_dim)

#Create the optimizer
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Create the loss function
criterion = nn.MSELoss()

#Separate the training data and the test data with the ratio 0.8
ratio = 0.8
trainData = data[0:int(ratio*len(data))]
testData = data[int(ratio*len(data)):]
trainDataC = trainData.copy()
#Train the model
num_epochs = 10
errorIt = []
errorTest = []
for epoch in range(num_epochs):
    iterE = 0
    trainData = trainDataC
    for i in range(len(trainData)):
        #Convert the data to a tensor
        #print(data[i].shape)
        x = torch.from_numpy(trainData[i][1:]).float()
        result = np.array(normalize(trainData[i][0]))
        #Ponderate the result to (0,1) using as a reference min and max of the difference
        y = torch.from_numpy(result).float()
        x = x.view(1, -1)
        y = y.view(1, -1)
        #Forward pass
        #Turn x from 2 dimension to 3 dimension
        x = x.view(1, 1, -1)

        #print(x)

        outputs = model(x)
        #Calculate the loss
        loss = criterion(outputs, y)
        #print(outputs, y, normalize(trainData[i][0]))
        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iterE += loss.item()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, len(trainData), loss.item()))
    errorIt.append(iterE)
    #Print the error of the iteration
    print('Error of the iteration: {:.4f}'.format(iterE))

#Plot the error of the iteration
plt.plot(errorIt)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error of the iteration')
plt.show()
#Plot the error of the test data
plt.plot(errorTest)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error of the test data')
plt.show()

testData = np.loadtxt('dataTrain.csv', delimiter=',', skiprows=0, usecols=range(1,44))

percent = 0
difer = 0
testError = 0
#Append real difference

#Add to value the prediction of the model and save the points in falseCh to plot the graph
value = 0
falseCh = []
for i in range(len(data)):
    #Convert the data to a tensor
    #print(data[i].shape)
    x = torch.from_numpy(data[i][1:]).float()
    result = np.array(normalize(data[i][0]))
    y = torch.from_numpy(result).float()
    x = x.view(1, -1)
    y = y.view(1, -1)
    #Forward pass
    #Turn x from 2 dimension to 3 dimension
    x = x.view(1, 1, -1)
    outputs = model(x)
    loss = criterion(outputs, y)
    #Convert the output to a range of (difs_min, difs_max)
    out = outputs.item()
    difer = abs(out - result)
    #print(x, denormalize(out), data[i][0])
    value += denormalize(out)
    falseCh.append(value)
    testError += loss.item()
percent = difer/len(data) * 100
print('Error of the test data: {:.4f}'.format(testError))
print('Percent of the error: {:.4f}'.format(percent))

#Print falseCh and realCh in the same plot
plt.plot(falseCh, label='False')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error of the test data')
plt.legend()
plt.show()
