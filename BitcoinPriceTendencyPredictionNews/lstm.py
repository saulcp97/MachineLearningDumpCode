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
hidden_dim = 5
num_layers = 2
output_dim = 1
model = LSTM(input_dim, hidden_dim, num_layers, output_dim)

#Create the optimizer
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Create the loss function
criterion = nn.MSELoss()

#Random ordered list of range(0, len(data))
train_idx = np.arange(len(data))

#Separate randomly the training data and the test data with the ratio 0.8 using train_idx
trainData = data[train_idx[:int(0.8*len(train_idx))]]
testData = data[train_idx[int(0.8*len(train_idx)):]]

#Train the model
num_epochs = 10
errorIt = []
errorTest = []
for epoch in range(num_epochs):
    iterE = 0
    for i in range(len(trainData)):
        #Clean the gradient
        optimizer.zero_grad()
        #Convert the data to a tensor
        #print(data[i].shape)
        x = torch.from_numpy(data[i][1:]).float().clone()
        result = data[i][0:1].copy()
        #turn result numpy vector all to 0 or 1 if they are negative or positive
        if result[0] < 0:
            result[0] = 0
        else:
            result[0] = 1

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
        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iterE += loss.item()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, len(data), loss.item()))
    errorIt.append(iterE)
    #Print the error of the iteration
    print('Error of the iteration: {:.4f}'.format(iterE))
    #Check the error of the test data
    #Calculate the % of the error
    percent = 0
    testError = 0
    for i in range(len(testData)):
        #Convert the data to a tensor
        #print(data[i].shape)
        x = torch.from_numpy(testData[i][1:]).float()
        result = testData[i][0:1]
        #turn result numpy vector all to 0 or 1 if they are negative or positive
        if result[0] < 0:
            result[0] = 0
        else:
            result[0] = 1

        y = torch.from_numpy(result).float()
        x = x.view(1, -1)
        y = y.view(1, -1)
        #Forward pass
        #Turn x from 2 dimension to 3 dimension
        x = x.view(1, 1, -1)
        outputs = model(x)
        print(outputs.item(), result[0])
        loss = criterion(outputs, y)
        testError += loss.item()
    percent = testError/len(testData)
    print('Error of the test data: {:.4f}'.format(testError))
    errorTest.append(testError)

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

#Check the accuracy of the model
#Calculate the % of the error
percent = 0
testCorrect = 0
for i in range(len(testData)):
    #Convert the data to a tensor
    #print(data[i].shape)
    x = torch.from_numpy(testData[i][1:]).float().clone()
    result = testData[i][0:1].copy()
    #turn result numpy vector all to 0 or 1 if they are negative or positive
    if result[0] < 0:
        result[0] = 0
    else:
        result[0] = 1

    y = torch.from_numpy(result).float()
    x = x.view(1, -1)
    y = y.view(1, -1)
    #Forward pass
    #Turn x from 2 dimension to 3 dimension
    x = x.view(1, 1, -1)
    outputs = model(x)
    output = 0
    if outputs.item() > 0.5:
        output = 1
    
    if output == result[0]:
        testCorrect += 1
percent = testCorrect/len(testData)
print('Accuracy of the test data: {:.4f}'.format(percent))