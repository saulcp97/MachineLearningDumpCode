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
data = np.loadtxt('dataTest2_5.csv', delimiter=',', skiprows=0, usecols=range(1,216))
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
input_dim = 214
hidden_dim = 32
num_layers = 10
output_dim = 1
model = LSTM(input_dim, hidden_dim, num_layers, output_dim)

#Create the optimizer
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

#Define the dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx][1:], 1 if self.data[idx][0] > 0 else 0

#Define the training dataset with ratio 90%/10%
train_size = int(0.9 * len(data))
test_size = len(data) - train_size

train_idx = np.arange(len(data))
#Separate randomly the training data and the test data with the ratio 0.8 using train_idx
trainData = data[train_idx[:int(0.8*len(train_idx))]]
testData = data[train_idx[int(0.8*len(train_idx)):]]

train_dataset = Dataset(trainData)
test_dataset = Dataset(testData)

#Train the model
num_epochs = 50
errorIt = []
for epoch in range(num_epochs):
    errItA = 0
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=True)
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.view(1, 1, -1)
        #Turn in float
        x = x.float()
        y = y.float()
        output = torch.reshape(model(x), (-1,))
        #print(output.shape)
        #print(y.shape)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        errItA += loss.item()
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, len(train_dataset), loss.item()))
    errorIt.append(errItA/len(train_dataset))

#Plot the error
plt.plot(errorIt)
plt.show()

#Test the model
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=1,
                                            shuffle=True)
aciertos = 0
for i, (x, y) in enumerate(test_loader):
    x = x.view(1, 1, -1)
    x = x.float()
    y = y.float()
    output = model(x)
    if (output > 0.5 and y == 1) or (output < 0.5 and y == 0):
        aciertos += 1
    if (i+1) % 100 == 0:
        print('Test: [{}/{}], Loss: {:.4f}'.format(i+1, len(test_dataset), loss.item()))

print("Aciertos: ", aciertos/len(test_dataset) * 100, "%")
