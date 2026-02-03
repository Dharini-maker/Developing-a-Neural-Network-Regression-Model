# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY

Neural network regression is a supervised learning technique used to predict continuous output values based on given input data. In this approach, a dataset containing one numeric input and one numeric output is used to train the neural network. The network learns the underlying relationship between input and output by adjusting its weights through repeated iterations to minimize the training loss. The training loss versus iteration graph provides insight into the learning behavior of the model and helps in analyzing its convergence and performance during training.

## Neural Network Model
Include the neural network model diagram.

<img width="1116" height="760" alt="image" src="https://github.com/user-attachments/assets/80303916-8def-46a8-947e-ffba82497b7b" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:

### Register Number:

```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
dataset1 = pd.read_csv('/content/exp1.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values
print(X)
print(y)
```
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}
  def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss=criterion(ai_brain(X_train),y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)
```
```
with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')
```
```
loss_df = pd.DataFrame(ai_brain.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()
```
```
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')

```

### Dataset Information

<img width="270" height="438" alt="image" src="https://github.com/user-attachments/assets/f3f661bc-5928-41a8-bcb5-703838a0e5aa" />


### OUTPUT

<img width="797" height="302" alt="image" src="https://github.com/user-attachments/assets/d8fc5c32-47c9-4917-8e6f-732c19c46546" />

<img width="666" height="152" alt="image" src="https://github.com/user-attachments/assets/8850b458-ef82-4c08-ad07-cd541f3d680e" />


### Training Loss Vs Iteration Plot

<img width="950" height="126" alt="image" src="https://github.com/user-attachments/assets/865f30f1-e99f-4d5b-9309-1f2bcdf79d89" />


### New Sample Data Prediction

<img width="950" height="126" alt="image" src="https://github.com/user-attachments/assets/d73d2f37-2e9b-43e7-9083-44606c610d03" />

## RESULT

Thus, a neural network regression model was successfully developed and trained using PyTorch.
