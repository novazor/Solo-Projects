import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image
import numpy as np

# Load the dataset
dataset = pd.read_csv('datasets/pokemon.csv')
print(f'Dataset shape: {dataset.shape}')
dataset.head()

# Visualize the ratio of legendary and non-legendary Pokémon
plt.figure(figsize=(10, 6))
dstrbtn = sns.countplot(x='is_legendary', data=dataset)
for p in dstrbtn.patches:
    dstrbtn.annotate(f'\n{p.get_height()}', (p.get_x() + 0.3, p.get_height()), ha='center', va='center', color='black', size=12)
plt.title('Distribution of Legendary vs Non-Legendary Pokémon')
plt.xlabel('Is Legendary')
plt.ylabel('Count')
plt.show()

# Clean the dataset
dataset_cleaned = dataset.select_dtypes(exclude=['object']).drop(['pokedex_number', 'percentage_male'], axis=1).dropna()
print(f'Cleaned dataset shape: {dataset_cleaned.shape}')
dataset_cleaned.head()

# Check the ratio of legendaries hasn't changed too much
dstrbtn = sns.countplot(x='is_legendary', data=dataset_cleaned)
for p in dstrbtn.patches:
    dstrbtn.annotate(f'\n{p.get_height()}', (p.get_x() + 0.3, p.get_height()), ha='center', va='center', color='black', size=12)
plt.show()

# Split the data into input features (X) and labels (y)
X = dataset_cleaned.iloc[:, :-1]
y = dataset_cleaned.iloc[:, -1]

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X.values)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=60, stratify=y)

# Custom Dataset classes
class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.x_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return len(self.x_data)
    
train_data = TrainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train))

class TestData(Dataset):
    def __init__(self, X_data):
        self.x_data = X_data

    def __getitem__(self, index):
        return self.x_data[index]
    
    def __len__(self):
        return len(self.x_data)
    
test_data = TestData(torch.FloatTensor(X_test))

# Hyperparameters
HIDDEN_SIZE = 64
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# DataLoaders
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)

# Defining the model
class PokePredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PokePredictor, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.out(x)
        return x

model = PokePredictor(input_size=X_train.shape[1], hidden_size=HIDDEN_SIZE)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

# Training loop, printing accuracy at each step
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(f'Epoch {e:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

# Testing the model
y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_list)
print('Confusion Matrix:')
print(cm)

# Compute the accuracy
accuracy = accuracy_score(y_test, y_pred_list)
print(f'Accuracy: {accuracy:.4f}')

# Prediction for a single Pokemon (randomly chosen)
pokemon = dataset.sample()
pokemon_name = pokemon['name'].values[0].lower()

pokemon = pokemon[dataset_cleaned.columns].values[0][:-1]
pokemon = scaler.transform(pokemon.reshape(1, -1))
pokemon = torch.FloatTensor(pokemon).to(device)

model.eval()
with torch.no_grad():
    y_pred = model(pokemon)
    y_pred = torch.sigmoid(y_pred)

img = Image.open(f'Pokemon Dataset/{pokemon_name}.png')
print(f'Pokemon: {pokemon_name}')
print(f'Legendary confidence level: {y_pred.item():.3f}')
img