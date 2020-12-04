'''in Pytorch'''
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim, tensor, cuda
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from google.colab import drive

drive.mount('/content/gdrive')

# Training settings
batch_size = 64
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Training Otto Group Product Model on {device}\n{"=" * 44}')


# Dataset
class ProductDataset(Dataset):
  """ Otto Group Product dataset. """

  # initialize your data, download, etc.
  def __init__(self):
    xy = np.genfromtxt('/content/gdrive/My Drive/Colab Notebooks/data/product_train.csv', 
                       delimiter=',', dtype=None, encoding='utf8')
    
    self.len = xy[1:, 1:].shape[0] # 열의 이름에 해당하는 첫 행 제외, id에 해당하는 첫번째 열 제외
    self.x_data = from_numpy(np.array(xy[1:,1:-1]).astype(np.float32))
    y_data = np.array(xy[1:, [-1]])
    data = pd.DataFrame(y_data)
    data = data.astype('category')
    y_data = pd.get_dummies(data)
    self.y_data = from_numpy(np.array(y_data))


  def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]

  def __len__(self):
    return self.len




# Data Loader (Input Pipeline)
dataset = ProductDataset()
train_loader = DataLoader(dataset = dataset, 
                          batch_size = batch_size, 
                          shuffle = True)
test_loader = DataLoader(dataset = dataset, 
                         batch_size = batch_size, 
                          shuffle = False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(93, 80)
        self.l2 = nn.Linear(80, 60)
        self.l3 = nn.Linear(60, 40)
        self.l4 = nn.Linear(40, 20)
        self.l5 = nn.Linear(20, 9)
        self.softmax = nn.Softmax
        self.f = nn.ReLU

    def forward(self, x):
        x = x.view(-1, 93)  # Flatten the data -> (n, 93)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() #zero gradient
        output = model(data)
        loss = criterion(output, target)
        loss.backward() 
        optimizer.step() #gradient disent 
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = float(data), float(target) 
        output = model(data) # data in model
        # sum up batch loss 
        test_loss += criterion(output, target).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test()



