'''in Pytorch'''
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim, tensor
import numpy as np
from sklearn.preprocessing import LabelEncoder
from google.colab import drive

drive.mount('/content/gdrive')


class TitanicDataset(Dataset):
  """ Titanic dataset. """

  # initialize your data, download, etc.
  def __init__(self):
    # 기존의 파일에서 수정할 특징의 열만 가져와서 새로운 파일로 만듦. (pclass, sex, sibsp, parch, survived)
    xy = np.genfromtxt('/content/gdrive/My Drive/Colab Notebooks/data/titanic_train_change.csv', 
                       delimiter=',', dtype=None, encoding='utf8')
    
    self.len = xy[1:, :].shape[0]
    x_data = np.array(xy[1:, 0:4])
    self.y_data = from_numpy(np.array(xy[1:,[-1]]).astype(np.float32))
    x_data[:,1] = LabelEncoder().fit_transform(x_data[:, 1]) #성별 encoding
    self.x_data=from_numpy(x_data.astype(np.float32))


  def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]

  def __len__(self):
    return self.len


dataset = TitanicDataset()
train_loader = DataLoader(dataset = dataset, 
                          batch_size = 32, 
                          shuffle = True, 
                          num_workers = 2)


class Model(nn.Module):
  def __init__(self):
    """
    In the constructor we instantiate two nn.Linear module
    """
    super(Model, self).__init__()
    self.l1 = nn.Linear(4, 3)
    self.l2 = nn.Linear(3, 2)
    self.l3 = nn.Linear(2, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    """
    In the forward function we accept a Variable of input data and we must return
    a Vriable of output data. We can use Modules defined in the constructor as
    well as arbitraty operators on Variables.
    """
    out1 = self.sigmoid(self.l1(x))
    out2 = self.sigmoid(self.l2(out1))
    y_pred = self.sigmoid(self.l3(out2))
    return y_pred

model = Model()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.BCELoss(reduction = 'mean')
optimizer = optim.SGD(model.parameters(), lr=0.1)


# Training loop
for epoch in range(100):
  for i, data in enumerate(train_loader, 0):
    # get the inputs
    inputs, labels = data

    # Forward pass: Compute predicted y by to the model
    y_pred = model(inputs)

    # Compute and print loss
    loss = criterion(y_pred, labels)
    print(f'Epoch {epoch+1} | Batch: {i+1} | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


