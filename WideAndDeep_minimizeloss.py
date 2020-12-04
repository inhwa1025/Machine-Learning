'''in Pytorch'''
from torch import nn, optim, from_numpy
import numpy as np
from google.colab import drive

drive.mount('/content/gdrive')

xy = np.loadtxt('/content/gdrive/My Drive/Colab Notebooks/data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = from_numpy(xy[:, 0:-1])
y_data = from_numpy(xy[:, [-1]])
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}')


class Model(nn.Module):
  def __init__(self):
    """
    In the constructor we instantiate two nn.Linear Module
    """
    super(Model, self).__init__()
    self.l1 = nn.Linear(8, 7)
    self.l2 = nn.Linear(7, 6)
    self.l3 = nn.Linear(6, 5)
    self.l4 = nn.Linear(5, 4)
    self.l5 = nn.Linear(4, 3)
    self.l6 = nn.Linear(3, 2)
    self.l7 = nn.Linear(2, 1)
    self.sigmoid = nn.Sigmoid()
    self.selu = nn.SELU()

  def forward(self, x):
    """
    In the forward function we accept a Variable of input data and we must return
    a Vriable of output data. We can use Modules defined in the constructor as
    well as arbitraty operators on Variables.
    """
    out1 = self.selu(self.l1(x))
    out2 = self.selu(self.l2(out1))
    out3 = self.selu(self.l3(out2))
    out4 = self.selu(self.l4(out3))
    out5 = self.selu(self.l5(out4))
    out6 = self.selu(self.l6(out5))
    y_pred = self.sigmoid(self.l7(out6))
    return y_pred

model = Model()


criterion = nn.BCELoss(reduction = 'mean')
optimizer = optim.ASGD(model.parameters(), lr=0.15)


# Training Loop
for epoch in range(100):
  # Forward pass: Compute predicted y by passing x to the model
  y_pred = model(x_data)

  # Compute and print loss
  loss = criterion(y_pred, y_data)
  print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item(): .4f}')

  # Zero gradients, perform a backward pass, and update the weights.
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()



