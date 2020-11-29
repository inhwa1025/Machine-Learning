'''in Pytorch'''
import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [-2.0, -2.0, 0.0, 4.0, 10.0]

w2 = torch.tensor([1.0], requires_grad=True)
w1 = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)

def forward(x):
  return x*x*w2 + x*w1 + b

def loss(x,y):
  return (y_pred - y) **2

#before training
print("predict (before training)", 4, forward(4).item())

#Training loop
for epoch in range(10):
  for x_val, y_val in zip(x_data, y_data):
    y_pred = forward(x_val)     # 1) Forward pass
    l = loss(y_pred, y_val)     # 2) Compute loss
    l.backward()                # 3) Back propagation to update weights
    print("\tgrad: ", x_val, y_val, w2.grad.item(), w1.grad.item(), b.grad.item())
    w2.data = w2.data - 0.003 * w2.grad.item()
    w1.data = w1.data - 0.003 * w1.grad.item()
    b.data = b.data - 0.003 * b.grad.item()

    # Manually zero the gradients after updating weights
    w2.grad.data.zero_()
    w1.grad.data.zero_()
    b.grad.data.zero_()

  print(f"Epoch: {epoch} | Loss: {l.item()}")

# After training
print("Prediction (after training)", 4, forward(4).item())
