'''in Pytorch'''
import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0 #random guess : random value

#our model forward pass
def forward(x):
  return x*w

#Loss function
def loss(x,y):
  y_pred = forward(x)
  return (y_pred - y) * (y_pred - y)

#compute gradient
def gradient(x,y): #d_loss/d_w
  return 2*x*(x*w-y)

#before training
print("predict (before training)", 4, forward(4))

#Training Loop
for epoch in range(100):
  for x_val, y_val in zip(x_data, y_data):
    grad = gradient(x_val, y_val)
    l = loss(x_val, y_val)
    w = w - 0.01 * grad
    print("\tgrad: ", x_val, y_val, grad)
    

  print("progress:", epoch, "w=", w, "loss=", l)

# After training
print("predict (after training)", "4 hours", forward(4))  
