'''in Pytorch'''
import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [-2.0, -2.0, 0.0, 4.0, 10.0]

w1 = 1.0 #random guess : random value
w2 = 1.0
b = 3.0


#our model forward pass
def forward(x):
  return x*x*w2 + x*w1 + b

#Loss function
def loss(x,y):
  y_pred = forward(x)
  return (y_pred - y) * (y_pred - y)

#compute gradient
def gradient2(x,y): #d_loss/d_w2
  return 2 * (x*x) * (x*x*w2 + x*w1 + b - y)

def gradient1(x,y): #d_loss/d_w1
  return 2 * x * (x*x*w2 + x*w1 + b - y)

def gradient0(x,y): #d_loss/d_b
  return 2 * (x*x*w2 + x*w1 + b - y)


#before training
print("predict (before training)", 4, forward(4))

#Training Loop
for epoch in range(100):
  for x_val, y_val in zip(x_data, y_data):
    grad0 = gradient0(x_val, y_val)
    grad1 = gradient1(x_val, y_val)
    grad2 = gradient2(x_val, y_val)
    b = b - 0.003 * grad0
    w1 = w1 - 0.003 * grad1
    w2 = w2 - 0.003* grad2
    print("\tgrad: ", x_val, y_val, grad2, grad1, grad0)
    l = loss(x_val, y_val)

  print("progress:", epoch, "w2=", w2, "w1=", w1, "b=", b, "loss=", l)

# After training
print("predict (after training)", "4 hours", forward(4))  
