import torch
import matplotlib.pyplot as plt


# Create known parameters
weight = 0.7
bias = 0.3

# Create
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)  # add an extra dimension
y = weight * X + bias

print(X[:10])
print(y[:10])
print(len(X))
print(len(y))


# Splitting data into training and test sets
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

plt()
