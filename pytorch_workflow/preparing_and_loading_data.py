import torch
from torch import nn
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

# print(X[:10])
# print(y[:10])
# print(len(X))
# print(len(y))


# Splitting data into training and test sets
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# print(len(X_train)) 
# print(len(y_train)) 
# print(len(X_test)) 
# print(len(y_test))

def plot_predictions(train_data=X_train, 
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10,7))
  
  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Test data")
  
  if predictions is not None:
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    
  plt.legend(prop={"size": 14})
  plt.savefig("output.jpg")
  
# How to get plot to render in wsl2: https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2
# plot_predictions()

class LinearRegressionModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.weights = nn.Parameter(torch.randn(1,
                                            requires_grad=True,
                                            dtype=torch.float))
    self.bias = nn.Parameter(torch.randn(1,
                                         requires_grad=True,
                                         dtype=torch.float))
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.weights * x + self.bias # this is the linear regression formula
  
  
# Create a random seed to keep example consistant every run
torch.manual_seed(11)

# Create an instance of the modael 
model_0 = LinearRegressionModel()

# print(list(model_0.parameters()))
# print(model_0.state_dict())

# Make predictions with model

with torch.inference_mode():
  y_preds = model_0(X_test)
  
# print(y_preds)

# plot_predictions(predictions=y_preds)

# Set up a loss funciton
loss_fn = nn.L1Loss()

# Set up an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01) # lr = learning rate


# Build a training loop

# An epoch is one loop through the data...
epochs = 500

# Loop through the data
for epoch in range(epochs):
  # Set the model to training mode
  model_0.train() # train mode turns on gradient descent
  
  # Forward pass
  y_pred = model_0(X_train)
  
  # Calculate the loss
  loss = loss_fn(y_pred, y_train)
  
  # Optimizer zero grad
  optimizer.zero_grad()
  
  # Back propagation
  loss.backward()
  
  # Gradient descent
  optimizer.step()
  
  
  # Testing
  model_0.eval() # eval mode turns off gradient descent
  with torch.inference_mode():
    # Do forward pass
    test_pred = model_0(X_test)
    
    # Calculate the loss
    test_loss = loss_fn(test_pred, y_test)
  
  if epoch % 10 == 0:
    print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")  
    print(model_0.state_dict())
  
  
with torch.inference_mode():
  y_preds_new = model_0(X_test)
  
plot_predictions(predictions=y_preds_new)
