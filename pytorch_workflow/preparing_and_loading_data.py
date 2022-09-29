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
    plt.scatter(test_data, predictions, c="r", s=4, labe="Predictions")
    
  plt.legend(prop={"size": 14})
  plt.show(block=True)

plot_predictions()