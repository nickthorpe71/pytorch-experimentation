from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

n_samples = 1000

X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

print(len(X), len(y))

print(f"First 5 samples of X:\n {X[:5]}")
print(f"First 5 samples of y:\n {y[:5]}")

circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})

print(circles.head(10))

plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)

plt.savefig("output.jpg")

# Check input and output shapes
print(X.shape, y.shape)

# View the first example of features and labels
X_sample = X[0]
y_sample = y[0]

# We heave 2 features of X trying to predict 1 value y
print(X_sample)
print(y_sample)

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float) 

print(X[:5], y[:5])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(X_train), len(X_test), len(y_train), len(y_test))



# device agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"

# build a model
class CircleModelV0(nn.Module):
  def __init__(self):
    super().__init__()
    # create 2 nn.Linear layers capable of handling the shapes of our data
    self.layer_1 = nn.Linear(in_features=2, out_features=5)
    self.layer_2 = nn.Linear(in_features=5, out_features=1)
  
  # define a forward() method that outlines the forward pass
  def forward(self, x):
    return self.layer_2(self.layer_1(x))
  
# instantiate an instance of our model class and send to target device
model_0 = CircleModelV0().to(device)
print(model_0)
print(next(model_0.parameters()).device)


# create the same model in less code
model_1 = nn.Sequential(
  nn.Linear(in_features=2, out_features=5),
  nn.Linear(in_features=5, out_features=1)
).to(device)

print(model_1)

print(model_1.state_dict())

# make predictions
with torch.inference_mode():
  untrained_preds = model_1(X_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(X_test)}, Shape:{X_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 labels:\n{y_test[:10]}")
