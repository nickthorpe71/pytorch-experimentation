import torch 

weights = torch.ones(4, requires_grad=True)

for epoch in range(2):
  model_output = (weights*3).sum()
  model_output.backward()
  # at this point gradienrs are accumulated which is incorrect
  print(weights.grad) 
  # to fix this we need to zero the gradients
  weights.grad.zero_()