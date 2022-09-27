import torch

tensor_A = torch.tensor(
    [
        [1,2],
        [3,4],
        [5,6]
    ]
)

tensor_B = torch.tensor(
    [
        [7,10],
        [8,11],
        [9,12]
    ]
)

# torch.matmul(tensor_A, tensor_B) 
# this will not work as AB not "defined" meaning
# the number of rows in tensor_B do not match the number
# of columns in tensor_A


# how to reshape
print(f"Original tensor: {tensor_B}")
print(f"Transposed tensor: {tensor_B.T}")

# you can potentially use transpose to reshape one of the tensors
# so that the inner dimensions match

# this example will work
print(torch.matmul(tensor_A, tensor_B.T))


