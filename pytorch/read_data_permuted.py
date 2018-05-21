import numpy as np
import torchvision
from utility import trans
import torchvision.transforms as transforms

def read_data_permuted(input_dim,N_task):

    # Generate random permutations
    perms = []
    perms.append(np.arange(input_dim))  # Non-permuted
    for i in range(N_task-1):
        perms.append(np.random.permutation(perms[0]))
            
    # Train datasets
    trainsets = [torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=trans(perm)) for perm in perms]
        
    # Test datasets
    testsets = [torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=trans(perm)) for perm in perms]
    
    return trainsets, testsets