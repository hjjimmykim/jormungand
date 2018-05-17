import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Display image
def display_image(image):
    # image = 1 x width*width

    # Convert to suitable forms for display
    width = int(np.sqrt(len(image)))
    image = image.reshape([width,width]) # 28 x 28

    # Display
    plt.imshow(image,cmap='gray')
    plt.axis('off')

# Apply random permutation to dataset
def permutate(dataset):
    # Random permutation
    permutation = np.arange(dataset.train.images.shape[1])
    np.random.shuffle(permutation)
    
    # New permuted dataset
    dataset_2 = deepcopy(dataset)
    
    # _images = property as opposed to images = function
    dataset_2.train._images = dataset.train.images[:,permutation]
    dataset_2.validation._images = dataset.validation.images[:,permutation]
    dataset_2.test._images = dataset.test.images[:,permutation]
    
    return dataset_2
    
    