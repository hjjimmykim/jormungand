import numpy as np
import matplotlib.pyplot as plt

# Display image
def display_image(image):
    # image = 1 x width*width

    # Convert to suitable forms for display
    width = int(np.sqrt(len(image)))
    image = image.reshape([width,width]) # 28 x 28

    # Display
    plt.imshow(image,cmap='gray')
    plt.axis('off')

# Apply random permutation to sets of images
def permutate(X_list):
    # X_list = list containing arrays of shape batch_size x input_dim
    
    # Random permutation
    permutation = np.arange(X_list[0].shape[1])
    np.random.shuffle(permutation)
    
    # Apply permutation
    X_list2 = []
    for i in range(len(X_list)):
        X_list2.append(X_list[i][:,permutation])
    
    return X_list2, permutation