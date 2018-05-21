import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from torch import autograd
import matplotlib.pyplot as plt
import numpy as np

# Data preprocessing -------------------------------------------------------------------

# Permutate pixels
def permutate(image, permutation):
    # c = number of channels (e.g. 3 for RGB)
    # h = height
    # w = width
    c, h, w = image.size()
    
    image = image.view(-1,c)        # Resize to 1D array
    image = image[permutation, :]   # Apply permutation
    image = image.view(c, h, w)     # Resize to original shape
    return image
    
# Define transform
def trans(permutation):
    # Transforms (convert to tensor, normalize [0,1] -> [-1,1], apply permutation)
    # mean = 0.5, std = 0.5 => subtract 0.5 from each pixel and divide by 2
    # 3-tuple for 3 channels
    return transforms.Compose([ \
            transforms.ToTensor(), \
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)), \
            transforms.Lambda(lambda x: permutate(x, permutation))])
            
# Extra loss functions -----------------------------------------------------------------

def calc_EWC_loss(net, fixed, Fisher):
    losses = []
    
    for p1, p2, Fish in zip(net.parameters(),fixed,Fisher):
        losses.append((Fish * (p1 - p2)**2).sum())
        
    return sum(losses)

def calc_L2_loss(net, fixed):
    losses = []

    for p1, p2 in zip(net.parameters(),fixed):
        
        losses.append(((p1 - p2)**2).sum())
    
    return sum(losses)
    
# Fisher Info. -------------------------------------------------------------------------

def calc_Fisher(net, dataset, sample_size = 1024, use_cuda = False):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,num_workers=2)

    # Preallocate
    Fisher = [torch.zeros(x.size()) for x in list(net.parameters())]
    if torch.cuda.is_available() and use_cuda:
        Fisher = [x.cuda() for x in Fisher]
    print('blah')
    # Take expectation over log-derivative squared
    num_sampled = 0 # Counter for number of samples so far
    for data, label in dataloader:
        data = data.view(1, -1)

        if torch.cuda.is_available() and use_cuda:
            data = data.cuda()
            label = label.cuda()
        
        # Sample log likelihood

        net.eval()      # Disable dropout layer
        output = net(data)
        net.train()     # Enable dropout layer

        prob = F.softmax(net(data),1)
        
        y_sample = torch.multinomial(prob,1).data # Sample from model
        
        #y_sample = label.data  # Given by data

        logL = F.log_softmax(output,1)[range(1),y_sample.item()]

        # First derivative
        logL_grad = autograd.grad(logL, net.parameters())
        
        # Squared & convert to tensor
        logL_grad_sq = [x.data**2 for x in logL_grad]
        
        # Accumulate Fisher
        for i in range(len(logL_grad_sq)):
            Fisher[i] += logL_grad_sq[i]
        
        num_sampled += 1
        if num_sampled >= sample_size:
            break
            
    # Average
    for i in range(len(Fisher)):
        Fisher[i] /= sample_size

    return Fisher

# Test accuracy ------------------------------------------------------------------------

def test_acc(net, dataset, input_dim, batch_size, use_cuda=False, disp=False):
    net.eval()  # Switch to eval mode (disable dropout layer)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images = images.view(-1,input_dim) # First dim is batch_size
        if torch.cuda.is_available() and use_cuda:
            images = images.cuda()
            labels = labels.cuda()
        
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0) # Batch size
        correct += (predicted == labels).sum()
        
    acc = correct/total
        
    if disp:
        print('Accuracy of the network on the 10000 test images: %d %%' % (100* acc))
        
    net.train() # Revert to train mode
    
    return acc
    
# Plot accuracy ------------------------------------------------------------------------

def plot_accuracy(time_list, acc_lists, acc_L2_lists, acc_EWC_lists, N_task, ylabelname = 'Accuracy', save=False, savename='blah'):
    num_lists = len(acc_lists)

    # Colormap for separate tasks
    colormap = plt.cm.jet
    colors = [colormap(i) for i in np.linspace(0,1,N_task)]

    plt.figure()

    # Vanilla
    plt.subplot(1,3,1)
    plt.title('Vanilla')
    for j in range(N_task):
        plt.plot(time_list, acc_lists[j], label = 'Task ' + str(j+1), c=colors[j])
    plt.ylim([0,1])
    plt.xlabel('Training time')
    plt.ylabel(ylabelname)
    plt.legend()

    # L2
    plt.subplot(1,3,2)
    plt.title('L2')
    for j in range(N_task):
        plt.plot(time_list, acc_L2_lists[j], label = 'Task ' + str(j+1), c=colors[j])
    plt.ylim([0,1])
    plt.xlabel('Training time')
    plt.legend()

    # EWC
    plt.subplot(1,3,3)
    plt.title('EWC')
    for j in range(N_task):
        plt.plot(time_list, acc_EWC_lists[j], label = 'Task ' + str(j+1), c=colors[j])
    plt.ylim([0,1])
    plt.xlabel('Training time')
    plt.legend()

    plt.savefig(savename + '.png')
    plt.show()