# Number of samples to display per task
batch_size_sample = 4

# Train dataset loaders
trainloaders_sample = []
for i in range(N_task):
    trainloaders_sample.append(torch.utils.data.DataLoader(trainsets[i], batch_size=batch_size_sample, shuffle=True,num_workers=2))

# Show image
def imshow(img):
    img = img/2 + 0.5   # Unnormalize ([-1,1] -> [0,1])
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    
for i in range(N_task):
    trainloader_sample = trainloaders_sample[i]
    
    dataiter = iter(trainloader_sample)     # Convert to iterator
    images, labels = dataiter.next()        # Get next minibatch
    
    imshow(torchvision.utils.make_grid(images))
    plt.show()
    
    # Show labels
    print(' '.join('%10s' % classes[labels[j]] for j in range(batch_size_sample)))