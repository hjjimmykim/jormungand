import numpy as np
import time

def train(sess, model, trainset, testsets, inputs, outputs, N_it, batch_size, type, lamb, ep_rec, ep_time):
    # sess = tensorflow session
    # model = network
    # trainset = dataset
    # testsets = datasets for different tasks
    # inputs = input placeholder
    # outputs = output placeholder
    # N_it = number of iterations (batches)
    # batch_size = size of batch
    # type = 0: Vanilla, 1: L2, 2: EWC
    # lamb = regularization parameter for L2 and EWC
    # ep_rec = record test accuracy every ep_rec iterations
    # ep_time = display runtime every ep_time iterations
    
    # test_acc_list = list containing test accuracies for each task
    
    N_task = len(testsets) # Number of tasks

    model.load_parameters(sess) # Restore optimal weights from previous session

    if type == 0:
        model.set_vanilla() # Vanilla
        name = 'Vanilla'
    elif type == 1:
        model.set_L2(lamb) # L2
        name = 'L2'
    elif type == 2:
        model.set_EWC(lamb) # EWC
        name = 'EWC'
        
    # Initialize test accuracy arrays
    test_acc_list = []
    for i in range(N_task):
        test_acc_list.append(np.zeros(int(N_it/ep_rec)))

    #print(name + ' training started...')
    time_start = time.time()
    time_p1 = time.time()
    for i_it in range(N_it):
        batch = trainset.train.next_batch(batch_size)
        model.step.run(feed_dict={inputs: batch[0], outputs: batch[1]})
        
        # Record test accuracies
        if i_it % ep_rec == 0:
            for i_task in range(N_task):
                feed_dict = {inputs: testsets[i_task].test.images, outputs:testsets[i_task].test.labels}
                test_acc_list[i_task][int(i_it/ep_rec)] = model.acc.eval(feed_dict=feed_dict)
                
        # Record time
        if i_it % ep_time == 0 and i_it != 0:
            time_p2 = time.time()
            print('Runtime for iterations ' + str(i_it-ep_time) + '-' + str(i_it) + ': ' + str(time_p2-time_p1) + ' s')
            time_p1 = time_p2
    time_finish = time.time()
    print(name +' training runtime: ' + str(time_finish-time_start) + ' s')
    #print(name + ' training finished...')

    return test_acc_list
    