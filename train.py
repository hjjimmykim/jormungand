import numpy as np
import time
import tensorflow as tf

def train(sess, model, trainset, testsets, inputs, labels, N_it, batch_size, ep_rec, ep_time, name, task=0):
    # sess = tensorflow session
    # model = network
    # trainset = dataset
    # testsets = datasets for different tasks
    # inputs = input placeholder
    # labels = label placeholder
    # N_it = number of iterations (batches)
    # batch_size = size of batch
    # ep_rec = record test accuracy every ep_rec iterations
    # ep_time = display runtime every ep_time iterations
    # name = name of the training type (for display)
    # task = for debugging
    
    # test_acc_list = list containing test accuracies for each task
    
    N_task = len(testsets) # Number of tasks

    # Initialize test accuracy arrays
    test_acc_list = []
    for i in range(N_task):
        test_acc_list.append(np.zeros(int(N_it/ep_rec)))

    #print(name + ' training started...')
    time_start = time.time()
    time_p1 = time.time()
    for i_it in range(N_it):
        batch = trainset.train.next_batch(batch_size)
        
        # Gradient descent with clipping
        #optim = tf.train.GradientDescentOptimizer(0.1)
        #gradients, variables = zip(*optim.compute_gradients(model.loss))
        #gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        #optimize = optim.apply_gradients(zip(gradients, variables))
        #optimize.run(feed_dict={inputs: batch[0], labels: batch[1]})
        
        model.step.run(feed_dict={inputs: batch[0], labels: batch[1]})
        
        '''
        # Debugging
        if hasattr(model,'loss_EWC') and task==4:
            
            
            # Model param
            a = 0
            for j in range(len(model.params)):
                a += model.params[j].eval().sum()
                
            # Loss
            loss_blah = model.loss_EWC.eval(feed_dict={inputs: trainset.train.images,labels:trainset.train.labels})
            print('-------------------')
            print('iter', i_it)
            print('param', a)
            print('loss', loss_blah)
            
            # Gradient
            print('Grads----')
            grads_and_vars=tf.train.GradientDescentOptimizer(0.1).compute_gradients(model.loss_EWC,model.params)
            for j in range(len(model.params)):
                print(grads_and_vars[j][0].eval(feed_dict={inputs: batch[0], labels: batch[1]}).sum())
            
            if a!= a:
                model.loss_EWC.eval(feed_dict={inputs: trainset.train.images,labels:trainset.train.labels})
                print('-----Param nan error--------')
                print('iter', i_it)
                print('param', a)
                print('loss', model.loss_EWC.eval(feed_dict={inputs: trainset.train.images,labels:trainset.train.labels}))
                print('----------------------------')
                break;
        '''
            
        
        # Record test accuracies
        if i_it % ep_rec == 0:
            for i_task in range(N_task):
                feed_dict = {inputs: testsets[i_task].test.images, labels:testsets[i_task].test.labels}
                test_acc_list[i_task][int(i_it/ep_rec)] = model.acc.eval(feed_dict=feed_dict)
                
        # Record time
        if i_it % ep_time == 0 and i_it != 0:
            time_p2 = time.time()
            print('Runtime for iterations ' + str(i_it-ep_time) + '-' + str(i_it) + ': ' + str(time_p2-time_p1) + ' s')
            time_p1 = time_p2
    time_finish = time.time()
    print(name +' training runtime: ' + str(time_finish-time_start) + ' s')
    #print(name + ' training finished...')

    return test_acc_list, batch
    