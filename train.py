def train(model, trainset, testsets, lambda_L2, lambda_EWC, N_it, batch_size, ep_rec):
    model.restore(sess) # Restore optimal weights from previous session
    
    # Vanilla
    model.set_vanilla()
    
    # L2
    model.set_L2()
    
    # EWC
    model.set_EWC()
    
    for i_it in range(N_it):
        batch = trainset.train.next_batch(batch_size)
        