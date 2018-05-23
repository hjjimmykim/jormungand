import tensorflow as tf
import numpy as np

class Net:
    def __init__(self, inputs, labels, hidden_dim=50):
        self.x = inputs # input placeholder
        
        input_dim = int(self.x.shape[1])
        output_dim = int(labels.shape[1])
    
        # Weight initializations
        W1 = tf.Variable(tf.truncated_normal([input_dim,hidden_dim],stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1,shape=[hidden_dim]))
        
        W2 = tf.Variable(tf.truncated_normal([hidden_dim,output_dim],stddev=0.1))
        b2 = tf.Variable(tf.constant(0.1,shape=[output_dim]))
        
        # Hidden layer
        h1 = tf.nn.relu(tf.matmul(self.x,W1) + b1)
        # Output layer
        self.outputs = tf.matmul(h1,W2) + b2
        
        # Keep track of parameters
        self.params = [W1,b1,W2,b2]
        
        # Vanilla loss (cross entropy)
        self.loss_vanilla = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=self.outputs))
        
        # Accuracy
        correct = tf.equal(tf.argmax(self.outputs,1), tf.argmax(labels,1))
        self.acc = tf.reduce_mean(tf.cast(correct,tf.float32))
        
    def set_vanilla(self, lr=0.1, optim_type=0):
        # Vanilla
        #self.loss = self.loss_vanilla
        if optim_type:
            self.optim = tf.train.AdamOptimizer(lr)
        else:
            self.optim = tf.train.GradientDescentOptimizer(lr)
        self.step = self.optim.minimize(self.loss_vanilla)
        
    def set_L2(self, lr=0.1, optim_type=0):
        self.loss_L2 = self.loss_vanilla
        if optim_type:
            self.optim = tf.train.AdamOptimizer(lr)
        else:
            self.optim = tf.train.GradientDescentOptimizer(lr)
        self.step = self.optim.minimize(self.loss_L2)
        
    def set_EWC(self, lr=0.1, optim_type=0):
        # Elastic weight consolidation
        self.loss_EWC = self.loss_vanilla
        if optim_type:
            self.optim = tf.train.AdamOptimizer(lr)
        else:
            self.optim = tf.train.GradientDescentOptimizer(lr)
        self.step = self.optim.minimize(self.loss_EWC)
        
    def update_L2(self, lambda_L2):
        # L2
        for i in range(len(self.params)):
            self.loss_L2 += (lambda_L2/2) * tf.reduce_sum(tf.square(self.params[i]-self.params_prev[i]))
        self.step = self.optim.minimize(self.loss_L2)
        #self.loss = self.loss_L2
       
    def update_EWC(self, lambda_EWC):
        # Summing up losses
        for i in range(len(self.params)):
            self.loss_EWC += (lambda_EWC/2) * tf.reduce_sum(tf.multiply(self.Fisher[i].astype(np.float32),tf.square(self.params[i]-self.params_prev[i])))
        self.step = self.optim.minimize(self.loss_EWC)
        #self.loss = self.loss_EWC
        
        
    def save_parameters(self):
        # Save parameters after a training run
        self.params_prev = []
        for i in range(len(self.params)):
            self.params_prev.append(self.params[i].eval())
            
    def load_parameters(self, sess):
        # Restore parameters for previous task
        if hasattr(self, "params_prev"):
            for i in range(len(self.params)):
                sess.run(self.params[i].assign(self.params_prev[i]))
                
    def set_Fisher(self):
        # Sample random class
        self.probs = tf.nn.softmax(self.outputs)
        self.output_sample = tf.to_int32(tf.multinomial(tf.log(self.probs),1)[0][0])
        self.log_L = tf.gradients(tf.log(self.probs[0,self.output_sample]),self.params)
                
    def compute_Fisher(self, data, sess, sample_size=200):
        # Compute Fisher Info. for EWC
        
        # Initialize
        self.Fisher = []
        for i in range(len(self.params)):
            self.Fisher.append(np.zeros(self.params[i].get_shape().as_list()))
        
        for i in range(sample_size):
            # Sample random data point
            i_data = np.random.randint(data.shape[0])
            # First derivative
            log_L_grad = sess.run(self.log_L, feed_dict={self.x: data[i_data:i_data+1]})
            # Squared and added
            for j in range(len(self.Fisher)):
                self.Fisher[j] += np.square(log_L_grad[j])
                
        # Average
        for i in range(len(self.Fisher)):
            self.Fisher[i] /= sample_size