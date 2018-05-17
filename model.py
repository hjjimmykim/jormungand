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
        self.set_vanilla()
        
        # Accuracy
        correct = tf.equal(tf.argmax(self.outputs,1), tf.argmax(labels,1))
        self.acc = tf.reduce_mean(tf.cast(correct,tf.float32))
        
    def set_vanilla(self):
        # Vanilla
        self.step = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss_vanilla)
        
    def set_L2(self, lambda_L2):
        # L2
        if not hasattr(self, "loss_L2"):
            self.loss_L2 = self.loss_vanilla
        else:
            for i in range(len(self.params)):
                self.loss_L2 += (lambda_L2/2) * tf.reduce_sum(tf.square(self.params[i]-self.params_prev[i]))
        self.step = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss_L2)
        
    def set_EWC(self, lambda_EWC):
        # Elastic weight consolidation
        if not hasattr(self, "loss_EWC"):
            self.loss_EWC = self.loss_vanilla
        else:
            print('blah_EWC')
            # Summing up losses
            for i in range(len(self.params)):
                self.loss_EWC += (lambda_EWC/2) * tf.reduce_sum(tf.multiply(self.Fisher[i].astype(np.float32),tf.square(self.params[i]-self.params_prev[i])))
        self.step = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss_EWC)
        
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
                
    def compute_Fisher(self, data, sess, sample_size=200):
        # Compute Fisher Info. for EWC
        
        # Initialize
        self.Fisher = []
        for i in range(len(self.params)):
            self.Fisher.append(np.zeros(self.params[i].get_shape().as_list()))
            
        # Sample random class
        probs = tf.nn.softmax(self.outputs)
        output_sample = tf.to_int32(tf.multinomial(tf.log(probs),1)[0][0])
        
        for i in range(sample_size):
            # Sample random data point
            i_data = np.random.randint(data.shape[0])
            # First derivative
            log_L_grad = sess.run(tf.gradients(tf.log(probs[0,output_sample]),self.params), feed_dict={self.x: data[i_data:i_data+1]})
            # Squared and added
            for j in range(len(self.Fisher)):
                self.Fisher[j] += np.square(log_L_grad[j])
                
        # Average
        for i in range(len(self.Fisher)):
            self.Fisher[i] /= sample_size