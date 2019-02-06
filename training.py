from helpers import sigmoid
import numpy as np


def one_iter(examples, transforms, b, learning_rate=0.1):
    '''
    Runs one iteration through all provided examples
    '''
    total_loss = 0
    for q,h,t in examples:
        # project the query with all of the projection matrices
        
        p = [np.dot(transforms[i],q).T for i in range(k)]
        
        
        # Normalize the projections
        for i in range(k):
            p[i] = p[i] / np.sqrt((np.sum(p[i]**2)))
        
        # compute similarities between all projections and the candidate
        s = np.dot(p,h)
        
        # Apply sigmoid function and find the transformation that resulted
        # in the closest projection to the candidate
        sigmoids = sigmoid(np.add(s,b))
        maxsim = np.argmax(sigmoids)
        
        # y: predicted similarity
        # x: the corresponding projected vector 
        y = sigmoids[maxsim]
        x = p[maxsim]
        
        if(y == 0 or y==1):
            print("Perfect hit")
            continue
        # Compute the loss and update the corresponding projection matrix
        # in accordance with gradient descent.
        loss = t*np.log(y) + (np.subtract(1,t)*np.log(np.subtract(1,y)))
        total_loss += loss
        
        gradient = np.dot(x.T,loss)
        
        #print("Grad: ",gradient,"\n")
        #print("Grad*lr: ", np.multiply(learning_rate,gradient))
        prev_theta = transforms[maxsim]
        #print("Before: ", prev_theta)
        transforms[maxsim] = np.subtract(prev_theta, np.multiply(learning_rate,gradient))
        #print("OLD: ", prev_theta ,"\n", "NEW: ", transforms[maxsim],"\n")
        #print("After: ", transforms[maxsim],"\n")
        #Update the bias
        b_loss = y + (b[maxsim]-t)
        prev_b = b[maxsim]
        b[maxsim] =  np.subtract(prev_b, np.multiply(learning_rate, b_loss))
    return total_loss
        

def train(embeddings, train_examples, num_iter=1000):
    '''
    The main training algorithm. 
    '''
    transforms = [] #phi
    for i in range(k):
        transforms.append(np.identity(d) +np.random.normal(0, 0.1, [d,d]))
    b = np.zeros(k)  #bias
    
    for it in range(num_iter):
        loss = one_iter(train_examples, transforms,b)
        print("Loss, iter: ", loss/len(train_examples), it, "\n")
    #print(b)
    return (transforms, b)
