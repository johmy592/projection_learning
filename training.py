from helpers import sigmoid
from globals import *
import numpy as np
import random


def split_pos_neg(data):
    '''
    splits the training data into two sets, one with positive
    examples and one with negative examples
    '''
    pos_data = [(q,h,t) for q,h,t in data if t == 1]
    neg_data = [(q,h,t) for q,h,t in data if t == 0]
    return pos_data, neg_data

def select_batch(pos_data, neg_data, num_pos=32, num_neg=32*5):
    '''
    Selects a random batch from all training data
    '''
    training_batch = []
    negatives_to_add = []
    training_batch += random.sample(pos_data, num_pos)
    #for q,h,t in training_batch:
    #    for q_,h_,t_ in neg_data:
    #        if (q == q_).all():
    #            negatives_to_add += [(q_,h_,t_)]
    #training_batch += negatives_to_add
    if neg_data:
        training_batch += random.sample(neg_data, num_neg)
    return training_batch

def one_iter(examples, transforms, b, learning_rate=0.01):
    '''
    Runs one iteration through all provided examples
    '''
    #maxnorm = 0
    N = len(examples)
    total_loss = 0
    for q,h,t in examples:
        # project the query with all of the projection matrices

        p = [np.dot(transforms[i],q).T for i in range(k)]


        # Normalize the projections

        for i in range(k):
            p[i] = p[i] / np.sqrt((np.sum(p[i]**2)))
        #p = p/np.sqrt(np.sum(p**2))

        # compute similarities between all projections and the candidate
        s = np.dot(p,h)

        # Apply sigmoid function and find the transformation that resulted
        # in the closest projection to the candidate
        sigmoids = sigmoid(np.add(s,b))
        #sigmoids = sigmoid(s)
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
        loss = t*np.log(y) - (np.subtract(1,t)*np.log(np.subtract(1,y)))
        total_loss += -abs(loss)

        gradient = np.dot(x.T,loss)

        #gradient = np.dot(np.subtract(x,h))
        #if(t == 1):
        #    gradient = np.dot(x.T,np.subtract(x,h))/N
        #elif t == 0:
        #    gradient = np.dot(x.T, np.subtract(x,h))*((-1)/N)
        #gradnorm = np.linalg.norm(gradient,2)
        #if(gradnorm > maxnorm):
        #    maxnorm = gradnorm

        #print("Grad: ",gradient,"\n")
        #print("Grad*lr: ", np.multiply(learning_rate,gradient))
        prev_theta = transforms[maxsim]
        #print("Before: ", prev_theta)
        transforms[maxsim] = np.subtract(prev_theta, np.multiply(learning_rate,gradient))
        #print("OLD: ", prev_theta ,"\n", "NEW: ", transforms[maxsim],"\n")
        #print("After: ", transforms[maxsim],"\n")
        #Update the bias

        b_loss = min(1,max(0,y + b[maxsim])) - t
        prev_b = b[maxsim]
        b[maxsim] =  np.subtract(prev_b, np.multiply(learning_rate, b_loss))
    #print("Maxnorm: ", maxnorm, "\n")
    return total_loss


def train(embeddings, train_examples, num_iter=10, batching=False, learning_rate=0.01):
    '''
    The main training algorithm.
    '''
    rounds_per_batch = 8
    if batching:
        pos_examples,neg_examples = split_pos_neg(train_examples)

    transforms = [] #phi
    for i in range(k):
        transforms.append(np.identity(d) +np.random.normal(0, 0.5, [d,d]))
    b = np.zeros(k)  #bias

    for it in range(num_iter):
        if batching:
            train_examples = select_batch(pos_examples,neg_examples)
        for _ in range(rounds_per_batch):
            loss = one_iter(train_examples, transforms,b, learning_rate)
        print("Loss, iter: ", loss/len(train_examples), it, "\n")
    #print(b)
    return (transforms, b)
