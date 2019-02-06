from helpers import sigmoid
import numpy as np

def extract(q, transforms, embeddings, b, n=3):
    '''
    Extract the n top ranked candidates
    '''
    
    #print("Q: ", q,"\n")
    # p: All projections of q, list of k vectors
    p = [np.dot(transforms[i],q).T for i in range(k)]
    
    # Normalize all projections
    for i in range(k):
        p[i] = p[i] / np.sqrt((np.sum(p[i]**2)))
    
    # s: similarities of all projections of q, with all other terms
    s = [(np.dot(p,embeddings[h]), h) for h in embeddings]
    
    #print("Projections: ",p,"\n")
    #print("embeddings: ", embeddings,"\n")
    maxsims = [(_s[np.argmax(_s)], np.argmax(_s), h) for _s,h in s]
    #print(s)
    sigmoids = [(sigmoid(np.add(_s,b[idx])), h) for _s,idx,h in maxsims]
    sigmoids.sort()
    return sigmoids[-n:]

