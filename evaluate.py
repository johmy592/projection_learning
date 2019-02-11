from helpers import sigmoid
import numpy as np
from globals import *


def extract(q, candidates, em, transforms, b, n=3):
    '''
    Extract the n top ranked hypernyms among candidates
    input:
    q: the embedding of the query
    candidates: a list of candidate hypernym strings
    em: dictionary containing all word add_embeddings
    transforms: projection matrices
    b: bias
    '''

    #print("Q: ", q,"\n")
    # p: All projections of q, list of k vectors
    p = [np.dot(transforms[i],q).T for i in range(k)]

    # Normalize all projections
    for i in range(k):
        p[i] = p[i] / np.sqrt((np.sum(p[i]**2)))
    #p = p/np.sqrt(np.sum(p**2))


    # s: similarities of all projections of q, with all other terms
    """
    Right now checks with all other embeddings (400k),
    takes a while, but manageable
    """
    
    s = [(np.dot(p,em[h]), h) for h in candidates]
    print(len(candidates))
    #s = [(np.dot(p,em[h]), h) for h in em]

    #print("Projections: ",p,"\n")
    #print("embeddings: ", embeddings,"\n")
    maxsims = [(_s[np.argmax(_s)], np.argmax(_s), h) for _s,h in s]
    #print(s)
    sigmoids = [(sigmoid(np.add(_s,b[idx])), h, idx) for _s,idx,h in maxsims]
    #sigmoids = [(sigmoid(_s), h, idx) for _s,idx,h in maxsims]
    sigmoids.sort()
    return sigmoids[-n:]
