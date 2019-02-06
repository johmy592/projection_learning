import numpy as np
from training import *
from evaluate import *
from load_glove import load_embeddings

def normalize_embeddings(embeddings):
    print("Normalizing word embeddings\n")
    for w in embeddings:
        embeddings[w] = embeddings[w] / np.sqrt((np.sum(embeddings[w]**2)))
    print("Done!\n")


def test():
    gf = "/home/johannes/thesis_code/ml_experimentation/glove.6B.50d.txt"
    em = load_embeddings(gf)
    normalize_embeddings(em)
    print("Len: ", np.sqrt((np.sum(em["this"]**2)))) 
