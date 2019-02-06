import numpy as np
import random
from training import *
from evaluate import *
from load_glove import load_embeddings

def normalize_embeddings(embeddings):
    print("Normalizing word embeddings\n")
    for w in embeddings:
        embeddings[w] = embeddings[w] / np.sqrt((np.sum(embeddings[w]**2)))
    print("Done!\n")


def replace_with_embedding(train_examples, embeddings, add_neg = 0):
    embedding_triples = []
    corresponding_words = []
    possible_negatives = list(embeddings.keys())[:1000]
    for q,h,t in train_examples:
        if((not q in embeddings) or (not h in embeddings)):
            continue
        embedding_triples += [(embeddings[q],embeddings[h],t)]
        embedding_triples += [(embeddings[q],embeddings[random.choice(possible_negatives)],0) for _ in range(add_neg)]
    print("Created ", len(embedding_triples), " training examples with embeddings\n")    
    return embedding_triples

def read_train_examples(data_file, gold_file, num_examples = 1500, num_neg=3):
    training_triples = []
    gf = open(gold_file, 'r')
    df = open(data_file, 'r')
    for i in range(num_examples):
        q = df.readline().split('\t')[0]
        h = [_h.strip('\n') for _h in gf.readline().split('\t')]
        training_triples += [(q,_h,1) for _h in h]
    print("Generated ", len(training_triples), "positive training examples\n")
    gf.close()
    df.close()
    return training_triples

def test_train():
    data_file = "/home/johannes/thesis_code/ml_experimentation/data/1A.english.training.data.txt"
    gold_file = "/home/johannes/thesis_code/ml_experimentation/data/1A.english.training.gold.txt"
    
    train_data = read_train_examples(data_file, gold_file)

def test():
    gf = "/home/johannes/thesis_code/ml_experimentation/glove.6B.50d.txt"
    em = load_embeddings(gf)
    normalize_embeddings(em)
    print("Len: ", np.sqrt((np.sum(em["this"]**2))))

    # Set some parameters for the model
    d = 50
    k = 24 
