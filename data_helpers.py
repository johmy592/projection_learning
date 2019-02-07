import numpy as np
import random


def add_embeddings(words, embeddings):
    '''
    input: list of strings
    output: list of (string, embedding) pais
    '''
    pairs = []
    for w in words:
        if w not in embeddings:
            continue
        pairs += [(w,embeddings[w])]
    print("Generated ", len(pairs), " test examples with embeddings")
    return pairs

def read_test_examples(data_file, num_examples=1500):
    '''
    Reads in a set of strings that can be use for testing purposes
    '''
    df = open(data_file,'r')
    test_examples = []
    for line in df:
        test_examples += line.split('\t')
    test_examples = list(set(test_examples))
    test_examples = [w.strip('\n') for w in test_examples]
    df.close()
    print("Generated ", len(test_examples), " words for testing")
    return test_examples

def normalize_embeddings(embeddings):
    '''
    Normalizes all embeddings to unit length
    '''
    print("Normalizing word embeddings\n")
    for w in embeddings:
        embeddings[w] = embeddings[w] / np.sqrt((np.sum(embeddings[w]**2)))
    print("Done!\n")


def replace_with_embedding(train_examples, embeddings, add_neg = 0):
    '''
    input:
    list of (string,string,int) triples, and a dictionary
    wit all word embeddings.
    output:
    list of (embedding,embedding,int) triples
    '''
    embedding_triples = []
    corresponding_words = []
    possible_negatives = list(embeddings.keys())
    for q,h,t in train_examples:
        if((not q in embeddings) or (not h in embeddings)):
            continue
        embedding_triples += [(embeddings[q],embeddings[h],t)]
        embedding_triples += [(embeddings[q],embeddings[random.choice(possible_negatives)],0) for _ in range(add_neg)]
    print("Created ", len(embedding_triples), " training examples with embeddings\n")
    return embedding_triples

def read_train_examples(data_file, gold_file, num_examples = 1500, num_neg=3):
    '''
    Reads a set of training examples from text file.
    '''
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
