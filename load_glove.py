import numpy as np

def load_embeddings(glove_file):
    f = open(glove_file,'r')
    model = {}
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    f.close()
    return model
