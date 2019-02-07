import numpy as np
import random
from training import *
from evaluate import *
from load_glove import load_embeddings



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
