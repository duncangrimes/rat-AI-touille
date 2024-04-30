# keras module for building LSTM 
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 

# set seeds for reproducability
from tensorflow import set_random_seed
from numpy.random import seed
set_random_seed(2)
seed(1)

import pandas as pd
import numpy as np
import string, os 
import json

def get_sequence_of_tokens(corpus, tokenizer):
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words


# Load and preprocess dataset
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    recipes = pd.DataFrame(data)
    # Tokenize titles
    # embedded_titles = recipes['title'].map(lambda title : simple_encode(title))
    recipes['dishTypes'] = recipes['dishTypes'].map(lambda x : x[0] if len(x) > 0 else '')
    
    return recipes[['title', 'dishTypes']]


def main():
    tokenizer = Tokenizer()
    file_path = '/Users/carlymiles/Desktop/EECS/DIS-NN/rat-AI-touille/data/processing/stage_1.5/recipes_df.json'
    corpus = load_data(file_path)
    
    inp_sequences, total_words = get_sequence_of_tokens(corpus, tokenizer)
    
if __name__ == '__main__':
    main()