# Import necessary libraries
import sys
import gensim.downloader as api
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

# Check if input file exists
if len(sys.argv) != 2:
    print("Usage: python train_model.py <input_file>")
    exit(1)

input_file = sys.argv[1]
if not input_file or not input_file.endswith('.txt'):
    print("Invalid input file path")
    exit(1)

# Load pre-generated pairing paths
with open(input_file, 'r') as f:
    lines = f.readlines()

# Train the Word2Vec model
model = Word2Vec([line.strip().split() for line in lines], size=128, window=5, min_count=1, workers=4)

# Save the trained model
model.save('flavorgraph2vec.model')

# Save the word vectors
vectors = model.wv
vectors.save('flavorgraph2vec.vec')


# TO run this :  python train_model.py pairing_paths.txt - this trains
 #a Word2Vec model using the pairing paths in pairing_paths.txt 
##and save the trained model and word vectors to files named 
#flavorgraph2vec.model and flavorgraph2vec.vec

