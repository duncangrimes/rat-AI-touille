from transformers import BertTokenizer

#from transformers imporcondt AutoTokenizer
##tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')



tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("This is an example of the bert tokenizer")
print(tokens)
# ['this', 'is', 'an', 'example', 'of', 'the', 'bert', 'token', '##izer']


import torch
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-uncased")

# get the embedding vector for the word "example"
example_token_id = tokenizer.convert_tokens_to_ids(["example"])[0]
example_embedding = model.embeddings.word_embeddings(torch.tensor([example_token_id]))

print(example_embedding.shape)
# torch.Size([1, 768])

