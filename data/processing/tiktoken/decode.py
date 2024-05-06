import json
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

def load_tokenized_data(filepath: str):
    with open(filepath, 'r') as file:
        # Load the JSON data into a dictionary
        tokenized_ingredients = json.load(file)
    return tokenized_ingredients

def decode_ingreds_or_equip(filepath: str):
    tokenized_data = load_tokenized_data(filepath)
    
    decoded_data = dict()
    max = 0
    for key, tokens in tokenized_data.items():
        for token in tokens:
            if token > max:
                max = token
        # Decode the list of tokens into a string of ingredients
        decoded_text = encoding.decode(tokens)
        # Convert string to set of ingredients (if originally multiple ingredients)
        decoded_set = set(decoded_text.split(', '))  # Assuming space-separated ingredients
        decoded_data[key] = decoded_set
    print(max)
    return decoded_data

def decode_dishname(filepath: str):
    tokenized_dishnames = load_tokenized_data(filepath)

    decoded_dishnames = dict()
    for key, tokens in tokenized_dishnames.items():
        print(len(tokens))

        # Decode the list of tokens into a string of ingredients
        decoded_text = encoding.decode(tokens)
        decoded_dishnames[key] = decoded_text
    return decoded_dishnames

decode_ingreds_or_equip('data/storage/stage_3/tokenized_ingredients.json')