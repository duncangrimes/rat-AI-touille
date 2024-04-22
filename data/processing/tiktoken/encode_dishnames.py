import json
import pandas as pd
import tiktoken

def prepare_dictionaries(filepath: str):
    with open(filepath, 'r') as file:
        data = json.load(file)
        recipes = pd.DataFrame(data)
        recipes_dict = dict()
        
        for index, row in recipes.iterrows():
            # Extract the ID
            recipe_id = row['id']
            dishname = row["title"]
            # Assign the set to the corresponding ID in the dictionary
            recipes_dict[recipe_id] = dishname
    
    return recipes_dict

def tokenize_dishnames(recipes_dict: dict):
    tokenized_dishname = {}

    for recipe_id, dishname in recipes_dict.items():
        encoding = tiktoken.get_encoding("cl100k_base")

        # Tokenize the text
        tokens = encoding.encode(dishname)

        # Store tokens in dictionary using the recipe ID as key
        tokenized_dishname[recipe_id] = tokens
    return tokenized_dishname

def write_json(data: dict, filepath: str):
    with open(filepath, 'w') as file:
        json.dump(data, file)
    

recipes_dict = prepare_dictionaries('data/storage/stage_1.5/recipes_df.json')
tokenized_dishnames = tokenize_dishnames(recipes_dict)
write_json(tokenized_dishnames, 'data/storage/stage_2/tokenized_dishnames.json')

more_recipes_dict = prepare_dictionaries('data/storage/stage_1.5/more_recipes_df(NO_INSTRUCTS).json')
more_tokenized_dishnames = tokenize_dishnames(more_recipes_dict)
write_json(more_tokenized_dishnames, 'data/storage/stage_2/more_tokenized_dishnames(NO_INSTRUCTS).json')