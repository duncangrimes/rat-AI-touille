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
            # Collect all non-null ingredients into a set
            ingredients = set(value for key, value in row.items() if key.startswith('i_') and pd.notnull(value))
            # Assign the set to the corresponding ID in the dictionary
            recipes_dict[recipe_id] = ingredients
    
    return recipes_dict

def tokenize_ingredients(recipes_dict: dict):
    tokenized_ingredients = {}

    for recipe_id, ingredients in recipes_dict.items():
        encoding = tiktoken.get_encoding("cl100k_base")

        # Convert set of ingredients to a single string
        ingredients_text = ' '.join(ingredients)
        # Tokenize the text
        tokens = encoding.encode(ingredients_text)
        # Store tokens in dictionary using the recipe ID as key
        tokenized_ingredients[recipe_id] = tokens
    return tokenized_ingredients

def write_json(data: dict, filepath: str):
    with open(filepath, 'w') as file:
        json.dump(data, file)
    

recipes_dict = prepare_dictionaries('data/processing/stage_1.5/ingredient_df.json')
tokenized_ingredients = tokenize_ingredients(recipes_dict)
write_json(tokenized_ingredients, 'data/tiktoken/stage_2/tokenized_ingredients.json')