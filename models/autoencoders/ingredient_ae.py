import json
import numpy as np

max_ingredient_token_size = 100163
dishname_token_count = 41
ingredient_token_count = 100

def load_data(dishname_filename, ingredient_filename):
    with open(f'data/storage/stage_3/{dishname_filename}', 'r') as file:
        data = json.load(file)
        dishnames = dict(data)

    with open(f'data/storage/stage_3/{ingredient_filename}', 'r') as file:
        data = json.load(file)
        ingredients = dict(data)

    return dishnames, ingredients

def prepare_data(dishname_filename, ingredient_filename, test_size=0.2, val_size=0.2):
    dishnames, ingredients = load_data(dishname_filename, ingredient_filename)

    dishname_array = np.zeros((len(dishnames), dishname_token_count))
    ingredient_array = np.zeros((len(ingredients), ingredient_token_count))

    # Normalize the ingredient tokens
    for (key, tokens) in ingredients.items():
        for token in tokens:
            token = token/1000000

    keys = sorted(dishnames.keys())
    for i, key in enumerate(keys):
        dishname_array[i] = dishnames[key]
        ingredient_array[i] = ingredients[key]
    
    return dishname_array, ingredient_array

