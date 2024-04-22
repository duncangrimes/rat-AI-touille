import json

def consistent_token_count(filename: str):
    with open(f'data/storage/stage_2/{filename}', 'r') as file:
        data = json.load(file)
        recipes = dict(data)
    max_tokens = 0
    for recipe_id, tokens in recipes.items():
        if len(tokens) > max_tokens:
            max_tokens = len(tokens)
    
    for recipe_id, tokens in recipes.items():
        if len(tokens) < max_tokens:
            for i in range(max_tokens - len(tokens)):
                tokens.append(0)

    with open(f'data/storage/stage_3/{filename}', 'w') as file:
        json.dump(recipes, file)

filenames = ['tokenized_dishnames.json', 'tokenized_ingredients.json', 'tokenized_equipment.json', 'more_tokenized_dishnames(NO_INSTRUCTS).json']

for filename in filenames:
    consistent_token_count(filename)