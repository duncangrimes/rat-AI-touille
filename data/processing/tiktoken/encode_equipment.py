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
            equipment = set(value for key, value in row.items() if key.startswith('e_') and pd.notnull(value))
            # Assign the set to the corresponding ID in the dictionary
            recipes_dict[recipe_id] = equipment
    
    return recipes_dict

def tokenize_equipment(recipes_dict: dict):
    tokenized_equipment = {}

    for recipe_id, equipment in recipes_dict.items():
        encoding = tiktoken.get_encoding("cl100k_base")

        # Convert set of equipment to a single string
        equipment_text = ' '.join(equipment)
        # Tokenize the text
        tokens = encoding.encode(equipment_text)
        # Store tokens in dictionary using the recipe ID as key
        tokenized_equipment[recipe_id] = tokens
    return tokenized_equipment

def write_json(data: dict, filepath: str):
    with open(filepath, 'w') as file:
        json.dump(data, file)
    

recipes_dict = prepare_dictionaries('data/storage/stage_1.5/equipment_df.json')
tokenized_equipment = tokenize_equipment(recipes_dict)
write_json(tokenized_equipment, 'data/storage/stage_2/tokenized_equipment.json')