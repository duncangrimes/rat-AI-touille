import json
import pandas as pd
def extract_ingredients_and_equipment(step):
    """
    Extract the ingredients and equipment used in a single step.
    """
    ingredients = [ingredient for ingredient in step['ingredients'] if ingredient['quantity']]
    equipment = [equipment for equipment in step['equipment'] if equipment]
    return ingredients, equipment

# Read the JSON file
#with open('recipes1250-2500.json') as f:
 #   recipes = json.load(f)


#access granting chmod u+w /Users/alexandraszczerba/NNfinal/rat-AI-touille/data/processing
file_path = "/.../rat-AI-touille/data/collection/recipes1250-2500.json"

with open(file_path, 'w') as f:
#with open(file_path, 'r') as file:
    data = json.load(f)
    recipes = pd.DataFrame(data)
# Preprocess the recipes
for recipe in recipes:
    for instruction in recipe['analyzedInstructions']:
        for step in instruction['steps']:
            ingredients, equipment = extract_ingredients_and_equipment(step)
            step['ingredients'] = [ingredient['name'] for ingredient in ingredients]
            step['equipment'] = [equipment['name'] for equipment in equipment]

# Write the preprocessed JSON back to a file
with open('check_preprocessed_recipes1250-2500.json', 'w') as f:
    json.dump(recipes, f, indent=4)
