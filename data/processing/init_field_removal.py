import json
import pandas as pd

file_path = "data/collection/recipes1250-2500.json"

with open(file_path, 'r') as file:
    data = json.load(file)
    recipes = pd.DataFrame(data)


print(recipes.columns)
print(type(recipes.loc[0,'extendedIngredients']))

ingredients = recipes.loc[0,'extendedIngredients']
print(type(ingredients[0]))

# for index in range(10):
#     recipe = recipes[index]
#     print(recipe) 