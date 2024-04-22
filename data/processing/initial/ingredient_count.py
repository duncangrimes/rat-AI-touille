import json
import pandas as pd

with open('data/storage/stage_1.5/ingredient_df.json', 'r') as file:
    data = json.load(file)
    recipes = pd.DataFrame(data)

    ingredients = set()

    # Iterate through each cell in the DataFrame
    for index, row in recipes.iterrows():
        columns = recipes.columns
        columns = columns.drop(['id'])
        
        for column in columns:
            if(pd.notnull(row[column])):
                ingredients.add(row[column])

    print(f'{len(ingredients)} unique ingredients found in the recipes')
    print(ingredients)

    with open('data/storage/stage_1.5/extracted_ingredients_list.json', 'w') as file:
        json.dump(list(ingredients), file)