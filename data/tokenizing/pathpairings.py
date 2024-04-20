import json

# 
with open('extracted_ingredients_list.json') as f:
    ingredients = json.load(f)

# pairing paths
pairing_paths = []
for i in range(len(ingredients)):
    for j in range(i+1, len(ingredients)):
        pairing_paths.append(f"{ingredients[i]} {ingredients[j]}\n")

#  pairing paths to a text file
with open('pairing_paths.txt', 'w') as f:
    f.writelines(pairing_paths)