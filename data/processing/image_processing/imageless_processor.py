import json

def filter_recipes_without_url(input_json_file, output_json_file):
    with open(input_json_file, 'r') as f:
        recipes = json.load(f)

    filtered_recipes = [recipe for recipe in recipes if recipe.get('image') is not None]

    with open(output_json_file, 'w') as f:
        json.dump(filtered_recipes, f, indent=2)

    print(f"Filtered {len(recipes) - len(filtered_recipes)} recipes without URLs.")
    print(f"Filtered recipes saved to {output_json_file}.")

# Example usage:
input_json_file = '..//stage_1.5/recipes_df.json'
output_json_file = 'filtered_recipes.json'  # Specify the output file name
filter_recipes_without_url(input_json_file, output_json_file)
