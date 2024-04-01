import requests
import json

## Fill out the following variables:
# first_recipe: the id of the first recipe you want to get
# last_recipe: the id of the last recipe you want to get
# apiKey: your RapidAPI key
# filename: the name of the file you want to write the recipes to

## Check how many API calls you have made today @: https://rapidapi.com/developer/dashboard

first_recipe = 1457
last_recipe = 1650
apiKey = "bb5cf13bfcmsh842089eabd0407bp1d2db4jsn77597332fafe"
filename = "data/collection/recipes1250-2500.json"



headers = {
	"X-RapidAPI-Key": apiKey,
	"X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
}

filename = "data/collection/recipes1250-2500.json"

with open(filename, 'a') as file:
    for id in range(first_recipe, last_recipe+1):
        url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/{id}/information"


        response = requests.get(url, headers=headers)

        recipe_json = json.dumps(response.json(), indent=4)
        file.write(recipe_json + ",\n\n")