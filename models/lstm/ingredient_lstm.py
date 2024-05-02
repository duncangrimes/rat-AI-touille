from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, TimeDistributed, RepeatVector
from keras.optimizers import Adam
import json
import numpy as np
import tiktoken




def load_data(dishname_filename, ingredient_filename):
    with open(f'data/storage/stage_3/{dishname_filename}', 'r') as file:
        data = json.load(file)
        dishnames = dict(data)

    with open(f'data/storage/stage_3/{ingredient_filename}', 'r') as file:
        data = json.load(file)
        ingredients = dict(data)

    return dishnames, ingredients

def decode_ingredients(tokenized_ingredients):
    encoding = tiktoken.get_encoding("cl100k_base")
    decoded_text = encoding.decode(tokenized_ingredients)
    # Convert string to set of ingredients (if originally multiple ingredients)
    decoded_set = set(decoded_text.split(', '))
    return decoded_set

def prepare_data(dishname_filename, ingredient_filename, test_size=0.2, val_size=0.2):
    dishnames, ingredients = load_data(dishname_filename, ingredient_filename)

    dishname_array = np.zeros((len(dishnames), dishname_token_count))
    ingredient_array = np.zeros((len(ingredients), ingredient_token_count))

    # # Normalize the ingredient tokens
    # for (key, tokens) in ingredients.items():
    #     for token in tokens:
    #         token = token/1000000

    keys = sorted(dishnames.keys())
    for i, key in enumerate(keys):
        dishname_array[i] = dishnames[key]
        ingredient_array[i] = ingredients[key]
    
    return dishname_array, ingredient_array


dishname_token_count = 41
ingredient_token_count = 100
equipment_token_count = 49
max_ingredient_token_size = 100163

dishnames, ingredients = prepare_data('tokenized_dishnames.json', 'tokenized_ingredients.json')

vocab_size = 110000

embedding_dim = 64

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=dishname_token_count))
model.add(LSTM(256, return_sequences=False))
model.add(RepeatVector(100))
model.add(LSTM(256, return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))


model.compile(optimizer=Adam(learning_rate=.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(dishnames, np.expand_dims(ingredients, -1), epochs=10, batch_size=50)


# dishname: "Anchovy Fries with Smoked Paprika Aioli"

# tokenized dishname: [2127, 331, 62615, 435, 4108, 449, 4487, 11059, 32743, 41554, 57086, 14559, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# tokenized ingredients: [43326, 11, 30564, 64785, 11, 1253, 13767, 1082, 11, 7878, 34330, 12831, 11, 281, 93952, 11, 17685, 11, 7795, 11, 31735, 11, 17677, 5707, 11, 682, 7580, 20415, 11, 24522, 11, 274, 569, 1572, 11, 30564, 23661, 11, 16796, 14559, 11, 19151, 11, 13339, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

ex_dish = np.array([2127, 331, 62615, 435, 4108, 449, 4487, 11059, 32743, 41554, 57086, 14559, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

predicted_ingredients = model.predict(ex_dish)
print("predicted ingredients: ", predicted_ingredients)

print("decoded ingredients: ", decode_ingredients(predicted_ingredients))