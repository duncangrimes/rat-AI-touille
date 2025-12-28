# rat-AI-touille
AI book bot for you and me

Collection, filtering, preprocessing, tokenizing, encoding, embedding (with pretrained model)


Directory: 

    data/collection: 
    - collection_script collecton script from spoon API

    data/processing:
    
    
    data/storage:
    - stage_0   raw recipe data from Spoonacular API
    - stage_1   seperated out ingredients and equipment into their own dataframes
    - stage_1.5 dropped recipes with > 33 ingredients
    - stage_2   tokenized recipes using OpenAI TikToken
    - stage_3   all entries have equal number of tokens (appended 0's)

