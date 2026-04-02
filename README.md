# rat-AI-touille

An experimental recipe generator that uses neural networks to create novel dish names, ingredient lists, equipment lists, and food images from scratch. Built as a group project for the Artificial Neural Networks & Deep Learning course at DIS Copenhagen (Spring 2024).

**[Read the full write-up on Medium](https://medium.com/@carmiles/rat-ai-touille-an-experimental-recipe-generator-fd7bc85389d9)**

## Overview

We built an end-to-end pipeline that collects 5,000+ recipes from the Spoonacular API, processes them through a multi-stage data pipeline, and trains several neural network architectures to generate new recipes:

- **Dish Name Generation** -- LSTM trained on tokenized recipe titles (~75% token prediction accuracy)
- **Ingredient & Equipment Prediction** -- Sequence-to-sequence LSTMs that map dish names to ingredient/equipment lists
- **Food Image Generation** -- Conditional DCGAN that generates 64x64 food images from dish name embeddings
- **Image Classification** -- DenseNet201 CNN fine-tuned on the Food-101 dataset

## Tech Stack

| Category | Tools |
|---|---|
| **Deep Learning** | TensorFlow / Keras, PyTorch |
| **Models** | LSTM, DCGAN (Conditional GAN), CNN (DenseNet201), Autoencoder |
| **Tokenization** | OpenAI TikToken (`cl100k_base`) |
| **Data** | Spoonacular API, Kaggle Food-101 |
| **Processing** | Pandas, NumPy, scikit-learn, OpenCV, Pillow |
| **Visualization** | Matplotlib |

## Pipeline

```
Spoonacular API --> Raw JSON (5k recipes)
       |
   Stage 0: Raw data
   Stage 1: Extract ingredients, equipment, instructions into separate DataFrames
   Stage 1.5: Filter recipes (drop > 33 ingredients)
   Stage 2: Tokenize with TikToken (dish names, ingredients, equipment)
   Stage 3: Pad sequences to uniform length (41 / 100 / 49 tokens)
       |
   Models: LSTM (text) | DCGAN (images) | Autoencoder (ingredients) | CNN (classification)
       |
   Generated: dish names, ingredient lists, equipment lists, food images
```

## Project Structure

```
rat-AI-touille/
├── data/
│   ├── collection/          # Spoonacular API scraping script
│   ├── processing/
│   │   ├── tiktoken/        # Tokenization & encoding scripts
│   │   └── image_processing/# Image download & preprocessing
│   └── storage/             # Staged data (raw -> tokenized -> padded)
│       ├── stage_0/         # Raw recipe JSON from API
│       ├── stage_1/         # Separated ingredients & equipment DataFrames
│       ├── stage_1.5/       # Filtered (<=33 ingredients)
│       ├── stage_2/         # Tokenized with OpenAI TikToken
│       └── stage_3/         # Padded to consistent token counts
├── models/
│   ├── lstm/                # Ingredient sequence prediction
│   ├── images/              # GAN & CNN for food images + generated samples
│   └── autoencoders/        # Ingredient autoencoder
└── dish_name/               # LSTM & GAN for dish name generation
```

## Authors

Carly Miles, Duncan Grimes, Blake Layman, Alexandra Szczerba

