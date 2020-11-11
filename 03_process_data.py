from src.data_processing import create_processed_recipe_data, load_raw_data
from src.ingredient_tokenizer import create_vocab_file

if __name__ == "__main__":
    recipes, ingredients = load_raw_data()
    create_processed_recipe_data(recipes, ingredients)
    create_vocab_file(ingredients)
