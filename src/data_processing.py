from config import RAW_FILES, PROCESSED_FILES
import numpy as np
import random
import multiprocessing
from functools import partial
from src.non_ingredients import get_non_ingredients
from tqdm import tqdm
import os


def filter_split_ingredients(sorted_ingredients):
    if sorted_ingredients:
        return {
            i
            for i, next_i in zip(sorted_ingredients[:-1], sorted_ingredients[1:])
            if not next_i.startswith(i)
        } | {sorted_ingredients[-1]}
    else:
        return {}


def reverse_ingredients(ingredients):
    return {i[::-1] for i in ingredients}


def remove_split_ingredients(ingredients):
    ingredients = sorted(ingredients)
    ingredients = filter_split_ingredients(ingredients)

    ingredients = reverse_ingredients(ingredients)

    ingredients = sorted(ingredients)
    ingredients = filter_split_ingredients(ingredients)

    ingredients = reverse_ingredients(ingredients)

    return ingredients


def remove_non_ingredients(recipe_ingredients, non_ingredients):
    return {p for p in recipe_ingredients if p not in non_ingredients}


def load_raw_data():
    with np.load(RAW_FILES["simplified-recipes-1M.npz"], allow_pickle=True) as data:
        recipes = data["recipes"]
        ingredients = data["ingredients"]
    ingredients = [i.replace(" ", "_") for i in ingredients]

    return recipes, ingredients


def load_ingredients():
    recipes, ingredients = load_raw_data()
    non_ingredients = get_non_ingredients(ingredients)
    ingredients = [i for i in ingredients if i not in non_ingredients]
    return ingredients


def process_recipe(recipe, ingredients, non_ingredients):
    recipe_ingredients = [ingredients[i] for i in recipe]
    recipe_ingredients = remove_split_ingredients(recipe_ingredients)
    recipe_ingredients = remove_non_ingredients(recipe_ingredients, non_ingredients)

    if recipe_ingredients:
        return " ".join(recipe_ingredients) + "\n"


def create_processed_recipe_data(recipes, ingredients):
    with multiprocessing.Pool() as pool:
        non_ingredients = get_non_ingredients(ingredients)
        process_func = partial(
            process_recipe, ingredients=ingredients, non_ingredients=non_ingredients
        )
        results = pool.map(process_func, recipes)

        random.shuffle(results)

        train_size = 1040000
        with open(PROCESSED_FILES["recipes_train.txt"], "w") as f:
            f.writelines([r for r in results[:train_size] if r])

        with open(PROCESSED_FILES["recipes_val.txt"], "w") as f:
            f.writelines([r for r in results[train_size:] if r])


def shuffle_data():
    with open("data/processed/recipes_train.txt") as read_file:
        with open("data/processed/recipes_train.tmp", "w") as write_file:
            for line in read_file.readlines():
                ings = line.strip("\n").split(" ")
                random.shuffle(ings)
                write_file.write(" ".join(ings) + "\n")
    os.rename("data/processed/recipes_train.tmp", "data/processed/recipes_train.txt")
