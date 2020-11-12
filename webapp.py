import streamlit as st
from src.ingredient_tokenizer import load_tokenizer, get_tokenizer_vocab
import tfidf_matcher as tm
from src.ingredient_tokenizer import load_tokenizer
from src.non_ingredients import get_non_ingredients
from transformers import pipeline
import tokenizers
from pathlib import Path
import subprocess

if not Path('artifacts').exists():
    subprocess.call("./download_artifacts.sh")


@st.cache(hash_funcs={tokenizers.Tokenizer: lambda _: None})
def load_model():
    tokenizer = load_tokenizer('artifacts/')

    return pipeline("fill-mask",
                         model='artifacts/',
                         tokenizer=tokenizer,
                         topk=500)


def normalise_recipe(recipe):
    ingredients = get_tokenizer_vocab(load_tokenizer("artifacts"))
    non_ingredients = get_non_ingredients(ingredients)
    ingredients = [i for i in ingredients if i not in non_ingredients]
    matched_recipe = tm.matcher(recipe, ingredients, k_matches=1)["Lookup 1"].to_list()
    return matched_recipe

CUMULATIVE_CUTOFF = st.sidebar.slider('cumulative  probability cut-off',0.0,1.0,0.6, 0.01)

fill_mask = load_model()
recipe_messy = st.text_area('Recipe:')
if recipe_messy:
    if recipe_messy.count(',') > recipe_messy.count('\n'):
        messy_ingredients = recipe_messy.split(',')
    else:
        messy_ingredients = recipe_messy.split('\n')
    messy_ingredients = [m.replace(' ', '_').replace('finely_chopped', '')
    for m in messy_ingredients if m]

    recipe_normalised = normalise_recipe(messy_ingredients)

    'Normalised recipe: '
    st.write(', '.join(recipe_normalised))
    st.write('\n')
    result = fill_mask(' '.join(recipe_normalised) + ' <mask>')
    cumulative_probability = 0
    for r in result:
        r['token_str'], r['score']
        cumulative_probability += r['score']
        if cumulative_probability > CUMULATIVE_CUTOFF:
            break
