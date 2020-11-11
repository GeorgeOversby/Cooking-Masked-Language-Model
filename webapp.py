import streamlit as st
from src.ingredient_tokenizer import load_tokenizer, get_tokenizer_vocab
import tfidf_matcher as tm
from src.ingredient_tokenizer import load_tokenizer
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
    matched_recipe = tm.matcher(recipe, ingredients, k_matches=1)["Lookup 1"].to_list()
    return matched_recipe

CUMULATIVE_CUTOFF = st.sidebar.slider('cumulative  probability cut-off',0.0,1.0,0.6, 0.01)

fill_mask = load_model()
recipe_messy = st.text_area('Recipe:')
if recipe_messy:
    recipe_normalised = normalise_recipe(recipe_messy.split(','))
    'Normalised recipe: ', ', '.join(recipe_normalised)
    st.write('\n')
    result = fill_mask(' '.join(recipe_normalised) + ' <mask>')
    cumulative_probability = 0
    for r in result:
        r['token_str'], r['score']
        cumulative_probability += r['score']
        if cumulative_probability > CUMULATIVE_CUTOFF:
            break
