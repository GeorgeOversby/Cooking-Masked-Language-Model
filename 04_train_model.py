import os


from src.non_ingredients import get_non_ingredients

from src.ingredient_tokenizer import load_tokenizer, get_tokenizer_vocab
from src.data_processing import (
    create_processed_recipe_data,
    load_raw_data,
    shuffle_data,
)

from gensim import models
import random
from tqdm import tqdm
import numpy as np
import neptune
from pathlib import Path

from src.trainer import create_trainer, create_model
from src.validation import validate

tokenizer = load_tokenizer("artifacts")

model = create_model()


neptune.init("oversbyg/cook-mlm")
neptune.create_experiment(name="example")


trainer = create_trainer(tokenizer, model)

for epoch in range(40):
    validation_result = validate(model, tokenizer)[0]

    neptune.log_metric("top1", validation_result)
    trainer.save_model("./artifacts")

    trainer.train()

    shuffle_data()
    trainer = create_trainer(tokenizer, model)


for file in Path("./artifacts").iterdir():
    neptune.log_artifact(str(file))
