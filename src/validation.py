import numpy as np
from transformers import pipeline
from tqdm import tqdm
from more_itertools import chunked


def validate(model, tokenizer, ks=[1], device=0, limit=0, relative_base_path=""):
    fill_mask = pipeline(
        "fill-mask", model=model, tokenizer=tokenizer, topk=max(ks), device=device
    )

    data = []
    actuals = []
    with open(relative_base_path + "data/processed/recipes_val.txt") as f:
        for ri, ing in enumerate(f.readlines()):
            if limit and ri > limit:
                break
            for i in range(ing.count(" ") + 1):
                ing_list = ing.split(" ")
                actual = ing_list[i]
                ing_list[i] = "<mask>"
                masked_ingredients = " ".join(ing_list)
                data.append(masked_ingredients)
                actuals.append(actual)

    outputs = []
    batch_size = 128
    for chunk in tqdm(chunked(data, batch_size), total=len(data) // batch_size):
        outputs += [[g["token_str"] for g in o] for o in fill_mask(chunk)]

    results = [g.index(a) if a in g else max(ks) + 1 for g, a in zip(outputs, actuals)]
    results = np.array(results)
    return [(results < k).sum() / len(results) for k in ks]
