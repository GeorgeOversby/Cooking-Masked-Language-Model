import json

from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase, Strip
from tokenizers.pre_tokenizers import CharDelimiterSplit
from src.non_ingredients import get_non_ingredients

from tokenizers import BertWordPieceTokenizer, Tokenizer
from tokenizers.implementations.base_tokenizer import BaseTokenizer
from tokenizers.processors import BertProcessing
from transformers import PreTrainedTokenizerFast
from pathlib import Path

unk_token = "<unk>"
vocab_file = "vocab.json"


class WhitespaceTokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab_file,
        sep_token="<sep>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        lowercase: bool = True,
    ):

        tokenizer = Tokenizer(WordLevel(vocab_file, unk_token=unk_token))
        tokenizer.normalizer = Strip()
        tokenizer.pre_tokenizer = CharDelimiterSplit(" ")

        tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=512)

        # Let the tokenizer know about special tokens if they are part of the vocab
        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])
        if tokenizer.token_to_id(str(sep_token)) is not None:
            tokenizer.add_special_tokens([str(sep_token)])
        if tokenizer.token_to_id(str(cls_token)) is not None:
            tokenizer.add_special_tokens([str(cls_token)])
        if tokenizer.token_to_id(str(pad_token)) is not None:
            tokenizer.add_special_tokens([str(pad_token)])
        if tokenizer.token_to_id(str(mask_token)) is not None:
            tokenizer.add_special_tokens([str(mask_token)])

        parameters = {
            "model": "WordLevel",
            "unk_token": unk_token,
            "sep_token": sep_token,
            "cls_token": cls_token,
            "pad_token": pad_token,
            "mask_token": mask_token,
            "lowercase": lowercase,
        }

        super().__init__(tokenizer, parameters)


def load_tokenizer(folder="."):
    folder = Path(folder)
    return PreTrainedTokenizerFast(
        WhitespaceTokenizer(str(folder / vocab_file)),
        pad_token="<pad>",
        mask_token="<mask>",
    )


def create_vocab_file(ingredients):
    special = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ]
    non_ingredients = get_non_ingredients(ingredients)
    ingredients = [i for i in ingredients if i not in non_ingredients]
    vocab_file = "artifacts/vocab.json"
    Path("artifacts").mkdir(exist_ok=True)
    with open(vocab_file, "w") as file:
        vocab = {ing: i + len(special) for i, ing in enumerate(ingredients)}
        for i, s in enumerate(special):
            vocab[s] = i

        json.dump(vocab, file)


def get_tokenizer_vocab(tokenizer):
    return tokenizer.convert_ids_to_tokens(range(len(tokenizer.get_vocab())))
