from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM


def create_trainer(tokenizer, model):
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="data/processed/recipes_train.txt",
        block_size=256,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    training_args = TrainingArguments(
        output_dir="./artifacts",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_gpu_train_batch_size=128,
        save_steps=100_000_000,
        save_total_limit=2,
        fp16=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=True,
    )
    return trainer


def create_model():
    config = RobertaConfig(
        vocab_size=3437,
        max_position_embeddings=64,
        num_attention_heads=12,
        num_hidden_layers=8,
        type_vocab_size=1,
    )

    return RobertaForMaskedLM(config=config)
