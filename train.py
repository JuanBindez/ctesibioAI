from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

special_tokens = {"pad_token": "<PAD>", "bos_token": "<BOS>", "eos_token": "<EOS>"}
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

# datas example
data = [
    {"pergunta": "Qual é a capital do Brasil?", "resposta": "A capital do Brasil é Brasília."},
    {"pergunta": "Quem descobriu o Brasil?", "resposta": "O Brasil foi descoberto por Pedro Álvares Cabral."},
    {"pergunta": "Qual é a maior floresta tropical do mundo?", "resposta": "A maior floresta tropical do mundo é a Floresta Amazônica."},
]

formatted_data = [{"text": f"<BOS>{d['pergunta']} {d['resposta']}<EOS>"} for d in data]

dataset = Dataset.from_list(formatted_data)

def tokenize_function(example):
    encoding = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

tokenized_dataset = dataset.map(tokenize_function, batched=True)


training_args = TrainingArguments(
    output_dir="./ctesibioAI-model",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    num_train_epochs=50,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    report_to=[],  # Evita integração com wandb ou outros sistemas
    evaluation_strategy="no",  # Desabilita avaliação durante o treinamento
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# save model
trainer.train()
model.save_pretrained("./ctesibioAI-model")
tokenizer.save_pretrained("./ctesibioAI-model")