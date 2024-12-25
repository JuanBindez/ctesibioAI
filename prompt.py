from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("./ctesibioAI-model")
tokenizer = GPT2Tokenizer.from_pretrained("./ctesibioAI-model")

tokenizer.pad_token = "<PAD>"
tokenizer.bos_token = "<BOS>"
tokenizer.eos_token = "<EOS>"

input_text = "<BOS>capital do brasil?"
inputs = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(
    inputs,
    max_length=50,
    num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id,  # Garantir consistência com o treinamento
    temperature=0.7,  # Controle de aleatoriedade
    top_k=50,         # Considerar apenas os 50 tokens mais prováveis
    repetition_penalty=2.0,  # Penalizar repetições
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Ctesibio-model Text response:")
print(generated_text)