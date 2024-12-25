from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./ctesibioAI-model")
tokenizer = GPT2Tokenizer.from_pretrained("./ctesibioAI-model")

# Configure special tokens
tokenizer.pad_token = "<PAD>"
tokenizer.bos_token = "<BOS>"
tokenizer.eos_token = "<EOS>"

# Initialize the conversation history
history = "<BOS>"

print("Ctesibio AI - Interactive Chatbot Test")
print("Type 'exit' to end the conversation.\n")

while True:
    # User input
    user_input = input("You: ")
    if user_input.lower() in ["exit"]:
        print("Ending the conversation. Goodbye!")
        break
    
    # Update the history with the user's input
    history += f"{user_input}<EOS>"
    
    # Tokenize the history
    inputs = tokenizer.encode(history, return_tensors="pt")

    # Generate the model's response
    outputs = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,  # Ensure consistency with training
        temperature=0.7,  # Control randomness
        top_k=50,         # Consider only the top 50 most likely tokens
        repetition_penalty=2.0,  # Penalize repetitions
    )
    
    # Decode and display the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print("Ctesibio:", response)

    # Update the history with the model's response
    history += f"{response}<EOS>"
