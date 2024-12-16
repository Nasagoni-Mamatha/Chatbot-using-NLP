import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set the model to evaluation mode
model.eval()

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(prompt, max_length=100):
    # Encode the input text (prompt) into tokens
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate a response using the GPT-2 model
    outputs = model.generate(
        inputs, 
        max_length=max_length,         # Maximum length of the output
        num_return_sequences=1,        # Return only one response
        no_repeat_ngram_size=2,        # Prevent repetition
        top_p=0.9,                     # Top-p (nucleus sampling)
        top_k=50,                      # Top-k sampling
        temperature=0.7,                # Controls randomness
        do_sample=True                 # Use sampling instead of greedy decoding
    )
    
    # Decode the generated tokens into a string
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

def chat():
    print("Chatbot: Hello! How can I assist you today?")
    
    while True:
        # Take user input
        user_input = input("You: ")
        
        # Exit the chat loop if the user types "bye"
        if user_input.lower() == "bye":
            print("Chatbot: Goodbye! Have a great day!")
            break
        
        # Generate a response using GPT-2
        response = generate_response(user_input)
        print(f"Chatbot: {response}")

# Start the chatbot
chat()
