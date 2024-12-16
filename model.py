# Load pre-trained GPT-2 model and tokenizer from Hugging Face
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set the model to evaluation mode
model.eval()

# Check if CUDA is available for GPU use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
