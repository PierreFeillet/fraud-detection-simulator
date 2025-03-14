import tiktoken

# Load the tokenizer for Mistral (assuming it follows the GPT-4 tokenizer)
tokenizer = tiktoken.encoding_for_model("gpt-4")

# Sample Mistral-generated text (truncated for brevity)
file_path = "/Users/molocco/IBM_secondment/fraud-detection-simulator/LLMs_approach/outputs/develop_phase/llm_chain_of_thought_mistral-large.txt"
mistral_output = open(file_path, "r")
mistral_output = mistral_output.read()

# Tokenize the output
num_tokens = len(tokenizer.encode(mistral_output))

# Display the token count
print(num_tokens)
