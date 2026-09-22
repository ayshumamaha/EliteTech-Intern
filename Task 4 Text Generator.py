from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def generate_text(prompt, max_length=100):
    """
    Generate text from a given prompt using GPT-2.
    :param prompt: str, user input prompt
    :param max_length: int, max number of tokens in generated text
    :return: str, generated text
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=max_length, 
                                 num_return_sequences=1, 
                                 no_repeat_ngram_size=2, 
                                 do_sample=True, 
                                 top_k=50,
                                 top_p=0.95,
                                 temperature=0.8)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    user_prompt = "In the future, artificial intelligence will"
    generated = generate_text(user_prompt)
    print("Generated Text:\n")
    print(generated)
