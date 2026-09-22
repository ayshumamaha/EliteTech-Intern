Task 4: GPT-2 Text Generator
A simple text generation utility built on Hugging Face's transformers library using the pretrained GPT-2 model.

How it works
Loads the GPT-2 tokenizer and language model (gpt2).
Encodes a text prompt into token IDs.
Generates a continuation using sampling (top-k, top-p / nucleus sampling) with temperature control and repetition avoidance.
Decodes and returns the generated text.
Requirements
bash
pip install torch transformers

Usage
Run directly to see an example generation:
bash
python Task_4_Text_Generator.py
Or import the function in your own code:
python
from Task_4_Text_Generator import generate_text
output = generate_text("Once upon a time", max_length=150)
print(output)

Key parameters
max_length — maximum number of tokens in the generated output.
top_k, top_p, temperature — control randomness/creativity of the generated text.
no_repeat_ngram_size=2 — prevents repeating 2-grams for more natural output.

Notes
Both scripts download pretrained weights on first run (VGG19 and GPT-2), so an internet connection is required initially.
GPU is recommended for Task 3 (style transfer) for reasonable runtime; Task 4 runs fine on CPU for short generations.

Author
M. Ayshwarya
