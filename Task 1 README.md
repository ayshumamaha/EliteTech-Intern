# Text Summarizer
## Overview
This project is a simple **Text Summarizer** built using Python and the Hugging Face Transformers library. It uses a transformer-based summarization pipeline to generate a shorter version of a given text while preserving its main information.

## Features
* Accepts text as input and generates a concise summary.
* Uses the `transformers` library and its built-in summarization pipeline.
* Allows the minimum and maximum summary lengths to be specified.
* Produces deterministic summaries by disabling sampling.
* Includes an example article about Artificial Intelligence.

## Requirements
Install the required Python library using:

## How It Works
The program creates a summarization pipeline using the Transformers library. The `summarize_text()` function accepts the input text along with optional `max_length` and `min_length` parameters. The text is then passed to the transformer model, which returns the generated summary.
By default, the maximum summary length is set to 130 and the minimum length is set to 30.

## Usage
Run the Python file:
The program displays both the original article and its summarized version.

## Example
The included example uses a short article describing **Artificial Intelligence (AI)** and prints the generated summary.

Author
M. Ayshwarya
