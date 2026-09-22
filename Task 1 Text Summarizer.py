from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization")

def summarize_text(text, max_length=130, min_length=30):
    """
    Summarize the input text using a transformer-based model.
    :param text: str, input text to summarize
    :param max_length: int, maximum length of summary
    :param min_length: int, minimum length of summary
    :return: str, summarized text
    """
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Example usage
if __name__ == "__main__":
    article = (
        "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines "
        "that are programmed to think like humans and mimic their actions. The term may also be applied "
        "to any machine that exhibits traits associated with a human mind such as learning and problem-solving. "
        "AI is continuously evolving to benefit many different industries. Machines are wired using a cross-disciplinary "
        "approach based on mathematics, computer science, linguistics, psychology, and more."
    )

    print("Original Article:\n", article)
    print("\nSummarized Article:\n", summarize_text(article))
