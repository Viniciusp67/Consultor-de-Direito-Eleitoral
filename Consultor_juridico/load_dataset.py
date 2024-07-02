from transformers import DistilBertTokenizer
from datasets import Dataset

with open('Compêndio-Eleitoral_2024_Legislação-Eleitoral-e-Normativos-TREPR.txt', 'r', encoding='utf-8') as file:
    text = file.read()


def split_long_text(text, tokenizer, max_length=512):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        token_length = len(tokenizer.tokenize(word))
        if current_length + token_length > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = token_length
        else:
            current_chunk.append(word)
            current_length += token_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
chunks = split_long_text(text, tokenizer)

dataset = Dataset.from_dict({
    'text': chunks,
    'label': [0 if i % 2 == 0 else 1 for i in range(len(chunks))]
})

unique_labels = set(dataset['label'])
print(f"Labels únicos no dataset: {unique_labels}")

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
