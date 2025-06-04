import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
block_size = 128

val_reader = pq.ParquetFile(r"C:\Users\johns\OneDrive\Desktop\Programming\Python\AI\Projects\Translator\Dataset\Parquet\val.parquet")
n_rows_val = 100_000

def load_data(reader, n_rows):
    data = []
    loaded = 0
    for batch in reader.iter_batches(batch_size=1_000_000):
        chunk = batch.to_pandas()
        data.append(chunk)
        loaded += len(chunk)
        if loaded >= n_rows:
            break
    return pd.concat(data, ignore_index=True)

def encode_data(data, column):
    for text in data.iloc[:, column]:
        yield tokenizer.encode(text, max_length=block_size, padding="max_length", truncation=True)

print("Loading validation data...")
val_data = load_data(val_reader, n_rows=n_rows_val)

print("Tokenizing validation data...")
val_inputs = torch.tensor(list(encode_data(val_data, 0)))
val_targets = torch.tensor(list(encode_data(val_data, 1)))

print("Saving tokenized data to file...")
torch.save((val_inputs, val_targets), r"C:\Users\johns\OneDrive\Desktop\Programming\Python\AI\Projects\Translator\Dataset\tokenized_val_data.pt")

print("Tokenization and saving completed.")
