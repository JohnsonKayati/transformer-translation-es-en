from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
import pyarrow.parquet as pq
import os

# Global Tokenizer Initialization
def init_worker():
    """
    Initialize the tokenizer globally for each worker.
    This avoids reloading the tokenizer repeatedly in each worker process.
    """
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")


# Tokenization Function
def tokenize_sentence(sentence):
    """
    Tokenize a single sentence using the globally initialized tokenizer.
    """
    return tokenizer(sentence)


# Parallel Tokenization
def parallel_tokenization(sentences, pool, chunk_size=10_000):
    """
    Perform tokenization in parallel using multiprocessing.
    Adjusts the number of workers and chunk size dynamically.
    """
    all_results = []
    for i in tqdm(range(0, len(sentences), chunk_size), desc="Processing in chunks"):
        chunk = sentences[i:i + chunk_size]
        results = list(
            tqdm(
                pool.imap(tokenize_sentence, chunk, chunksize=max(1, len(chunk) // (cpu_count() * 5))),
                total=len(chunk),
                desc=f"Tokenizing sentences (chunk {i // chunk_size + 1})",
            )
        )
        all_results.extend(results)
    return all_results


# Append Batch Directly to File
def append_batch_to_file(batch_data, file_path):
    """
    Append a batch of tokenized sentences directly to the .pt file without loading previous data.
    Args:
        batch_data (list): The current batch to save.
        file_path (str): The path to the .pt file.
    """
    mode = "ab" if os.path.exists(file_path) else "wb"
    with open(file_path, mode) as f:
        torch.save(batch_data, f)


# Main Execution
if __name__ == "__main__":
    # Path to the Parquet file
    parquet_file = r"C:\Users\johns\OneDrive\Desktop\Programming\Python\AI\Projects\Translator\Dataset\Parquet\50M_train.parquet"

    # Set batch size (number of rows per batch)
    batch_size = 500_000

    # File to save incrementally
    save_path = r"C:\Users\johns\OneDrive\Desktop\Programming\Python\AI\Projects\Translator\Dataset\Tokenized\engl_train_data.pt"

    # Maximum rows to process
    max_rows = 11_000_000
    total_processed_rows = 0

    # Open Parquet file
    print("Opening parquet file...")
    parquet_reader = pq.ParquetFile(parquet_file)

    # Calculate the total number of batches
    total_batches = parquet_reader.num_row_groups

    # Set up multiprocessing pool
    num_workers = max(1, int(cpu_count() * 0.5))
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        batch_index = 0

        # Initialize the progress bar for batches
        with tqdm(total=total_batches, desc="Processing batches") as batch_progress:
            for batch in parquet_reader.iter_batches(batch_size=batch_size):
                print(f"Processing batch {batch_index + 1}...")

                # Convert Arrow RecordBatch to a pandas DataFrame
                df = batch.to_pandas()

                # Extract English sentences (assuming English is in the first column)
                english_sentences = df.iloc[:, 0].tolist()

                # Check if processing this batch would exceed the max rows
                if total_processed_rows + len(english_sentences) > max_rows:
                    # Truncate the batch to fit within the max row limit
                    remaining_rows = max_rows - total_processed_rows
                    english_sentences = english_sentences[:remaining_rows]

                # Perform parallel tokenization on this batch
                print("Starting parallel tokenization for this batch...")
                final_tokens = parallel_tokenization(english_sentences, pool, chunk_size=10_000)

                # Append the current batch to the file
                append_batch_to_file(final_tokens, save_path)

                # Update the total processed rows
                total_processed_rows += len(english_sentences)

                # Update the progress bar
                batch_progress.update(1)

                batch_index += 1

                # Break the loop if the max row limit is reached
                if total_processed_rows >= max_rows:
                    print(f"Reached the maximum row limit of {max_rows}. Stopping processing.")
                    break

    print(f"All tokenized sentences (up to {max_rows} rows) saved incrementally to {save_path}!")