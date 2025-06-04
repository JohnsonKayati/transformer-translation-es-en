import torch
import torch.nn.functional as F
import Train.translator
from transformers import AutoTokenizer
from transformers import LogitsProcessorList, BeamSearchScorer
import importlib
importlib.reload(Train.translator)
from Train.translator import GPT
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
tokenizer.add_special_tokens({'bos_token': '<sos>'})
# Hyperparameters
vocab_size = len(tokenizer)
n_embd = 512
block_size = 128
num_heads = 8
n_blocks = 6
padding_token = tokenizer.pad_token_id
dropout = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"
user = ""

checkpoint = torch.load(
    r"C:\Users\johns\OneDrive\Desktop\Programming\Python\AI\Projects\Translator\Saved_Models\GPT_translation_7.0.pt",
    weights_only=True
)
model = GPT(
    n_embd=n_embd,
    vocab_size=vocab_size,
    n_head=num_heads,
    n_blocks=n_blocks,
    block_size=block_size,
    dropout=dropout
).to(device)

model.load_state_dict(checkpoint, strict=True)
model.eval()

def convert_to_tensor(input_str):
    tokenized_str = tokenizer.encode(input_str, max_length=block_size, padding='max_length', truncation=True)
    tensor_data = torch.tensor([tokenized_str]).to(device)
    return tensor_data

user = input("Enter a sentence: ").lower()

while user.lower() != "stop":
    tokenized_input_engl = convert_to_tensor(user)
    spanish_generate = model.generate(tokenized_input_engl)
    print(spanish_generate)
    user = input("Enter a sentence: ").lower()