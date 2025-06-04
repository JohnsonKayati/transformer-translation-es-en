import importlib
from translator import GPT, batch, load_data
import translator
importlib.reload(translator)
from translator import GPT, batch, load_data
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
import importlib
import itertools

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
tokenizer.add_special_tokens({'bos_token': '<sos>'})

# hyperparameters: 
vocab_size = len(tokenizer)
learning_rate = 2.4e-4
batch_size = 32
n_embd = 512
block_size = 128
num_heads = 8
n_blocks = 6
padding_token = tokenizer.pad_token_id
dropout = 0.1
max_iter = 50000
eval_iter = 250
large_batch_size = 10

train_reader = pq.ParquetFile(r"C:\Users\johns\OneDrive\Desktop\Programming\Python\AI\Projects\Translator\Dataset\Parquet\50M_train.parquet")
val_reader = pq.ParquetFile(r"C:\Users\johns\OneDrive\Desktop\Programming\Python\AI\Projects\Translator\Dataset\Parquet\val.parquet")
save_path = r'C:\Users\johns\OneDrive\Desktop\Programming\Python\AI\Projects\Translator\Saved_Models\GPT_translation_7.0.pt'
n_rows_train = 50_000_000
n_rows_val = 50_000

print("Loading data...")
train_data = load_data(train_reader, n_rows=n_rows_train)
val_data = load_data(val_reader, n_rows=n_rows_val)

print("Loading tokenized validation data...")
val_inputs, val_targets = torch.load(r"C:\Users\johns\OneDrive\Desktop\Programming\Python\AI\Projects\Translator\Dataset\tokenized_val_data.pt", weights_only=True)

val_dataset = TensorDataset(val_inputs, val_targets)

model = GPT(
    n_embd=n_embd,
    vocab_size=vocab_size, 
    n_head=num_heads, 
    n_blocks=n_blocks, 
    block_size=block_size,
    dropout=dropout
).to("cuda")

checkpoint = torch.load(
    r"C:\Users\johns\OneDrive\Desktop\Programming\Python\AI\Projects\Translator\Saved_Models\GPT_translation_5.0.pt",
    weights_only=True
)

model.load_state_dict(checkpoint, strict=True)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=1)
scaler = torch.GradScaler("cuda")

train_losses = []
val_losses = []
train_loss_total = []
val_loss_total = []

counter = 0
patience = 50
best_val_loss = float('inf')
model_saves = 1

print("starting training...")
for iter in range(max_iter + 1):
    xb, yb = batch('train', train_data, val_data, block_size, batch_size)
    mask = xb == tokenizer.pad_token_id
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type='cuda', enabled=True):
        _, train_loss = model(xb, yb)

    scaler.scale(train_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    train_losses.append(train_loss.item())

    if iter % eval_iter == 0:
        model.eval()
        val_losses = []
        with torch.no_grad():
            val_loader = DataLoader(val_dataset, batch_size=large_batch_size, shuffle=True)
            for xb_val, yb_val in itertools.islice(val_loader, eval_iter):
                xb_val, yb_val = xb_val.to('cuda'), yb_val.to('cuda')
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    _, val_loss = model(xb_val, yb_val)
                val_losses.append(val_loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        train_loss_total.append(avg_train_loss)
        val_loss_total.append(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Iteration: {iter}/{max_iter}, Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}, Learning rate: {current_lr:.1e}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model #{model_saves} saved!")
            model_saves += 1
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered.")
            break

        train_losses = []
        model.train()

print(f'Total Train Loss: {(sum(train_loss_total)) / len(train_loss_total):.4f}')
print(f'Total Test Loss: {(sum(val_loss_total)) / len(val_loss_total):.4f}')