import pyarrow.parquet as pq
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
tokenizer.add_special_tokens({'bos_token': '<sos>'})

def load_data(reader, n_rows):
    train_data = []
    loaded = 0
    for batch in reader.iter_batches(batch_size=1_000_000):
        chunk = batch.to_pandas()
        train_data.append(chunk)
        loaded += len(chunk)
        if loaded >= n_rows:
            break
    
    return pd.concat(train_data, ignore_index=True)

def batch(split, train_data, val_data, max_len, batch_size):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(0, len(data), (batch_size, ))
    X = []
    Y = []
    for i in idxs:
        input_seq = tokenizer.encode(data.iloc[i.item(), 0], max_length=max_len - 1, padding='max_length', truncation=True)
        input_seq = [tokenizer.bos_token_id] + input_seq
        X.append(input_seq)

        target_seq = tokenizer.encode(data.iloc[i.item(), 1], max_length=max_len - 1, padding='max_length', truncation=True)
        target_seq = [tokenizer.bos_token_id] + target_seq
        Y.append(target_seq)

    X = torch.tensor(X).to('cuda')
    Y = torch.tensor(Y).to('cuda')
    return X, Y

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding size must be divisible by number of heads"

        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.q = nn.Linear(n_embd, n_embd, bias=False)
        self.k = nn.Linear(n_embd, n_embd, bias=False)
        self.v = nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask, causal_mask):
        B, T, C = query.shape

        q = self.q(query).view(B, -1, self.n_head, self.head_size).transpose(1, 2)
        k = self.k(key).view(B, -1, self.n_head, self.head_size).transpose(1, 2)
        v = self.v(value).view(B, -1, self.n_head, self.head_size).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        if causal_mask:
            causal = torch.tril(torch.ones(T, T, device=query.device)).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
            
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.out_proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.ffwd(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, is_decoder=False):
        super().__init__()
        if is_decoder:
            self.cross_attention = MultiHeadAttention(n_embd=n_embd, n_head=n_head, dropout=dropout)
        self.self_attention = MultiHeadAttention(n_embd=n_embd, n_head=n_head, dropout=dropout)
        self.layerNorm1 = nn.LayerNorm(n_embd)
        self.layerNorm2 = nn.LayerNorm(n_embd)
        self.layerNorm3 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd=n_embd, dropout=dropout)
        self.is_decoder = is_decoder

    def forward(self, x, padding_mask, encoder_output=None):       
        norm_x = self.layerNorm1(x)
        if self.is_decoder:
            expanded_padding_mask = torch.tril(
                torch.ones((norm_x.size(1), norm_x.size(1)), device=norm_x.device)
            ).unsqueeze(0).unsqueeze(0)
        else:
            expanded_padding_mask = padding_mask.expand(-1, -1, norm_x.size(1), -1)
      
        sa_out = self.self_attention(norm_x, norm_x, norm_x, mask=expanded_padding_mask, causal_mask=self.is_decoder)
        x = x + sa_out
        if self.is_decoder and encoder_output is not None:
            norm_x = self.layerNorm2(x)
            x = x + self.cross_attention(norm_x, encoder_output, encoder_output, mask=padding_mask, causal_mask=False)

        x = x + self.ffwd(self.layerNorm3(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_blocks, block_size, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)
        self.target_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)
        self.encoder_pos_table = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embd)
        self.decoder_pos_table = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embd)
        self.decoder_blocks = nn.ModuleList([Block(n_embd=n_embd, n_head=n_head, dropout=dropout, is_decoder=True) for _ in range(n_blocks)])
        self.encoder_blocks = nn.ModuleList([Block(n_embd=n_embd, n_head=n_head, dropout=dropout, is_decoder=False) for _ in range(n_blocks)])
        self.lin_proj = nn.Linear(n_embd, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size
        
    def forward(self, x, y):
        encoder_attention_mask = (x != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
        decoder_attention_mask = (y != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(y.size(1), y.size(1), device=y.device)).unsqueeze(0).unsqueeze(0)
        decoder_attention_mask = decoder_attention_mask.logical_and(causal_mask)

        tok_emb = self.token_embedding_table(x)
        pos_indxs = torch.arange(x.size(1), device=x.device)
        pos_emb = self.encoder_pos_table(pos_indxs)
        encoder_output = self.dropout(tok_emb + pos_emb)
        
        target_emb = self.target_embedding_table(y)
        pos_idxs_dec = torch.arange(y.size(1), device=y.device)
        pos_emb_dec = self.decoder_pos_table(pos_idxs_dec)
        decoder_output = self.dropout(target_emb + pos_emb_dec)
        
        for block in self.encoder_blocks:
            encoder_output = block(encoder_output, encoder_attention_mask)
        for block in self.decoder_blocks:
            decoder_output = block(decoder_output, decoder_attention_mask, encoder_output=encoder_output)

        logits = self.lin_proj(decoder_output)

        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)), 
            y[:, 1:].contiguous().view(-1), 
            ignore_index=tokenizer.pad_token_id,
        )
        return logits, loss
    
    def sample_next_token(logits, temperature=1.0, top_k=0):
        # Apply temperature scaling
        logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            values, indices = torch.topk(logits, top_k)
            min_values = values[:, -1].unsqueeze(1)
            logits[logits < min_values] = -float('Inf')  # Mask low probability values
        
        # Convert logits to probabilities and sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        return next_token

    def generate(self, x, max_len=50, temperature=1.0, top_k=0):
        self.eval()
        pos_idxs = torch.arange(x.size(1), device=x.device)
        tok_emb = self.token_embedding_table(x)
        pos_emb = self.encoder_pos_table(pos_idxs)
        encoder_output = self.dropout(tok_emb + pos_emb)
        encoder_attention_mask = (x != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2).expand(-1, -1, -1, x.size(1))

        for i, block in enumerate(self.encoder_blocks):
            encoder_output = block(encoder_output, encoder_attention_mask)

        generated = torch.tensor([[tokenizer.bos_token_id]], device=x.device)

        for step in range(max_len):
            pos_idxs = torch.arange(generated.size(1), device=x.device)

            if generated.size(1) >= self.block_size:
                break

            target_emb = self.target_embedding_table(generated)
            pos_emb = self.decoder_pos_table(pos_idxs)
            decoder_output = self.dropout(target_emb + pos_emb)

            seq_len = generated.size(1)
            decoder_attention_mask = torch.tril(
                torch.ones((seq_len, seq_len), device=x.device)
            ).unsqueeze(0).unsqueeze(0)

            for i, block in enumerate(self.decoder_blocks):
                decoder_output = block(
                    decoder_output,
                    encoder_attention_mask.expand(-1, -1, decoder_output.size(1), -1),
                    encoder_output=encoder_output
                )

            logits = self.lin_proj(decoder_output)
            next_token = GPT.sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k)
            #print(next_token)
            generated = torch.cat((generated, next_token), dim=-1)
            if (next_token == tokenizer.eos_token_id).any():
                break

        self.train()
        output_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        return output_text
        

def get_batch(split, train_data, val_data, max_len):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(0, len(data), (32, ))
    X = torch.tensor([tokenizer.encode(data.iloc[i.item(), 0], max_length=max_len, padding='max_length', truncation=True) for i in idxs])
    Y = torch.tensor([tokenizer.encode(data.iloc[i.item(), 1], max_length=max_len, padding='max_length', truncation=True) for i in idxs])
    X = X.to('cuda')
    Y = Y.to('cuda')
    return X, Y

@torch.no_grad()
def est_loss(model, eval_iters):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    return out

# vocab_size = tokenizer.vocab_size
# learning_rate = 3e-4
# batch_size = 32
# n_embd = 512
# block_size = 128
# num_heads = 8
# n_blocks = 6
# padding_token = tokenizer.pad_token_id
# dropout = 0.1
# max_iter = 10000
# eval_iter = 250
# large_batch_size = 10

# model = GPT(
#     n_embd=n_embd,
#     vocab_size=vocab_size, 
#     n_head=num_heads, 
#     n_blocks=n_blocks, 
#     block_size=block_size,
#     dropout=dropout
# ).to("cuda")


