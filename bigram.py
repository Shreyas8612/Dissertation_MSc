import requests
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 256
max_iter = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# URL for the Shakespeare dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
file_path = "input.txt"

if not os.path.exists(file_path):
    response = requests.get(url)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
else:
    pass

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

string_to_integer = {ch:i for i,ch in enumerate(chars)}
integer_to_string = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [string_to_integer[c] for c in s]
decode = lambda l: ''.join([integer_to_string[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # 90% of the data is for training and 10% for validation to generalise
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # Basically everything inside the estimate_loss function, do not call backward (No backpropogation) and make pytorch efficient in terms of memory
# This is where taking the average of loss over multiple batches
# This loss is a lot less noisy
def estimate_loss():
    out = {}
    model.eval() # Goes to evaluation phase
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Reseting it back to training phase
    return out

class Head(nn.Module): # Single head of self attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out
        
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
            )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Applying layer norm before it goes into self attention
        x = x + self.ffwd(self.ln2(x))  # Applying layer norm before it goes into Feedfoward
        return x

# B - batch size - Instead of reading the book normally, we tear the book into B different pieces so that they can work INDEPENDENTLY at the same time
# T - block size - The number of characters an assistant can look at once to guess the what the next character should be
# C = n_embed / channels - It is a description of what each character is in essence a description using numbers
# vocab_size - Total number of unique characters the model is allowed to know
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Each token has a C numbers to describe its identity
        tok_emb = self.token_embedding_table(idx) # Prediction of the next token (B, T, C)
        # Each position index gets its own vector of size C - It is like saying this is what position 1 looks like 
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        # The attention mechanism does not know where a word is in a sentance. By adding these numbers we put a position information onto the token's identity
        x = tok_emb + pos_embed # (B, T, C)
        # For every position in every batch, we now have vocab_size of scores representing how likely each character is to be next
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            
            targets = targets.view(B*T) # From B x T to (B*T), Can use -1 if you want Pytorch to guess what you want
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:] # Bring out the Last column of T - the prediction
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
model = BigramLanguageModel()
m=model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # Updating the weights using Adam Optimizer

for iter in range(max_iter):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))