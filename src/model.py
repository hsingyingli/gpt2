import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, embd_dim, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head   = n_head
        self.embd_dim = embd_dim
        assert embd_dim % n_head == 0

        self.depth =  self.embd_dim // self.n_head
        self.wk    = nn.Linear(embd_dim, embd_dim)
        self.wq    = nn.Linear(embd_dim, embd_dim)
        self.wv    = nn.Linear(embd_dim, embd_dim)

        self.out   = nn.Linear(embd_dim, embd_dim)  

    def dot_product(self, k, q, v, mask = None):
        w = torch.matmul(q, torch.transpose(k, 2, 3))

        w = w / math.sqrt(v.size(-1))
        if mask is not None:
            w += (mask * -1e9)

        attention_weight = F.softmax(w, dim = -1)
        out = torch.matmul(attention_weight, v)
        out = F.softmax(out, dim = -1)
        return out, attention_weight

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.n_head, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, key, query, value, mask = None):
        batch_size = key.size(0)
        key, query, value = self.wk(key), self.wq(query), self.wv(value) # (batch_size, seq_len, embd_dim)
        key   = self.split_heads(key, batch_size)                        # (batch_size, n_head, seq_len, depth)
        query = self.split_heads(query, batch_size)
        value = self.split_heads(value, batch_size)
        
        attention, attention_weights = self.dot_product(key, query, value)  # (batch_size, num_heads, seq_len_q, depth)
        attention = attention.permute(0, 2, 1, 3)                           # (batch_size, seq_len_q, num_heads, depth)
        attention = attention.reshape(batch_size, -1, self.embd_dim)        # (batch_size, seq_len_q, embd_dim)

        out = self.out(attention)
        return out, attention_weights


class MLP(nn.Module):
    def __init__(self, embd_dim):
        super(MLP, self).__init__()
        self.fc1  = nn.Linear(embd_dim, embd_dim*2)
        self.relu = nn.ReLU(True) 
        self.fc2  = nn.Linear(embd_dim*2, embd_dim)

    def forward(self, x):
        # (batch_size, seq_len, embd_dim)
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out


class Block(nn.Module):
    def __init__(self, embd_dim, n_head):
        super(Block, self).__init__()
        '''
        layer norm
        attention
        layer norm
        mlp
        '''
        self.layernorm1 = nn.LayerNorm(embd_dim)
        self.attention  = MultiHeadAttention(embd_dim, n_head)  # embd_dim, n_head
        self.layernorm2 = nn.LayerNorm(embd_dim)
        self.mlp        = MLP(embd_dim)

    def forward(self, x):
        x         = self.layernorm1(x)
        atten, _  = self.attention(x, x, x)
        x         = self.layernorm2(atten+ x)
        out       = self.mlp(x)
        return out



class GPT2(nn.Module):
    def __init__(self, blocks, vocab_size, target_vocab_size, embd_dim, n_head):
        super(GPT2, self).__init__()

        self.embd_dim = embd_dim
        self.embedding = nn.Embedding(vocab_size, embd_dim, padding_idx=0)
        self.positional_encoding = self.positional_encoding(vocab_size, embd_dim)
        self.block = nn.ModuleList([Block(embd_dim, n_head) for _ in range(blocks)])
        self.out   = nn.Linear(embd_dim, target_vocab_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        x  = self.embedding(x)
        x += self.positional_encoding[:, :seq_len, :].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        for block in self.block:
            x = block(x)
        
        out = self.out(x)
        return out[:, -1, :]


    def get_angles(self, pos, i, embd_dim):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(embd_dim))
        return pos * angle_rates

    def positional_encoding(self, position, embd_dim):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(embd_dim)[np.newaxis, :],
                                embd_dim)
        
        
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        
        pos_encoding = pos_encoding[np.newaxis, ...]
            
        return torch.FloatTensor(pos_encoding)





if __name__ == "__main__":
    test  = torch.randn(64,50)
    # blocks, vocab_size, embd_dim, n_head
    model = GPT2(5, 20000, 64, 12)
    out = model(test)