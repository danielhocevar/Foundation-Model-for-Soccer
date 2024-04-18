import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, ntoken, ninp=200, nhead=2, nhid=200, nlayers=2, dropout=0.2, seq_length=9):
        """
        ntoken: dictionary length
        ninp: size of word embeddings
        nhead: number of heads in the encoder/decoder
        nhid: number of hidden units in hidden layers
        nlayers: number of hidden layers
        dropout: dropout probability
        """
        super(MLP, self).__init__()
        self.model_type = 'MLP'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.ntoken = ntoken
        self.seq_length = seq_length

        modules = [nn.Linear(seq_length*ninp, nhid), nn.ReLU()]

        for i in range(nlayers):
            modules.append(nn.Linear(nhid, nhid))
            modules.append(nn.ReLU())

        modules.append(nn.Linear(nhid, seq_length*ntoken))
        self.mlp = nn.Sequential(*modules)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)

        for i in range(len(self.mlp)):
            if isinstance(self.mlp[i], nn.Linear):
                nn.init.zeros_(self.mlp[i].bias)
                nn.init.uniform_(self.mlp[i].weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        """
        input shape: [seq_length, batch_size]
        output shape: [seq_length, batch_size, ntoken]
        """
        src = src.permute(1, 0)

        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        
        src = src.permute(1, 0, 2)
        src = src.reshape(src.shape[0], -1)

        output = self.mlp(src)

        output = output.reshape(output.shape[0], self.seq_length, self.ntoken)
        return F.log_softmax(output, dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
