import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F


class Linear(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.01) 


class CellEncoder(nn.Module):
    def __init__(self, model_path):
        super(CellEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        
    def forward(self, input_ids, attention_masks):
        batch_size, max_cell, max_len = input_ids.shape

        input_ids = input_ids.view(-1, max_len)
        attention_masks = attention_masks.view(-1, max_len)

        tokens = self.bert(input_ids, attention_masks)[0]
        emb_dim = tokens.shape[-1]  # bert output embedding dim
        tokens = tokens.view(batch_size, max_cell, max_len, emb_dim)

        return tokens


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.W = Linear(dim, dim, bias=False)
        self.v = Linear(dim, 1, bias=False)
        
    def forward(self, keys, masks):
        weights = self.v(torch.tanh(self.W(keys)))
        weights.masked_fill_(masks.unsqueeze(-1).bool(), -6.5e4)
        weights = F.softmax(weights, dim=2)
        return torch.sum(weights * keys, dim=2)


class PositionalEncoder(nn.Module):
    def __init__(self):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

    def _create_angle_rates(self, dim):
        angles = torch.arange(dim)
        angles[1::2] = angles[0::2]
        angles = 1 / (10000 ** (angles / dim))
        angles = torch.unsqueeze(angles, axis=0)
        return angles

    def _generate_positional_encoding(self, pos, d_model):
        angles = self._create_angle_rates(d_model).type(torch.float32)
        pos = torch.unsqueeze(torch.arange(pos), axis=1).type(torch.float32)
        pos_angles = torch.matmul(pos, angles)
        pos_angles[:, 0::2] = torch.sin(pos_angles[:, 0::2])
        pos_angles[:, 1::2] = torch.cos(pos_angles[:, 1::2])
        pos_angles = torch.unsqueeze(pos_angles, axis=0)

        return pos_angles

    def forward(self, x):
        _, max_cell, emb_dim = x.shape

        pos_encoding = self._generate_positional_encoding(max_cell, emb_dim)

        x += pos_encoding[:, :max_cell, :].to(x.device)
        x = self.dropout(x)
        return x


class PointwiseHead(nn.Module):
    def __init__(self, dim):
        super(PointwiseHead, self).__init__()
        self.fc0 = Linear(dim, 256)
        self.fc1 = Linear(256, 128)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.LeakyReLU()
        self.top = Linear(128, 1)  
        
    def forward(self, x):
        x = x[:, 1:-1]

        x = self.fc0(x)
        x = self.act(x)

        x = self.dropout(x)

        x = self.fc1(x)
        x = self.act(x)

        x = self.top(x)
        
        return x.squeeze(-1)


class NotebookTransformer(nn.Module):
    def __init__(self, code_pretrained, md_pretrained, code_emb_dim, md_emb_dim, n_heads, n_layers):
        super(NotebookTransformer, self).__init__()
        self.code_encoder = CellEncoder(code_pretrained)
        self.md_encoder = CellEncoder(md_pretrained)
        self.code_pooling = Attention(code_emb_dim)
        self.md_pooling = Attention(md_emb_dim)
        self.code_positional_encoder = PositionalEncoder()
        self.code_cell_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=code_emb_dim, nhead=n_heads, batch_first=True), 
            num_layers=n_layers
        )
        self.md_cell_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=md_emb_dim, nhead=n_heads, batch_first=True),
            num_layers=n_layers
        )
        self.pointwise_head = PointwiseHead(md_emb_dim)

    def forward(self, 
                code_input_ids, code_attention_masks, 
                md_input_ids, md_attention_masks,
                code_cell_padding_masks, md_cell_padding_masks):
        # cell encoder
        code_embedding = self.code_encoder(code_input_ids, code_attention_masks)  # [..., max_code_cell+2, max_len, emb_dim]
        md_embedding = self.md_encoder(md_input_ids, md_attention_masks)  # [..., max_md_cell+2, max_len, emb_dim]

        # cell pooling
        code_embedding = self.code_pooling(code_embedding, code_attention_masks)  # [..., max_code_cell+2, emb_dim]
        md_embedding = self.md_pooling(md_embedding, md_attention_masks)  # [..., max_md_cell+2, emb_dim]

        # add positional encoder
        code_embedding = self.code_positional_encoder(code_embedding)  # [..., max_code_cell+2, emb_dim]

        # code cell encoder
        code_cell_embedding = self.code_cell_encoder(
            src=code_embedding,
            src_key_padding_mask=code_cell_padding_masks
        )  # [..., max_code_cell+2, emb_dim]

        # md cell decoder
        md_cell_embedding = self.md_cell_decoder(
            tgt=md_embedding,
            memory=code_cell_embedding,
            tgt_key_padding_mask=md_cell_padding_masks,
            memory_key_padding_mask=code_cell_padding_masks
        )  # [..., max_md_cell+2, emb_dim]

        # pointwise head
        x = self.pointwise_head(md_cell_embedding)  # [..., max_md_cell]

        return x
