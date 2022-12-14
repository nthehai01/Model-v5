import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class NotebookDataset(Dataset):
    def __init__(self, 
                 code_pretrained, 
                 md_pretrained, 
                 max_len, 
                 ellipses_token_id, 
                 df_id,
                 nb_meta_data, 
                 df_code_cell,
                 df_md_cell,
                 max_n_code_cells,
                 max_n_md_cells,
                 is_train=False):
        super(NotebookDataset, self).__init__()
        self.code_tokenizer = AutoTokenizer.from_pretrained(code_pretrained)
        self.md_tokenizer = AutoTokenizer.from_pretrained(md_pretrained)
        self.max_len = max_len
        self.front_lim = (max_len-2) // 2 + 2 - (max_len%2 == 0)
        self.back_lim = self.max_len - self.front_lim - 1
        self.ellipses_token_id = ellipses_token_id
        self.df_id = df_id
        self.nb_meta_data = nb_meta_data
        self.df_code_cell = df_code_cell
        self.df_md_cell = df_md_cell
        self.max_n_code_cells = max_n_code_cells
        self.max_n_md_cells = max_n_md_cells
        self.is_train = is_train


    def _trunc_mid(self, ids):
        """
        Truncate the middle part of the texts if it is too long
        Use a token (ellipses_token_id) to separate the front and back part
        """
        if len(ids) > self.max_len:
            return ids[:self.front_lim] + [int(self.ellipses_token_id)] + ids[-self.back_lim:]
        return ids


    def _encode_texts(self, df_cell, n_pads, tokenizer):
        texts = (
            ['starting' + tokenizer.sep_token] +
            df_cell['source'].tolist() + 
            ['ending' + tokenizer.sep_token] +
            n_pads * ['padding' + tokenizer.sep_token]
        )  # len = max_n_cells + 2
        print(len(texts))

        inputs = tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=False
        )

        tokens = list(map(self._trunc_mid, inputs['input_ids']))
        tokens = torch.LongTensor(tokens)
        
        cell_masks = list(map(lambda x: x[:self.max_len], inputs['attention_mask']))
        cell_masks = torch.LongTensor(cell_masks)
        
        return tokens, cell_masks


    def __getitem__(self, index):
        nb_id = self.df_id[index]
        n_code_cells = self.nb_meta_data[nb_id]['n_code_cells']
        n_md_cells = self.nb_meta_data[nb_id]['n_md_cells']
        df_code_cell = self.df_code_cell.loc[nb_id].copy()
        df_md_cell = self.df_md_cell.loc[nb_id].copy()

        if self.is_train:
            # code cells
            n_code_cell_pads = int(max(0, self.max_n_code_cells - n_code_cells))
            max_n_code_cells = self.max_n_code_cells
            # md cells
            n_md_cell_pads = int(max(0, self.max_n_md_cells - n_md_cells))
            max_n_md_cells = self.max_n_md_cells
        else:
            # code cells
            n_code_cell_pads = 0
            max_n_code_cells = n_code_cells
            # md cells
            n_md_cell_pads = 0
            max_n_md_cells = n_md_cells
        
        # encode cells
        code_input_ids, code_attention_masks = self._encode_texts(
            df_code_cell, 
            n_code_cell_pads, 
            self.code_tokenizer
        )
        md_input_ids, md_attention_masks = self._encode_texts(
            df_md_cell, 
            n_md_cell_pads, 
            self.md_tokenizer
        )

        # cell attention masks
        code_cell_padding_masks = torch.zeros(max_n_code_cells + 2).bool()  # start + n_cells + end
        code_cell_padding_masks[n_code_cells+2:] = True  # start to end are useful
        md_cell_padding_masks = torch.zeros(max_n_md_cells + 2).bool()
        md_cell_padding_masks[n_md_cells+2:] = True

        # n md cells
        n_md_cells_torch = torch.FloatTensor([n_md_cells])

        # regression md masks
        reg_masks = torch.ones(max_n_md_cells).bool()
        reg_masks[n_md_cells:] = False

        # pointwise target for md cells
        point_pct_target = torch.FloatTensor(df_md_cell['pct_rank'].tolist() + n_md_cell_pads*[0.])
        
        return {
            'nb_id': nb_id,
            'code_input_ids': code_input_ids, 
            'code_attention_masks': code_attention_masks,
            'md_input_ids': md_input_ids,
            'md_attention_masks': md_attention_masks,
            'code_cell_padding_masks': code_cell_padding_masks,
            'md_cell_padding_masks': md_cell_padding_masks,
            'n_md_cells': n_md_cells_torch,
            'reg_masks': reg_masks,
            'point_pct_target': point_pct_target
        }


    def __len__(self):
        return len(self.df_id)
