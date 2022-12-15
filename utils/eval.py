import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_raw_preds(model: nn.Module, loader: DataLoader, device, name):
    model.eval()
    pbar = tqdm(loader, desc=name)    
    nb_ids = []
    point_preds = []
    with torch.inference_mode():
        for batch in pbar:
            nb_ids.extend(batch['nb_id'])
            
            for attr in batch:
                if attr != 'nb_id':
                    batch[attr] = batch[attr].to(device)
            
            with torch.cuda.amp.autocast(False):
                point_pred = model(
                    batch['code_input_ids'],
                    batch['code_attention_masks'],
                    batch['md_input_ids'],
                    batch['md_attention_masks'],
                    batch['code_cell_padding_masks'],
                    batch['md_cell_padding_masks']
                )

            indices = torch.where(batch['reg_masks'] == True)
            point_preds.extend(point_pred[indices].cpu().numpy().tolist())
   
    return nb_ids, point_preds


def get_point_preds(point_preds: np.array, df: pd.DataFrame):
    df = df.reset_index()
    df.loc[df.cell_type == "markdown", 'rel_pos'] = point_preds
    df['pred_rank'] = df.groupby('id')['rel_pos'].rank()
    code_rank_correction(df)
    return df.sort_values('pp_rank').groupby('id')['cell_id'].apply(list)


def code_rank_correction(df):
    """Swap the code cells based on the given order
    """
    df['pp_rank'] = df['pred_rank'].copy()
    df.loc[df['cell_type'] == 'code', 'pp_rank'] = df.loc[
        df['cell_type'] == 'code'
    ].sort_values(['id', 'rel_pos'])['pred_rank'].values
    print('> Non-corrected %:', (df['pp_rank'] == df['pred_rank']).mean())


def predict(model, loader, df, device, name):
    _, preds = get_raw_preds(model, loader, device, name)
    pred_series = get_point_preds(preds, df)

    return pred_series
