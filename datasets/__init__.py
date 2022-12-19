import json
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import random
import torch
import os

from datasets.ordering_dataset import OrderingDataset
from datasets.mlm_dataset import MLMDataset


SEED = int(os.environ['SEED'])
MLM_PROBABILITY = float(os.environ['MLM_PROBABILITY'])


def remove_test_ids(train_ids, test_ids_path):
    """
    Filter out training notebooks that are in the test set.
    """
    test_ids = pd.read_pickle(test_ids_path)
    idx_in = np.isin(train_ids.values, test_ids.values)
    train_ids = train_ids.iloc[~idx_in].reset_index(drop=True)
    return train_ids


def remove_non_en_nb(train_ids, non_en_ids_path):
    """
    Filter out non-english notebooks.
    """
    non_en_ids = pd.read_pickle(non_en_ids_path)
    idx_in = np.isin(train_ids.values, non_en_ids.values)
    train_ids = train_ids.iloc[~idx_in].reset_index(drop=True)
    return train_ids


def seed_worker(worker_id):
    worker_seed = SEED
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(args, objective="ordering", mode="test"):
    is_train = (mode == "train")

    type = "{mode}_ids_path"  # "train_ids_path", "val_ids_path" or "test_ids_path"
    ids_path = getattr(args, type.format(mode=mode))
    df_id = pd.read_pickle(ids_path)
    if is_train:
        df_id = remove_test_ids(df_id, args.test_ids_path)
        df_id = remove_non_en_nb(df_id, args.non_en_ids_path)
    df_code_cell = pd.read_pickle(args.df_code_cell_path).set_index("id")
    df_md_cell = pd.read_pickle(args.df_md_cell_path).set_index("id")
    nb_meta_data = json.load(open(args.nb_meta_data_path, "rt"))

    if objective == "ordering":
        ds = OrderingDataset(
            args.code_pretrained, 
            args.md_pretrained, 
            args.max_len, 
            args.ellipses_token_id, 
            df_id,
            nb_meta_data, 
            df_code_cell,
            df_md_cell,
            args.max_n_code_cells,
            args.max_n_md_cells,
            is_train
        )
    if objective == "mlm":
        ds = MLMDataset(
            args.code_pretrained, 
            args.md_pretrained, 
            args.max_len, 
            args.ellipses_token_id, 
            df_id,
            nb_meta_data, 
            df_code_cell,
            df_md_cell,
            args.max_n_code_cells,
            args.max_n_md_cells,
            MLM_PROBABILITY,
            is_train
        )

    g = torch.Generator()
    g.manual_seed(SEED)
    data_loader = DataLoader(
        ds, 
        batch_size=(is_train*args.batch_size or 1), 
        shuffle=is_train, 
        num_workers=args.n_workers,
        pin_memory=False, 
        drop_last=is_train,
        worker_init_fn=seed_worker,
        generator=g
    )

    return data_loader
