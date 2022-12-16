import json
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import random
import torch
import os
from pathlib import Path

from datasets.notebook_dataset import NotebookDataset


SEED = int(os.environ['SEED'])


def seed_worker(worker_id):
    worker_seed = SEED
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(args, mode="test"):
    is_train = (mode == "train")

    # type = "{mode}_ids_path"  # "train_ids_path", "val_ids_path" or "test_ids_path"
    # ids_path = getattr(args, type.format(mode=mode))
    # df_id = pd.read_pickle(args.train_ids_path)

    df_id = (
        pd.read_pickle(args.train_ids_path) 
        if is_train else pd.read_pickle(args.val_ids_path)
    ) 
    if mode == "train" and not args.preprocess_data:
        n_trains = int(len(list(df_id)) * args.train_size)
        df_id = df_id.head(n_trains)

    df_code_cell = pd.read_pickle(args.df_code_cell_path).set_index("id")
    df_md_cell = pd.read_pickle(args.df_md_cell_path).set_index("id")
    nb_meta_data = json.load(open(args.nb_meta_data_path, "rt"))

    ds = NotebookDataset(
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
