import os
import numpy as np
import random
import torch
from typing import List


def make_folder(folder: str):
    """
    Makes the folder if not already present
    Args:
        folder (str): Name of folder to create
    Returns:
        created (bool): Whether or not a folder was created
    """
    
    try:
        os.mkdir(folder)
        return True
    except FileExistsError:
        pass
    return False


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lr_to_4sf(lr: List[float]) -> str:
    """Get string of lr list that is rounded to 4sf to not clutter pbar

    Warning:
        Doesn't work for floats > 10000
    """
    def _f(x) -> str:
        a = str(x).partition('e')
        return a[0][:5] + 'e' + a[-1]
    return '[' + ', '.join(map(_f, lr)) + ']'
