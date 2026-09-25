# coding: utf-8
__author__ = 'Ilya Kiselev (kiselecheck): https://github.com/kiselecheck'
__version__ = '1.0.1'

import os

import torch
import torch.multiprocessing as mp
from train import train_model
from utils.settings import  cleanup_ddp, find_free_port
import warnings

warnings.filterwarnings("ignore")


def train_model_single(rank: int, world_size: int, args=None):
    """
    Trains the model based on the provided arguments, including data preparation, optimizer setup,
    and loss calculation. The model is trained for multiple epochs with logging via wandb.

    Args:
        world_size:
        rank:
        args: Command-line arguments containing configuration paths, hyperparameters, and other settings.

    Returns:
        None
    """
    train_model(args, rank, world_size)  # Close DDP


def train_model_ddp(args=None):
    world_size = torch.cuda.device_count()
    # [fork patch] Choose the DDP rendezvous address ONCE, here in the parent, so every
    # spawned rank inherits the identical value through its environment. Upstream let
    # each child derive the port from its own int(time.time()) reading, which made the
    # launch a race: ranks that imported torch either side of a second boundary picked
    # different ports and init_process_group blocked until it timed out.
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', str(find_free_port()))
    print(f'DDP rendezvous: {os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}, '
          f'world_size={world_size}', flush=True)
    try:
        mp.spawn(train_model_single, args=(world_size, args), nprocs=world_size, join=True)
    except Exception as e:
        cleanup_ddp()
        raise e


if __name__ == "__main__":
    train_model_ddp()
