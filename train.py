import os
import time
import argparse
import numpy as np
import random
import torch
import dnnlib

import torch.distributed as dist
from pathlib import Path
#import trainers

from torch_utils.logging import setup_logging
import config

def main(args, configs):
    # setup experiment logging
    if not configs['experiment']['debug']:
        setup_logging(args, configs)

    #folder = [Path.cwd() / 'trainers']
    #trainer_name = configs['experiment']['trainer']
    #current_module = 'trainers'
    #trainer = trainers.recursive_find_python_class(folder, trainer_name, current_module)\
    #    (configs=configs, rank=args.local_rank, world_size=args.world_size, mode='train')
    trainer_kwargs = dnnlib.EasyDict(class_name=f"trainers.{configs['experiment']['trainer']}.{configs['experiment']['trainer']}", configs=configs, rank=args.local_rank, world_size=args.world_size, mode='train', random_seed=args.used_seed)
    trainer = dnnlib.util.construct_class_by_name(**trainer_kwargs)

    print(trainer.TRAINER_NAME)
    torch.cuda.empty_cache()
    if args.local_rank == 0:
        trainer.init_wandb()
    trainer.training()
    if args.local_rank == 0:
        trainer.close_wandb()


if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser(description='LENeRF')
    parser.add_argument('--config-path', type=str, default='./config', help='path to configuration')
    parser.add_argument('--experiment-id', type=str, default=None, help='experiment id containing configs to use')

    # torch distributed
    parser.add_argument('--syncbn', action='store_true', default=True, help='Use Synchronized BN')
    parser.add_argument('--dist_url', default='127.0.0.1:', type=str, help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank for torch distributed training')
    parser.add_argument('--restore_optimizer', action='store_true', default=True, help='restore optimizer')
    args = parser.parse_args()

    configs = config.walk_configs(args.config_path)
    configs['ngpus'] = torch.cuda.device_count()

    # Set all random seeds
    random_seed = configs['experiment']['random_seeds']
    used_seed = random_seed * configs['ngpus'] + args.local_rank

    torch.manual_seed(used_seed)
    torch.cuda.manual_seed(used_seed)
    torch.cuda.manual_seed_all(used_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(used_seed)
    random.seed(used_seed)
    args.used_seed = used_seed

    args.world_size = 1

    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        print(f"Total world size: {args.world_size}")
        #args.local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(args.local_rank)
    print(f"My Rank: {args.local_rank}, seed = {used_seed}")

    dist.init_process_group(backend='nccl',
            init_method="env://")
    #torch.distributed.barrier(device_ids=[args.local_rank])
    print('finished init')
    main(args, configs)

    dist.destroy_process_group()

