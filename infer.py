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


config_list = [
["/project/dataset/users/liam.1234/localnerf_logs/logs/eg3d_ffhq_clipedit_eyebrows/2022_09_20_16_25_05/config.yaml", "eyebrows_arched", 4000],
        ]

def main(args):
    configs = config.walk_configs(config_list[0][0])
    configs['ngpus'] = torch.cuda.device_count()
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

    zs = torch.randn([args.N_images, 512]).cuda()

    for idx, (config_path, name, step) in enumerate(config_list):
        configs = config.walk_configs(config_path)
        configs['ngpus'] = torch.cuda.device_count()

        # use one gpu for inference_qual
        assert configs['ngpus'] == 1
        # Set all random seeds
        # assert random seeds are all same
        assert random_seed == configs['experiment']['random_seeds']


        trainer_kwargs = dnnlib.EasyDict(class_name=f"trainers.{configs['experiment']['trainer']}.{configs['experiment']['trainer']}", configs=configs, rank=args.local_rank, world_size=args.world_size, mode='train', random_seed=args.used_seed)
        trainer = dnnlib.util.construct_class_by_name(**trainer_kwargs)

        torch.cuda.empty_cache()
        trainer.load_checkpoints(config_path, step)
        #trainer.inference_qual(args.name)
        save_origin = True if idx==0 else False
        trainer.inference_random(zs, name, noise_mode=args.noise_mode, save_origin=save_origin)


if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser(description='LENeRF')
    #parser.add_argument('--config-path', type=str, default='./config', help='path to configuration')
    parser.add_argument('--experiment-id', type=str, default=None, help='experiment id containing configs to use')
    #parser.add_argument('--step', type=int, default=None, required=True, help='checkpoint step used for inference')
    #parser.add_argument('--name', type=str, default=None, required=True, help='name of save folder')
    parser.add_argument('--noise_mode', type=str, required=True, help='StyleGAN2 noise_mode, const or random')
    parser.add_argument('--N_images', type=int, required=True)

    # torch distributed
    parser.add_argument('--syncbn', action='store_true', default=True, help='Use Synchronized BN')
    parser.add_argument('--dist_url', default='127.0.0.1:', type=str, help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank for torch distributed training')
    parser.add_argument('--restore_optimizer', action='store_true', default=True, help='restore optimizer')
    args = parser.parse_args()

    args.world_size = 1

    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        print(f"Total world size: {args.world_size}")
        #args.local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(args.local_rank)
    print(f"My Rank: {args.local_rank}")

    dist.init_process_group(backend='nccl',
            init_method="env://")
    #torch.distributed.barrier(device_ids=[args.local_rank])
    print('finished init')
    main(args)

    dist.destroy_process_group()

