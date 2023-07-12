"""
Miscellanous Functions
"""

import torch
import os
import logging
import sys
from PIL import ImageFile
from datetime import datetime
from pathlib import Path
import shutil
ImageFile.LOAD_TRUNCATED_IMAGES = True

### logger = logging.getLogger(__name__)

def setup_logging(args, configs):
    fmt = '%(asctime)s.%(msecs)03d %(message)s'
    date_fmt = '%m-%d %H:%M:%S'

    log_date = str(datetime.now().strftime('%m_%d_%H'))
    log_time = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    prefix = 'log'

    if args.local_rank == 0:
        exp_master_path = configs['experiment']['log_dir']
        os.makedirs(exp_master_path, exist_ok=True)
        exp_dir = os.path.join(exp_master_path, log_time)

        os.makedirs(exp_dir, exist_ok=True)
        shutil.copyfile(f"{args.config_path}", f"{exp_dir}/config.yaml")

        # newly define exp_dir
        configs['experiment']['exp_dir'] = exp_dir
        filename = os.path.join(exp_dir,
                                f'{prefix}_{log_time}_rank_'
                                f'{args.local_rank}.log')
        print(f"Logging: {filename}")
        logging.basicConfig(level=logging.INFO, format=fmt, datefmt=date_fmt,
                            filename=filename, filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # dump configs and arguments
        open(os.path.join(exp_dir, f'{log_time}.txt'), 'w').write(
             f'{str(args)}\n\n{str(configs)}')

        print(f"Save path : {exp_dir}")

        generated_dir = f"{exp_dir}/{configs['experiment']['generated_dir']}"
        os.makedirs(generated_dir, exist_ok=True)

    """
    else:
        fh = logging.FileHandler(filename)
        logging.getLogger('').addHandler(fh)
    """

'''
def save_model(args, configs, model_type, net, optimizer, scheduler, epoch, fid=None,
               save_optimizer=True, optimizer_at=None, scheduler_at=None):

    if 'last_records' not in configs:
        configs['last_records'] = {'epoch': -1, 'fid': None}


    elif configs['last_records']['epoch'] != epoch:
        # update the latest snapshot
        dataset_name = configs['experiment']['dataset_name']
        last_epoch = configs['last_records']['epoch']
        last_fid = configs['last_records']['fid']
        last_snapshot = f'{model_type}_{dataset_name}_epoch_{last_epoch}.pth'
        last_snapshot = os.path.join(args.exp_path, last_snapshot)

        # remove the last checkpoint
        try:
            os.remove(last_snapshot)
        except OSError:
            pass

    # create a new one
    last_snapshot = f'{model_type}_{dataset_name}_epoch_{epoch}.pth'
    last_snapshot = os.path.join(args.exp_path, last_snapshot)

    args.last_record['epoch'] = epoch
    args.last_record['fid'] = fid

    torch.cuda.synchronize()

    # save checkpoints
    if optimizer_at is not None:
        torch.save({
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'optimizer_at': optimizer_at.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scheduler_at': scheduler_at.state_dict(),
            'epoch': epoch,
            'command': ' '.join(sys.argv[1:])
        }, last_snapshot)
    else:
        torch.save({
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'command': ' '.join(sys.argv[1:])
        }, last_snapshot)


def save_optimizer(configs, model_type, optimizer, scheduler, epoch, step,
               save_optimizer=True, optimizer_at=None, scheduler_at=None):

    fid = None
    prev_last_snapshot = None
    dataset_name = configs['experiment']['dataset_name']
    if 'last_records' not in configs:
        configs['last_records'] = {'epoch': -1, 'fid': None}

    elif configs['last_records']['epoch'] != epoch:
        # update the latest snapshot
        last_epoch = configs['last_records']['epoch']
        last_fid = configs['last_records']['fid']
        prev_last_snapshot = f'{model_type}_{dataset_name}_epoch_{last_epoch}.pth'
        prev_last_snapshot = os.path.join(configs['experiment']['exp_dir'], prev_last_snapshot)

    # create a new one
    last_snapshot = f'{model_type}_{dataset_name}_epoch_{epoch}.pth'
    last_snapshot = os.path.join(configs['experiment']['exp_dir'], last_snapshot)

    configs['last_records']['epoch'] = epoch
    configs['last_records']['fid'] = fid

    torch.cuda.synchronize()

    # save checkpoints
    save_dict = {'epoch': epoch, 'steps':step, 'command': ' '.join(sys.argv[1:])}

    for key, value in optimizer.items():
        save_dict[key] = value.state_dict()

    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()

    if optimizer_at is not None:
        save_dict['optimizer_at'] = optimizer_at.state_dict()
    if scheduler_at is not None:
        save_dict['scheduler_at'] = scheduler_at.state_dict()

    torch.save(save_dict, last_snapshot)

    # remove the last checkpoint
    if prev_last_snapshot is not None:
        try:
            os.remove(prev_last_snapshot)
        except OSError:
            pass

def load_model(configs, model_type, load_path=None, dataset_name=None):
    if dataset_name == None:
        dataset_name = configs['experiment']['dataset_name']
    prefix = f"{model_type}_{dataset_name}_epoch"
    if load_path is None:
        load_path = f"{configs['experiment']['log_dir']}/{configs['experiment']['resume_folder']}"

    pth_names = [[int(f.split(".")[-2].split('_')[-1]), f] for f in os.listdir(load_path) if f[:len(prefix)] == prefix]
    pth_names.sort()
    pth_name = pth_names[-1][1]

    ckpt_dict = torch.load(f"{load_path}/{pth_name}")
    return ckpt_dict, pth_name

def create_dir_ifnot_exist(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path
'''
