import logging
import os
import sys
import numpy 
import random

from src.trainer import which #checks GPU requirements
if which('nvidia-smi') is not None:
    min=8000
    deviceid = 0
    name, mem = os.popen('"nvidia-smi" --query-gpu=gpu_name,memory.total --format=csv,nounits,noheader').read().split('\n')[deviceid].split(',')
    print(mem)
    mem = int(mem)
    if mem < min:
        print(f'Less GPU memory than requested ({mem}<{min}). Terminating.')
        sys.exit()

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from src.models import PELICANClassifier
from src.models import tests
from src.trainer import Trainer
from src.trainer import init_argparse, init_file_paths, init_logger, init_cuda, logging_printout, fix_args, set_seed, get_world_size
from src.trainer import init_optimizer, init_scheduler

from src.dataloaders import initialize_datasets, collate_fn

from src.models.dynamic_entropy_loss import DynamicEntropyLoss

# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000, sci_mode=False)


def main():

    # Initialize arguments
    args = init_argparse()

    print("DEBUG: num_channels_2to2 =", args.num_channels_2to2)
    print("DEBUG: num_channels_m =", args.num_channels_m)
    print("DEBUG: num_channels_scalar =", args.num_channels_scalar)


    # Choose the right performance tracking tools based on the number of classes
    if args.num_classes<=2:
        from src.models.metrics_classifier import metrics, minibatch_metrics, minibatch_metrics_string
    else:
        from src.models.metrics_multiclass import metrics, minibatch_metrics, minibatch_metrics_string

    # Initialize file paths
    args = init_file_paths(args)

    # Fix possible inconsistencies in arguments
    # args = fix_args(args)

    if 'LOCAL_RANK' in os.environ:
        device_id = int(os.environ["LOCAL_RANK"])
    else:
        device_id = -1
    
    # Initialize logger to track progress
    logger = logging.getLogger('')
    init_logger(args, device_id)

    if which('nvidia-smi') is not None:
        logger.info(f'Using {name} with {mem} MB of GPU memory (local rank {device_id})')

    # Write input paramaters and paths to log
    if device_id <= 0:
        logging_printout(args)

    # Initialize device and data type(CPU or GPU)
    device, dtype = init_cuda(args, device_id)

    # Fix possible inconsistencies in arguments
    args = fix_args(args)
    # Set a manual random seed for torch, cuda, numpy, and random (For consistant results)
    args = set_seed(args, device_id)

    distributed = (get_world_size() > 1)
    if distributed:
        world_size = dist.get_world_size()
        logger.info(f'World size {world_size}')

    # Fix a seed for dataloading (useful when num_train!=-1 and one wants the same training data across runs)
    if args.fix_data:
        torch.manual_seed(165937750084982)
    # Initialize dataloder(Prepares data sets and specifies data location)
    args.datadir = "data/sample_data/run12"
    args, datasets = initialize_datasets(args, args.datadir, num_pts=None, testfile=args.testfile, balance=(args.num_classes==2), RAMdataset=args.RAMdataset)

    # Construct PyTorch dataloaders from datasets(Function to format data into batches)
    collate = lambda data: collate_fn(data, scale=args.scale, nobj=args.nobj)
    
    # Whether testing set evaluation should be distributed
    print("Datasets keys:", datasets.keys())
    distribute_eval=args.distribute_eval

    # Set up how data will be loaded
    if distributed:
        samplers = {'train': DistributedSampler(datasets['train'], shuffle=args.shuffle),
                    'valid': DistributedSampler(datasets['valid'], shuffle=False),
                    'test': DistributedSampler(datasets['test'], shuffle=False) if distribute_eval else None}
    else:
        samplers = {split: None for split in datasets.keys()}

    # Create data loaders to fetch batches of data for training
    dataloaders = {split: DataLoader(dataset,
                                     batch_size = args.batch_size,
                                     shuffle = args.shuffle if (split == 'train' and not distributed) else False,
                                     num_workers = args.num_workers,
                                     pin_memory=True,
                                     worker_init_fn = seed_worker,
                                     collate_fn =collate,
                                     sampler = samplers[split]
                                     )
                   for split, dataset in datasets.items()}

    # Initialize model
    model = PELICANClassifier(args.rank1_width_multiplier, args.num_channels_scalar, args.num_channels_m, args.num_channels_2to2, args.num_channels_out, args.num_channels_m_out, 
                              stabilizer=args.stabilizer, method = args.method, num_classes=args.num_classes,
                              activate_agg=args.activate_agg, activate_lin=args.activate_lin,
                              activation=args.activation, config=args.config, config_out=args.config_out, average_nobj=args.nobj_avg,
                              factorize=args.factorize, masked=args.masked,
                              activate_agg_out=args.activate_agg_out, activate_lin_out=args.activate_lin_out, mlp_out=args.mlp_out,
                              scale=args.scale, irc_safe=args.irc_safe, dropout = args.dropout, drop_rate=args.drop_rate, drop_rate_out=args.drop_rate_out, batchnorm=args.batchnorm,
                              dataset=args.dataset, device=device, dtype=dtype)
    
    model.to(device)

    if distributed:
        model = DistributedDataParallel(model, device_ids=[device_id])

    restart_epochs = []

    # Initialize the scheduler and optimizer
    if args.task.startswith('eval'):
        optimizer = scheduler = None
        summarize = False
    else:
        optimizer = init_optimizer(args, model, len(dataloaders['train']))
        scheduler, restart_epochs = init_scheduler(args, optimizer)  # Unpacking the tuple


    # Define a loss function.(How the model should measure errors)
    # loss_fn = torch.nn.functional.cross_entropy
    if args.num_classes==1:
        raise NotImplementedError
    elif args.num_classes==2:
        loss_fn = DynamicEntropyLoss(total_epochs=args.num_epoch)
    else:
        loss_fn = lambda predict, targets: torch.nn.CrossEntropyLoss()(predict, targets.long().argmax(dim=1))
    
    # Apply the covariance and permutation invariance tests.
    if args.test and device_id <= 0:
        tests(model, dataloaders['train'], args, tests=['gpu','irc', 'permutation'])

    # Instantiate the training class
    trainer = Trainer(args, dataloaders, model, loss_fn, metrics,
                      minibatch_metrics, minibatch_metrics_string, optimizer, scheduler,
                      restart_epochs, device_id, device, dtype)
    
    if not args.task.startswith('eval'):
        # Load from checkpoint file (if one exists)
        trainer.load_checkpoint()
        # Restore random seed
        args = set_seed(args, device_id)
        # This makes the results exactly reproducible on a GPU (on CPU they're reproducible regardless) by banning certain non-deterministic operations
        if args.reproducible:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        # Train model.
        trainer.train()

    # Test predictions on best model and also last checkpointed model.
    # If distributed==False, only one GPU will do this during DDP sessions 
    # so that the order  of batches is preserved in the output file.
    trainer.evaluate(splits=['test'], distributed=distributed and distribute_eval)
    if distributed:
        dist.destroy_process_group()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':
    main()
