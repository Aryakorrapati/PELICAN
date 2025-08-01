import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import os
import logging
import optuna
from optuna.trial import TrialState

from src.models import PELICANClassifier
from src.models import tests
from src.trainer import Trainer
from src.trainer import init_argparse, init_file_paths, init_logger, init_cuda, logging_printout, fix_args
from src.trainer import init_optimizer, init_scheduler
from src.models.metrics_classifier import metrics, minibatch_metrics, minibatch_metrics_string

from src.dataloaders import initialize_datasets, collate_fn

import glob

print("CWD:", os.getcwd())
print("Expecting data in:", os.path.abspath('data/'))

# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000, sci_mode=False)

logger = logging.getLogger('')


def suggest_params(args, trial):

    # args.lr_init = trial.suggest_loguniform("lr_init", 0.0005, 0.005)
    # args.num_epoch = trial.suggest_int("num_epoch", 40, 80, step=10)
    # args.lr_final = trial.suggest_loguniform("lr_final", 1e-8, 1e-5)
    # args.scale = trial.suggest_loguniform("scale", 1e-2, 3)
    # args.sig = trial.suggest_categorical("sig", [True, False])
    # args.drop_rate = trial.suggest_float("drop_rate", 0, 0.5, step=0.05)
    # args.layernorm = trial.suggest_categorical("layernorm", [True, False])
    # args.lr_decay_type = trial.suggest_categorical("lr_decay_type", ['exp', 'cos'])

    args.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
    args.lr_init = trial.suggest_loguniform("lr_init", 1e-5, 1e-2)        # or whatever range you want
    args.pct_start = trial.suggest_float("pct_start", 0.05, 0.5)          # for OneCycleLR, 5% to 50% is typical
    args.max_lr = trial.suggest_loguniform("max_lr", 1e-5, 1e-2)



    args.double = trial.suggest_categorical("double", [False, True])
    args.factorize = trial.suggest_categorical("factorize", [False, True])
    args.nobj = trial.suggest_int("nobj", 50, 90)
    # args.ir_safe = trial.suggest_categorical("ir_safe", [False, True])
    args.masked = trial.suggest_categorical("masked", [False, True])

    args.config = trial.suggest_categorical("config", ["s", "m", "S", "M"]) # , "sM", "Sm"]) #, "S", "m", "M", "sS", "mM", "sM", "Sm", "SM"]) #, "mx", "Mx", "sSm", "sSM", "smM", "sMmM", "mxn", "mXN", "mxMX", "sXN", "smxn"])
    args.config_out = trial.suggest_categorical("config_out", ["s", "m", "S", "M"]) # , "sM", "Sm"]) #, "S", "m", "M", "sS", "mM", "sM", "Sm", "SM"]) #, "mx", "Mx", "sSm", "sSM", "smM", "sMmM", "mxn", "mXN", "mxMX", "sXN", "smxn"])
    
    n_layers1 = trial.suggest_int("n_layers1", 2, 6)
    num_layers = n_layers1 + 1  # Because num_channels = len(num_channels_2to2) + 1

    n_layersm = [trial.suggest_int("n_layersm", 1, 2) for i in range(num_layers)]  # Note range(num_layers) instead of n_layers1!
    args.num_channels_m = [[trial.suggest_int(f'n_channelsm[{i}, {k}]', 10, 50) for k in range(n_layersm[i])] for i in range(num_layers)]
    
    
    n_layersm_out = trial.suggest_int("n_layersm2", 1, 2)
    args.num_channels_m_out = [trial.suggest_int('n_channelsm_out['+str(k)+']', 10, 50) for k in range(n_layersm_out)]

    args.num_channels_2to2 = [trial.suggest_int("n_channels1["+str(i)+"]", 10, 40) for i in range(n_layers1 + 1)]
    # args.num_channels_2to2 = [trial.suggest_int("n_channels1", 3, 30)]
    # args.num_channels_2to2 = args.num_channels_2to2 * (n_layers1) + [args.num_channels_m[0][0] if n_layersm > 0 else args.num_channels_2to2[0]]

    # args.num_channels_2to2 = [trial.suggest_int("n_channels1", 1, 10)] * n_layers1
    # args.num_channels_m = [[trial.suggest_int("n_channels1", 1, 10), args.num_channels_2to2[0]*15*len(args.config)]] * n_layers1
    # args.num_channels_2to2 = args.num_channels_2to2 + [args.num_channels_m[0][0]]

    n_layers2 = trial.suggest_int("n_layers2", 1, 2)
    # n_layers2 = 1
    args.num_channels_out = [trial.suggest_int("n_channels2["+str(i)+"]", 10, 40) for i in range(n_layers2)]

    # args.activation = trial.suggest_categorical("activation", ["elu", "leakyrelu"]) #, "relu", "silu", "selu", "tanh"])
    # args.optim = trial.suggest_categorical("optim", ["adamw", "sgd", "amsgrad", "rmsprop", "adam"])

    # args.activate_agg = trial.suggest_categorical("activate_agg", [True, False])
    # args.activate_lin = trial.suggest_categorical("activate_lin", [True, False])
    # args.dropout = trial.suggest_categorical("dropout", [True])
    # args.batchnorm = trial.suggest_categorical("batchnorm", ['b'])

    return args

def define_model(trial):
   
    # Initialize arguments
    args = init_argparse()

    args.datadir = 'data/sample_data/run12'

    if not hasattr(args, 'add_beams'):
        args.add_beams = False  # Or whatever default you want

    if not hasattr(args, 'beam_mass'):
        args.beam_mass = 1.0    # Or your appropriate default


    # Initialize file paths
    args = init_file_paths(args)

    # Initialize logger
    init_logger(args)

    # Suggest parameters to optuna to optimize over
    args = suggest_params(args, trial)

    # Write input paramaters and paths to log
    git_status = logging_printout(args, trial)

    # Fix possible inconsistencies in arguments
    args = fix_args(args)

    trial.set_user_attr("git_status", git_status)
    trial.set_user_attr("args", vars(args))

    # Initialize device and data type
    device, dtype = init_cuda(args)

    # Initialize model
    model = PELICANClassifier(
        args.rank1_width_multiplier,
        args.num_channels_scalar,
        args.num_channels_m,
        args.num_channels_2to2,
        args.num_channels_out,
        args.num_channels_m_out,
        activate_agg=args.activate_agg,
        activate_lin=args.activate_lin,
        activation=args.activation,
        config=args.config,
        config_out=args.config_out,
        factorize=args.factorize,
        masked=args.masked,
        activate_agg_out=args.activate_agg_out,
        activate_lin_out=args.activate_lin_out,
        mlp_out=args.mlp_out,
        scale=args.scale,
        irc_safe=args.irc_safe,  # NOT 'ir_safe'
        dropout=args.dropout,
        drop_rate=args.drop_rate,
        batchnorm=args.batchnorm,
        device=device,
        dtype=dtype
    )


    model.to(device)

    return args, model, device, dtype

def define_dataloader(args):

    # Initialize dataloder
    args, datasets = initialize_datasets(args, args.datadir, num_pts=None)

    if args.train_fraction < 1.0:
        full_train = datasets['train']
        n_keep = int(len(full_train) * args.train_fraction)
        idx = torch.randperm(len(full_train))[:n_keep].tolist()
        datasets['train'] = Subset(full_train, idx)
        logger.info(f'Using {n_keep}/{len(full_train)} training samples (fraction={args.train_fraction}).')

    print("args.datadir:", args.datadir)


    print("initialize_datasets keys:", datasets.keys())
    print("initialize_datasets train files:", datasets.get('train', 'None'))


    # Construct PyTorch dataloaders from datasets
    collate = lambda data: collate_fn(data, scale=args.scale, nobj=args.nobj, beam_mass=getattr(args, "beam_mass", 1))
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=args.num_workers,
                                     collate_fn=collate)
                   for split, dataset in datasets.items()}

    return args, dataloaders


def objective(trial):

    args, model, device, dtype = define_model(trial)

    args, dataloaders = define_dataloader(args)

    trial.set_user_attr("seed", args.seed)

    distributed = False

    if distributed:
        model = torch.nn.DataParallel(model)

    # Initialize the scheduler and optimizer
    optimizer = init_optimizer(args, model)
    scheduler, restart_epochs = init_scheduler(args, optimizer)

    # Define a loss function.
    # loss_fn = torch.nn.functional.cross_entropy
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    
    # Apply the covariance and permutation invariance tests.
    if args.test:
        tests(model, dataloaders['train'], args, tests=['permutation','batch','irc'])

    summarize = False

    # Instantiate the training class
    trainer = Trainer(args, dataloaders, model, loss_fn, metrics, minibatch_metrics, minibatch_metrics_string, optimizer, scheduler, restart_epochs, summarize, device, dtype, trial_number=trial.number)

    # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
    trainer.load_checkpoint()

    # Train model.  
    metric_to_report='accuracy' 
    best_epoch, best_metrics = trainer.train(trial=None, metric_to_report=None)

    print(f"Best epoch was {best_epoch} with metrics {best_metrics}")

    if args.optuna_test:
        # Test predictions on best model.
        best_metrics=trainer.evaluate(splits=['test'], best=True, final=False)
        trial.set_user_attr("best_test_metrics", best_metrics)

    return best_metrics[metric_to_report]

if __name__ == '__main__':

    # Initialize arguments
    args = init_argparse()
    
    if args.storage == 'remote':
        storage=optuna.storages.RDBStorage(url=f'postgresql://{os.environ["USER"]}:{args.password}@{args.host}:{args.port}', heartbeat_interval=100)  # For running on nodes with a distributed file system
    elif args.storage == 'local':
        storage = "sqlite:////eos/user/a/akorrapa/test_study.db"


    direction = 'maximize'
    # directions=['minimize', 'maximize', 'maximize']

    if args.sampler.lower() == 'random':
        sampler = optuna.samplers.RandomSampler()
    elif args.sampler.lower().startswith('tpe'):
        sampler = optuna.samplers.TPESampler(n_startup_trials=100, multivariate=True, group=True, constant_liar=True)

    if args.pruner == 'hyperband':
        pruner = optuna.pruners.HyperbandPruner()
    elif args.pruner == 'median':
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=20, n_min_trials=10)
    elif args.pruner == 'none':
        pruner = None

    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(f"Using storage string: {storage}")


    study = optuna.create_study(study_name=args.study_name, storage=storage, direction=direction, load_if_exists=True,
                                pruner=pruner, sampler=sampler)

    # init_params =  {
    #                 # 'activate_agg': False,
    #                 # 'activate_lin': True,
    #                 'activation': 'leakyrelu',
    #                 'batch_size': 32,
    #                 'config': 'learn',
    #                 # 'lr_final': 1e-07,
    #                 # 'lr_init': 0.001,
    #                 # 'scale': 0.33,
    #                 # 'num_epoch': 60,
    #                 # 'sig': False,
    #                 'n_channelsm[0, 0]': 35,
    #                 # 'n_channelsm[0, 1]': 25,                    
    #                 'n_channels1[0]': 35,
    #                 'n_channelsm[1, 0]': 20,
    #                 # 'n_channelsm[1, 1]': 20,
    #                 'n_channels1[1]': 20,
    #                 'n_channelsm[2, 0]': 20,
    #                 # 'n_channelsm[2, 1]': 15,
    #                 'n_channels1[2]': 20,
    #                 'n_channelsm[3, 0]': 15,
    #                 # 'n_channelsm[3, 1]': 20,
    #                 'n_channels1[3]': 15,
    #                 'n_channelsm[4, 0]': 25,
    #                 # 'n_channelsm[4, 1]': 25,
    #                 'n_channels1[4]': 25,
    #                 'n_channelsm[5, 0]': 35,
    #                 'n_channels1[5]': 35,
    #                 'n_channels1[6]': 35,     
    #                 'n_layers2': 1,
    #                 'n_channels2[0]': 25,
    #                 'n_layers1': 6,
    #                 'n_layersm[0]': 1,
    #                 'n_layersm[1]': 1,
    #                 'n_layersm[2]': 1,
    #                 'n_layersm[3]': 1,
    #                 'n_layersm[4]': 1,
    #                 'n_layersm[5]': 1,
    #                 # 'layernorm' : False,
    #                 # 'drop_rate' : 0.15,
    #                 # 'optim': 'adamw',
    #                 }
    # study.enqueue_trial(init_params)
                            
    study.optimize(objective, n_trials=100, callbacks=[optuna.study.MaxTrialsCallback(200, states=(TrialState.COMPLETE,))])

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
