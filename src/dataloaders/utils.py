import numpy as np
import torch
import logging, glob
import logging
logger = logging.getLogger(__name__)

from torch.utils.data import ConcatDataset
from . import JetDataset

def initialize_datasets(args, datadir='../../data/sample_data', num_pts=None, testfile='', balance=True, RAMdataset=True):
    """
    Initialize datasets.
    """

    ### ------ 1: Get the file names ------ ###
    # datadir should be the directory in which the HDF5 files (e.g. out_test.h5, out_train.h5, out_valid.h5) reside.
    # There may be many data files, in some cases the test/train/validate sets may themselves be split across files.
    # We will look for the keywords defined in splits to be be in the filenames, and will thus determine what
    # set each file belongs to.
    splits = ['train', 'test', 'valid'] # We will consider all HDF5 files in datadir with one of these keywords in the filename
    randomize_subset = {'train': True, 'valid': True, 'test': False} # Shuffle only the training set
    RAMdataset_splits = {'train': RAMdataset, 'valid': RAMdataset, 'test': RAMdataset} # always load validation set into RAM for speed (hopefully it's not too large)
    datafiles = {split:[] for split in splits}

    # now search datadir for h5 files and assign them to splits based on their filenames
    files = sorted(glob.glob(datadir + '/*.h5'))
    for split in splits:
        logger.info(f'Looking for {split} files in datadir:')
        for filename in files:
            if (split in filename.rsplit("/",1)[-1]):
                datafiles[split].append(filename)
                logger.info(filename)

    print("Globbing for:", datadir + '/*.h5')
    print("Glob found files:", files)


    # if a testfile is explicitly provided, that will override any test sets found in datadir
    if testfile != '': 
        datafiles['test']=[testfile]
        logger.info(f'Using the explicitly specified test dataset:')
        logger.info(testfile)

    nfiles = {split:len(datafiles[split]) for split in splits}

    ### ------ 2: Set the number of data points ------ ###
    # There will be a JetDataset for each file, so we divide number of data points by number of files,
    # to get data points per file. (Integer division -> must be careful!) #TODO: nfiles > npoints might cause issues down the line, but it's an absurd use case
    if num_pts is None:
        num_pts={'train': args.num_train, 'test': args.num_test, 'valid': args.num_valid}
        
    num_pts_per_file = {}
    for split in splits:
        num_pts_per_file[split] = []
        
        if num_pts[split] == -1:
            num_pts_per_file[split] = [-1 for _ in range(nfiles[split])]
        else:
            num_pts_per_file[split] = [int(np.floor(num_pts[split]/nfiles[split])) for _ in range(nfiles[split])]
            if nfiles[split]>0:
                num_pts_per_file[split][-1] = int(np.maximum(num_pts[split] - np.sum(np.array(num_pts_per_file[split])[0:-1]),0))
    
    ### ------ 3: Load the data ------ ###
    datasets = {}
    for split in splits:
        datasets[split] = []
        for filename in datafiles[split]:
                datasets[split].append(filename)
 
    ### ------ 4: Error checking ------ ###
    # Basic error checking: Check the files belonging to the same split have the same set of keys.
    # for split in splits:
    #     keys = []
    #     for dataset in datasets[split]:
    #         keys.append(dataset.keys())
        # assert all([key == keys[0] for key in keys]), 'Datasets must have same set of keys!'

    ### ------ 5: Initialize datasets ------ ###
    # Now initialize datasets based upon loaded data
    torch_datasets = {split: ConcatDatasetChild([JetDataset(filename, num_pts=num_pts_per_file[split][idx], randomize_subset=randomize_subset[split], balance=balance, RAMdataset=RAMdataset_splits[split]) for idx, filename in enumerate(datasets[split]) if num_pts_per_file[split][idx]!=0]) for split in splits if len(datasets[split])>0}

    # Now, update the number of training/test/validation sets in args
    if 'train' in torch_datasets.keys():
        args.num_train = torch_datasets['train'].cumulative_sizes[-1]
    if 'test' in torch_datasets.keys():
        args.num_test = torch_datasets['test'].cumulative_sizes[-1]
    if 'valid' in torch_datasets.keys():
        args.num_valid = torch_datasets['valid'].cumulative_sizes[-1]

    return args, torch_datasets



class ConcatDatasetChild(ConcatDataset):
    # Dummy class to allow one to quickly iterate through the dataset without actually reading any real data
    def __init__(self, list):
        ConcatDataset.__init__(self, list)
        self.fast_skip=False
    
    def __getitem__(self, idx):
        if self.fast_skip:
            return None
        return ConcatDataset.__getitem__(self, idx)
