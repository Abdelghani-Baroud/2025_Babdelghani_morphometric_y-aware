#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of fproducing the
# software by the user in light of its specific status of  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.
""" Data module
"""
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, BatchSampler
from torch.utils.data.sampler import Sampler
import random

from .create_datasets import create_sets_with_labels
from .create_datasets import create_sets_without_labels
from .create_datasets import create_sets_without_labels_without_load



class CustomSampler(Sampler):
    """Yield a mini-batch of indices. The sampler will drop the last batch of
            an image size bin if it is not equal to batch_size

    Args:
        data_source (list): Arrays corresponding to specific regions.
        batch_size (int): Size of mini-batch.
    """

    def __init__(self, data_source, batch_size, shuffle=False):
        super(CustomSampler, self).__init__(data_source)
        # build data for sampling here
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        # implement logic of sampling here

        if self.data_source.arrs is not None:
            nb_subjects = len(self.data_source.arrs[0])
            nb_regions = len(self.data_source.arrs)
        elif self.data_source.coords_arrs_dirs is not None:
            nb_subjects = len(self.data_source.coords_arrs_dirs[0])
            nb_regions = len(self.data_source.coords_arrs_dirs)
        else:
            raise ValueError('Neither the array nor the directories are provided.')

        # drop random idxs to get n*batch_size idxs per region
        idx_subjects = [i for i in range(nb_subjects)]
        if self.shuffle:
            random.shuffle(idx_subjects)
        n_to_drop = nb_subjects % self.batch_size
        if n_to_drop!=0:
            idx_subjects = idx_subjects[:-n_to_drop]
        
        # for each region, partition the idxs
        len_idx_subjects = len(idx_subjects)
        nb_batches_per_region = int(len_idx_subjects//self.batch_size)
        partitions_regions_list = []
        for k in range(nb_regions):
            if self.shuffle:
                random.shuffle(idx_subjects)
            rescaled_idx_subjects = [i+k*nb_subjects for i in idx_subjects]
            partitions_list = [rescaled_idx_subjects[k*self.batch_size:(k+1)*self.batch_size]
                                for k in range(nb_batches_per_region)]
            partitions_regions_list.append(partitions_list)
        # reformat to a simple list
        # regions are seen one after the other r1,...rn,r1,...rn,r1... and so on ...
        partition_full_indexes = [partitions_regions_list[region][idx]
                                  for idx in range(nb_batches_per_region)
                                  for region in range(nb_regions)]
        full_indexes = [x for xs in partition_full_indexes for x in xs]
        
        return(iter(full_indexes))

    def __len__(self):
        return (int(len(self.data_source)*len(self.data_source[0])))


class DataModule(pl.LightningDataModule):
    """Parent data module class
    """

    def __init__(self, config):
        super(DataModule, self).__init__()
        self.config = config

    def setup(self, stage=None, mode=None):
        if self.config.with_labels:
            datasets = create_sets_with_labels(self.config)
        else:
            if self.config.load_sparse:
                datasets = create_sets_without_labels_without_load(self.config)
            else:
                datasets = create_sets_without_labels(self.config)

        self.dataset_train = datasets['train']
        self.dataset_val = datasets['val']
        self.dataset_train_val = datasets['train_val']
        self.dataset_test = datasets['test']
        if 'test_intra_csv_file' in self.config.data[0].keys():
            self.dataset_test_intra = datasets['test_intra']


class DataModule_Learning(DataModule):
    """Data module class for Learning
    """

    def __init__(self, config):
        super(DataModule_Learning, self).__init__(config)

    def train_dataloader(self):
        if self.config.multiregion_single_encoder:
            cb_sampler = BatchSampler(CustomSampler(self.dataset_train,
                                                         batch_size=self.config.batch_size,
                                                         shuffle=True),
                                            batch_size=self.config.batch_size,
                                            drop_last=False)
            loader_train = DataLoader(self.dataset_train,
                                    batch_sampler=cb_sampler,
                                    pin_memory=self.config.pin_mem,
                                    multiprocessing_context='fork',
                                    num_workers=self.config.num_cpu_workers)

        else:
            loader_train = DataLoader(self.dataset_train,
                                    batch_size=self.config.batch_size,
                                    sampler=RandomSampler(
                                        data_source=self.dataset_train),
                                    pin_memory=self.config.pin_mem,
                                    multiprocessing_context='fork',
                                    num_workers=self.config.num_cpu_workers,
                                    drop_last=True)
        return loader_train

    def val_dataloader(self):
        if self.config.multiregion_single_encoder:
            cb_sampler = BatchSampler(CustomSampler(self.dataset_val,
                                                         batch_size=self.config.batch_size,
                                                         shuffle=False),
                                            batch_size=self.config.batch_size,
                                            drop_last=False)
            loader_val = DataLoader(self.dataset_val,
                                    batch_sampler=cb_sampler,
                                    pin_memory=self.config.pin_mem,
                                    multiprocessing_context='fork',
                                    num_workers=self.config.num_cpu_workers)
        else:
            loader_val = DataLoader(self.dataset_val,
                                    batch_size=self.config.batch_size,
                                    pin_memory=self.config.pin_mem,
                                    multiprocessing_context='fork',
                                    num_workers=self.config.num_cpu_workers,
                                    shuffle=False,
                                    drop_last=True)
        return loader_val

    def test_dataloader(self):
        loader_test = DataLoader(self.dataset_test,
                                 batch_size=self.config.batch_size,
                                 pin_memory=self.config.pin_mem,
                                 multiprocessing_context='fork',
                                 num_workers=self.config.num_cpu_workers,
                                 shuffle=False,
                                 drop_last=True)
        return loader_test

    def test_intra_dataloader(self):
        if 'test_intra_csv_file' in self.config.data[0].keys():
            loader_test_intra = DataLoader(
                self.dataset_test_intra,
                batch_size=self.config.batch_size,
                pin_memory=self.config.pin_mem,
                multiprocessing_context='fork',
                num_workers=self.config.num_cpu_workers,
                shuffle=False,
                drop_last=True)
            return loader_test_intra
        else:
            raise ValueError(
                "The datamodule used does not have a test_intra set.")


class DataModule_Evaluation(DataModule):
    """Data module class for evaluation/visualization
    """

    def __init__(self, config):
        super(DataModule_Evaluation, self).__init__(config)

    def train_val_dataloader(self):
        loader_train_val = DataLoader(self.dataset_train_val,
                                      batch_size=self.config.batch_size,
                                      pin_memory=self.config.pin_mem,
                                      num_workers=self.config.num_cpu_workers,
                                      shuffle=False
                                      )
        return loader_train_val

    def train_dataloader(self):
        loader_train = DataLoader(self.dataset_train,
                                  batch_size=self.config.batch_size,
                                  pin_memory=self.config.pin_mem,
                                  num_workers=self.config.num_cpu_workers,
                                  shuffle=False
                                  )
        return loader_train

    def val_dataloader(self):
        loader_val = DataLoader(self.dataset_val,
                                batch_size=self.config.batch_size,
                                pin_memory=self.config.pin_mem,
                                num_workers=self.config.num_cpu_workers,
                                shuffle=False
                                )
        return loader_val

    def test_dataloader(self):
        loader_test = DataLoader(self.dataset_test,
                                 batch_size=self.config.batch_size,
                                 pin_memory=self.config.pin_mem,
                                 num_workers=self.config.num_cpu_workers,
                                 shuffle=False
                                 )
        return loader_test

    def test_intra_dataloader(self):
        if 'test_intra_csv_file' in self.config.data[0].keys():
            loader_test_intra = DataLoader(
                self.dataset_test_intra,
                batch_size=self.config.batch_size,
                pin_memory=self.config.pin_mem,
                num_workers=self.config.num_cpu_workers,
                shuffle=False
            )
            return loader_test_intra
        else:
            raise ValueError(
                "The datamodule used does not have a test_intra set.")
