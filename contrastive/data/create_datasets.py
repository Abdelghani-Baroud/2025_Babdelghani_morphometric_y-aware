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
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.
"""
Tools to create datasets
"""

import pandas as pd
import numpy as np
import os
import copy

# only if foldlabel == True
try:
    from deep_folding.brainvisa.utils.save_data import quality_checks
    from deep_folding.brainvisa.utils.save_data import compare_array_aims_files
except ImportError:
    print("INFO: you cannot use deep_folding in brainvisa. Probably OK.")

from ..utils.logs import set_file_logger, set_root_logger_level

from .datasets import ContrastiveDatasetFusion

from .utils import \
    check_subject_consistency, extract_data, \
    check_if_same_subjects, \
    check_distbottom_npy_consistency, check_foldlabel_npy_consistency, \
    check_extremity_npy_consistency, check_if_same_shape, \
    check_if_skeleton, extract_data_with_labels, read_labels, \
    extract_train_and_val_subjects

import logging

log = set_file_logger(__file__)
root = logging.getLogger()


def sanity_checks_foldlabels_without_labels(config, skeleton_output, reg):
    # Loads and separates in train_val/test set foldlabels if requested
    check_subject_consistency(config.data[reg].subjects_all,
                              config.data[reg].subjects_foldlabel_all,
                              name='foldlabel')
    check_foldlabel_npy_consistency(config.data[reg].numpy_all,
                                    config.data[reg].foldlabel_all)
    # in order to avoid logging twice the same information
    if root.level == 20:  # root logger in INFO mode
        set_root_logger_level(0)
    # add all the other created objects in the next line
    foldlabel_output = extract_data(config.data[reg].foldlabel_all,
                                    config.data[reg].crop_dir,
                                    config, reg)
    if root.level == 10:  # root logger in WARNING mode
        set_root_logger_level(1)
    log.info("foldlabel data loaded")

    # Makes some sanity checks
    for subset_name in foldlabel_output.keys():
        log.debug("skeleton", skeleton_output[subset_name][1].shape)
        log.debug("foldlabel", foldlabel_output[subset_name][1].shape)
        check_if_same_subjects(skeleton_output[subset_name][0],
                               foldlabel_output[subset_name][0],
                               subset_name)
        check_if_same_shape(skeleton_output[subset_name][1],
                            foldlabel_output[subset_name][1],
                            subset_name)

    return foldlabel_output

def sanity_checks_distbottoms_without_labels(config, skeleton_output, reg):
    # Loads and separates in train_val/test set distbottoms if requested
    check_subject_consistency(config.data[reg].subjects_all,
                              config.data[reg].subjects_distbottom_all,
                              name='distbottom')
    check_distbottom_npy_consistency(file_path_arr_crops=config.data[reg].numpy_all,
                                     file_path_arr_distbottom=config.data[reg].distbottom_all)
    # in order to avoid logging twice the same information
    if root.level == 20:  # root logger in INFO mode
        set_root_logger_level(0)
    # add all the other created objects in the next line
    distbottom_output = extract_data(config.data[reg].distbottom_all,
                                    config.data[reg].crop_dir,
                                    config, reg)
    if root.level == 10:  # root logger in WARNING mode
        set_root_logger_level(1)
    log.info("distbottom data loaded")

    # Makes some sanity checks
    for subset_name in distbottom_output.keys():
        log.debug("skeleton", skeleton_output[subset_name][1].shape)
        log.debug("distbottom", distbottom_output[subset_name][1].shape)
        check_if_same_subjects(skeleton_output[subset_name][0],
                               distbottom_output[subset_name][0],
                               subset_name)
        check_if_same_shape(skeleton_output[subset_name][1],
                            distbottom_output[subset_name][1],
                            subset_name)

    return distbottom_output


def sanity_checks_extremities_without_labels(config, skeleton_output, reg):
    # Loads and separates in train_val/test set extremities if requested
    check_subject_consistency(config.data[reg].subjects_all,
                              config.data[reg].subjects_extremity_all,
                              name='extremity')
    check_extremity_npy_consistency(config.data[reg].numpy_all,
                                    config.data[reg].extremity_all)
    # in order to avoid logging twice the same information
    if root.level == 20:  # root logger in INFO mode
        set_root_logger_level(0)
    # add all the other created objects in the next line
    extremity_output = extract_data(config.data[reg].extremity_all,
                                    config.data[reg].crop_dir,
                                    config, reg)
    if root.level == 10:  # root logger in WARNING mode
        set_root_logger_level(1)
    log.info("extremity data loaded")

    # Makes some sanity checks
    for subset_name in extremity_output.keys():
        log.debug("skeleton", skeleton_output[subset_name][1].shape)
        log.debug("extremity", extremity_output[subset_name][1].shape)
        check_if_same_subjects(skeleton_output[subset_name][0],
                               extremity_output[subset_name][0],
                               subset_name)
        check_if_same_shape(skeleton_output[subset_name][1],
                            extremity_output[subset_name][1],
                            subset_name)

    return extremity_output


def create_sets_without_labels_without_load(config):
    """
    Create train / val / train-val / test sets when using individual directories
    and sparse matrices.
    Requires for coord_all in dataset config additionnaly to every modality.
    """
    ## TODO: make the behaviour consistent with other create_sets regarding test and test_intra sets.
    ## be robust to absence of foldlabel or distbottom.
    for reg in range(len(config.data)):
        if 'coords_all' not in config.data[reg].keys():
            raise ValueError("load_sparse requires coords_all in dataset config")
        if not (
            os.path.isdir(config.data[reg].coords_all) and os.path.isdir(config.data[reg].numpy_all) \
            and os.path.isdir(config.data[reg].foldlabel_all) and os.path.isdir(config.data[reg].distbottom_all) \
            and os.path.isdir(config.data[reg].extremity_all)
            ):
            raise ValueError("load_sparse requires numpy directories to be folders, not files")
        
    sub_dirs = {'filenames': [],
                'coords_dirs': [],
                'skeleton_dirs': [],
                'foldlabel_dirs': [],
                'distbottom_dirs': [],
                'extremity_dirs': []}
        
    dirs = {'train': copy.deepcopy(sub_dirs),
            'val': copy.deepcopy(sub_dirs),
            'train_val': copy.deepcopy(sub_dirs),
            'test': copy.deepcopy(sub_dirs)}

    for reg in range(len(config.data)):
        subjects_all = pd.read_csv(config.data[reg].subjects_all)
        # split subjects in train/val/train-val/test
        if 'train_csv_file' in config.data[reg].keys() and 'val_csv_file' in config.data[reg].keys():
            train_subjects = pd.read_csv(config.data[reg]['train_csv_file'], names=['Subject'])
            val_subjects = pd.read_csv(config.data[reg]['val_csv_file'], names=['Subject'])
            train_val_subjects = pd.concat((train_subjects, val_subjects), ignore_index=True)
        elif 'train_val_csv_file' in config.data[reg].keys():
            train_val_subjects = pd.read_csv(config.data[reg]['train_val_csv_file'], names=['Subject'])
            train_subjects, val_subjects = \
                extract_train_and_val_subjects(
                    train_val_subjects, config.partition, config.seed)
        if 'test_csv_file' in config.data[reg].keys():
            test_subjects = pd.read_csv(config.data[reg]['test_csv_file'], names=['Subject'])
        else:
            test_subjects = subjects_all.sample(1) # need not to be empty

        dirs['train']['filenames'].append(train_subjects.reset_index(drop=True))
        dirs['val']['filenames'].append(val_subjects.reset_index(drop=True))
        dirs['train_val']['filenames'].append(train_val_subjects.reset_index(drop=True))
        dirs['test']['filenames'].append(test_subjects.reset_index(drop=True))
        
        for subset in dirs.keys():
            # coords
            coords_dir = config.data[reg].coords_all
            coords_dirs = np.array([os.path.join(coords_dir,f'{sub}_coords.npy') for sub in dirs[subset]['filenames'][reg].Subject])
            #coords_dirs = np.expand_dims(coords_dirs, axis=-1)
            dirs[subset]['coords_dirs'].append(coords_dirs)
            # skels
            skels_dir = config.data[reg].numpy_all
            skeleton_dirs = np.array([os.path.join(skels_dir,f'{sub}_skeleton_values.npy') for sub in dirs[subset]['filenames'][reg].Subject])
            #skeleton_dirs = np.expand_dims(skeleton_dirs, axis=-1)
            dirs[subset]['skeleton_dirs'].append(skeleton_dirs)
            # foldlabels
            foldlabel_dir = config.data[reg].foldlabel_all
            foldlabel_dirs = np.array([os.path.join(foldlabel_dir,f'{sub}_foldlabel_values.npy') for sub in dirs[subset]['filenames'][reg].Subject])
            #foldlabel_dirs = np.expand_dims(foldlabel_dirs, axis=-1)
            dirs[subset]['foldlabel_dirs'].append(foldlabel_dirs)
            # distbottoms
            distbottom_dir = config.data[reg].distbottom_all
            distbottom_dirs = np.array([os.path.join(distbottom_dir,f'{sub}_distbottom_values.npy') for sub in dirs[subset]['filenames'][reg].Subject])
            #distbottom_dirs = np.expand_dims(distbottom_dirs, axis=-1)
            dirs[subset]['distbottom_dirs'].append(distbottom_dirs)
            # extremities
            extremity_dir = config.data[reg].extremity_all
            extremity_dirs = np.array([os.path.join(extremity_dir,f'{sub}_extremities_values.npy') for sub in dirs[subset]['filenames'][reg].Subject])
            #extremity_dirs = np.expand_dims(extremity_dirs, axis=-1)
            dirs[subset]['extremity_dirs'].append(extremity_dirs)

    datasets = {}

    for subset_name in dirs.keys():

        datasets[subset_name] = ContrastiveDatasetFusion(
            filenames=dirs[subset_name]['filenames'], # quelle forme pd ?
            coords_arrays_dirs=dirs[subset_name]['coords_dirs'],
            skeleton_arrays_dirs=dirs[subset_name]['skeleton_dirs'],
            foldlabel_arrays_dirs=dirs[subset_name]['foldlabel_dirs'],
            distbottom_arrays_dirs=dirs[subset_name]['distbottom_dirs'],
            extremity_arrays_dirs=dirs[subset_name]['extremity_dirs'],
            config=config,
            apply_transform=config.apply_augmentations)
        
    return datasets

def create_sets_without_labels(config):
    """Creates train, validation and test sets

    Args:
        config (Omegaconf dict): contains configuration parameters
    Returns:
        train_dataset, val_dataset, test_datasetset, train_val_dataset (tuple)
    """

    skeleton_all = []
    foldlabel_all = []
    distbottom_all = []
    extremity_all = []
    
    # checks consistency among regions
    if len(config.data) > 1:
        for reg in range(len(config.data)-1):
            check_if_same_csv(config.data[0].subjects_all,
                              config.data[reg+1].subjects_all,
                              "subjects_all")
            if 'train_val_csv_file' in config.data[0].keys():
                check_if_same_csv(config.data[0].train_val_csv_file,
                                  config.data[reg+1].train_val_csv_file,
                                  "train_csv")
            else:
                check_if_same_csv(config.data[0].train_csv_file,
                                  config.data[reg+1].train_csv_file,
                                  "train_csv")
            check_if_numpy_same_length(config.data[0].numpy_all,
                                       config.data[reg+1].numpy_all,
                                       "numpy_all")
            if config.foldlabel or config.trimdepth or config.random_choice or config.mixed:
                check_if_numpy_same_length(config.data[0].foldlabel_all,
                                           config.data[reg+1].foldlabel_all,
                                           "foldlabel_all")
            if config.trimdepth or config.random_choice or config.mixed:
                check_if_numpy_same_length(config.data[0].distbottom_all,
                                           config.data[reg+1].distbottom_all,
                                           "distbottom_all")
            if config.random_choice or config.mixed:
                check_if_numpy_same_length(config.data[0].extremity_all,
                                           config.data[reg+1].extremity_all,
                                           "extremity_all")

    for reg in range(len(config.data)):
        # Loads and separates in train_val/test skeleton crops
        skeleton_output = extract_data(
            config.data[reg].numpy_all,
            config.data[reg].crop_dir, config, reg)
        skeleton_all.append(skeleton_output)

        # Loads and separates in train_val/test set foldlabels if requested
        if config.apply_augmentations and (config.foldlabel or config.trimdepth
                                           or config.random_choice or config.mixed):
            foldlabel_output = sanity_checks_foldlabels_without_labels(config,
                                                            skeleton_output,
                                                            reg)
        else:
            foldlabel_output = None
            log.info("foldlabel data NOT requested. Foldlabel data NOT loaded")
        
        foldlabel_all.append(foldlabel_output)

        # same with distbottom
        if config.apply_augmentations and (config.trimdepth or config.random_choice or config.mixed):
            distbottom_output = sanity_checks_distbottoms_without_labels(config,
                                                            skeleton_output,
                                                            reg)
        else:
            distbottom_output = None
            log.info("distbottom data NOT requested. Distbottom data NOT loaded")
        
        distbottom_all.append(distbottom_output)

        # same with extremity # TODO: reduce to single sanity check function
        if config.apply_augmentations and (config.random_choice or config.mixed):
            extremity_output = sanity_checks_extremities_without_labels(config,
                                                            skeleton_output,
                                                            reg)
        else:
            extremity_output = None
            log.info("extremity data NOT requested. extremity data NOT loaded")
        
        extremity_all.append(extremity_output)
            

    # Creates the dataset from these data by doing some preprocessing
    datasets = {}
    for subset_name in skeleton_all[0].keys():
        log.debug(subset_name)
        # Concatenates filenames
        filenames = [skeleton_output[subset_name][0]
                     for skeleton_output in skeleton_all]
        # Concatenates arrays
        arrays = [skeleton_output[subset_name][1]
                  for skeleton_output in skeleton_all]

        # TODO: avoid copy/paste
        # Concatenates foldabel arrays
        foldlabel_arrays = []
        for foldlabel_output in foldlabel_all:
            # select the augmentation method
            if config.apply_augmentations:
                if config.trimdepth or config.random_choice or config.mixed or config.foldlabel:  # branch_clipping
                    foldlabel_array = foldlabel_output[subset_name][1]
                else:  # cutout
                    foldlabel_array = None  # no need of fold labels
            else:  # no augmentation
                foldlabel_array = None
            foldlabel_arrays.append(foldlabel_array)

        # Concatenates distbottom arrays
        distbottom_arrays = []
        for distbottom_output in distbottom_all:
            # select the augmentation method
            if config.apply_augmentations:
                if config.random_choice or config.mixed or config.trimdepth:  # trimdepth
                    distbottom_array = distbottom_output[subset_name][1]
                else:  # cutout
                    distbottom_array = None  # no need of fold labels
            else:  # no augmentation
                distbottom_array = None
            distbottom_arrays.append(distbottom_array)

        # Concatenates extremity arrays
        extremity_arrays = []
        for extremity_output in extremity_all:
            # select the augmentation method
            if config.apply_augmentations:
                if config.random_choice or config.mixed:
                    extremity_array = extremity_output[subset_name][1]
                else:  # cutout
                    extremity_array = None  # no need of fold labels
            else:  # no augmentation
                extremity_array = None
            extremity_arrays.append(extremity_array)

        # Checks if equality of filenames and labels
        check_if_list_of_equal_dataframes(
            filenames,
            "filenames, " + subset_name)

        datasets[subset_name] = ContrastiveDatasetFusion(
            filenames=filenames,
            arrays=arrays,
            foldlabel_arrays=foldlabel_arrays,
            distbottom_arrays=distbottom_arrays,
            extremity_arrays=extremity_arrays,
            config=config,
            apply_transform=config.apply_augmentations)

    return datasets


def sanity_checks_foldlabel_with_labels(config, skeleton_output, subject_labels, reg):
    """Checks alignment of the generated objects."""
    # remove test_intra if not in config
    subsets = [key for key in skeleton_output.keys()]
    if 'test_intra_csv_file' not in config.keys():
        subsets.pop(3)
    log.debug(f"SANITY CHECKS {subsets}")

    for subset_name in subsets:
        check_if_skeleton(skeleton_output[subset_name][1], subset_name)

    if config.environment == "brainvisa" and config.checking:
        for subset_name in subsets:
            compare_array_aims_files(skeleton_output[subset_name][0],
                                     skeleton_output[subset_name][1],
                                     config.data[reg].crop_dir)

    # Makes some sanity checks on ordering of label subjects
    for subset_name in subsets:
        check_if_same_subjects(skeleton_output[subset_name][0][['Subject']],
                               skeleton_output[subset_name][2][['Subject']],
                               f"{subset_name} labels")

    # Loads and separates in train_val/test set foldlabels if requested
    # TODO: add distbottom and extremity to check
    if (
        ('foldlabel' in config.keys())
        and ((config.foldlabel or config.trimdepth
              or config.random_choice or config.mixed))
        and (config.mode != 'evaluation')
    ):
        check_subject_consistency(config.data[reg].subjects_all,
                                  config.data[reg].subjects_foldlabel_all,
                                  subset_name)
        # in order to avoid logging twice the same information
        if root.level == 20:  # root logger in INFO mode
            set_root_logger_level(0)
        foldlabel_output = extract_data_with_labels(
            config.data[reg].foldlabel_all,
            subject_labels,
            config.data[reg].foldlabel_dir,
            config, reg)
        if root.level == 10:  # root logger in WARNING mode
            set_root_logger_level(1)
        log.info("foldlabel data loaded")

        # Makes some sanity checks
        for subset_name in subsets:
            check_if_same_subjects(skeleton_output[subset_name][0],
                                   foldlabel_output[subset_name][0],
                                   subset_name)
            check_if_same_shape(skeleton_output[subset_name][1],
                                foldlabel_output[subset_name][1],
                                subset_name)
            check_if_same_subjects(
                foldlabel_output[subset_name][0],
                skeleton_output[subset_name][2][['Subject']],
                f"{subset_name} labels")
            check_if_same_subjects(
                foldlabel_output[subset_name][2][['Subject']],
                skeleton_output[subset_name][2][['Subject']],
                f"{subset_name} labels")

        if config.environment == "brainvisa" and config.checking:
            for subset_name in foldlabel_output.keys():
                compare_array_aims_files(foldlabel_output[subset_name][0],
                                         foldlabel_output[subset_name][1],
                                         config.data[reg].foldlabel_dir)

    else:
        log.info("foldlabel data NOT requested. Foldlabel data NOT loaded")
        return None

    return foldlabel_output
    

def sanity_checks_distbottoms_with_labels(config, skeleton_output, subject_labels, reg):
    """Checks alignment of the generated objects."""
    # remove test_intra if not in config
    subsets = [key for key in skeleton_output.keys()]
    if 'test_intra_csv_file' not in config.keys():
        subsets.pop(3)
    log.debug(f"SANITY CHECKS {subsets}")

    for subset_name in subsets:
        check_if_skeleton(skeleton_output[subset_name][1], subset_name)

    if config.environment == "brainvisa" and config.checking:
        for subset_name in subsets:
            compare_array_aims_files(skeleton_output[subset_name][0],
                                    skeleton_output[subset_name][1],
                                    config.data[reg].crop_dir)

    # Makes some sanity checks on ordering of label subjects
    for subset_name in subsets:
        check_if_same_subjects(skeleton_output[subset_name][0][['Subject']],
                            skeleton_output[subset_name][2][['Subject']],
                            f"{subset_name} labels")
    if (
        (config.trimdepth or config.random_choice or config.mixed)
        and (config.mode != 'evaluation')
    ):
        check_subject_consistency(config.data[reg].subjects_all,
                                config.data[reg].subjects_distbottom_all,
                                subset_name)
        # in order to avoid logging twice the same information
        if root.level == 20:  # root logger in INFO mode
            set_root_logger_level(0)
        distbottom_output = extract_data_with_labels(
            config.data[reg].distbottom_all,
            subject_labels,
            config.data[reg].distbottom_dir,
            config, reg)
        if root.level == 10:  # root logger in WARNING mode
            set_root_logger_level(1)
        log.info("distbottom data loaded")

        # Makes some sanity checks
        for subset_name in subsets:
            check_if_same_subjects(skeleton_output[subset_name][0],
                                distbottom_output[subset_name][0],
                                subset_name)
            check_if_same_shape(skeleton_output[subset_name][1],
                                distbottom_output[subset_name][1],
                                subset_name)
            check_if_same_subjects(
                distbottom_output[subset_name][0],
                skeleton_output[subset_name][2][['Subject']],
                f"{subset_name} labels")
            check_if_same_subjects(
                distbottom_output[subset_name][2][['Subject']],
                skeleton_output[subset_name][2][['Subject']],
                f"{subset_name} labels")

        if config.environment == "brainvisa" and config.checking:
            for subset_name in distbottom_output.keys():
                compare_array_aims_files(distbottom_output[subset_name][0],
                                        distbottom_output[subset_name][1],
                                        config.data[reg].distbottom_dir)

    else:
        log.info("distbottom data NOT requested. Distbottom data NOT loaded")
        return None

    return distbottom_output


def sanity_checks_extremities_with_labels(config, skeleton_output, subject_labels, reg):
    """Checks alignment of the generated objects."""
    # remove test_intra if not in config
    subsets = [key for key in skeleton_output.keys()]
    if 'test_intra_csv_file' not in config.keys():
        subsets.pop(3)
    log.debug(f"SANITY CHECKS {subsets}")

    for subset_name in subsets:
        check_if_skeleton(skeleton_output[subset_name][1], subset_name)

    if config.environment == "brainvisa" and config.checking:
        for subset_name in subsets:
            compare_array_aims_files(skeleton_output[subset_name][0],
                                     skeleton_output[subset_name][1],
                                     config.data[reg].crop_dir)

    # Makes some sanity checks on ordering of label subjects
    for subset_name in subsets:
        check_if_same_subjects(skeleton_output[subset_name][0][['Subject']],
                               skeleton_output[subset_name][2][['Subject']],
                               f"{subset_name} labels")

    # Loads and separates in train_val/test set foldlabels if requested
    # TODO: add distbottom and extremity to check
    if (
    (config.trimdepth or config.random_choice or config.mixed)
    and (config.mode != 'evaluation')):
        check_subject_consistency(config.data[reg].subjects_all,
                                    config.data[reg].subjects_extremity_all,
                                    subset_name)
                        # in order to avoid logging twice the same information
        if root.level == 20:  # root logger in INFO mode
            set_root_logger_level(0)
        extremity_output = extract_data_with_labels(
            config.data[reg].extremity_all,
            subject_labels,
            config.data[reg].crop_dir,  # Assuming extremity data is also in crop_dir
            config, reg)
        if root.level == 10:  # root logger in WARNING mode
            set_root_logger_level(1)
        log.info("extremity data loaded")

        # Makes some sanity checks
        for subset_name in subsets:
            check_if_same_subjects(skeleton_output[subset_name][0],
                                extremity_output[subset_name][0],
                                subset_name)
            check_if_same_shape(skeleton_output[subset_name][1],
                                extremity_output[subset_name][1],
                                subset_name)
            check_if_same_subjects(
                extremity_output[subset_name][0],
                skeleton_output[subset_name][2][['Subject']],
                f"{subset_name} labels")
            check_if_same_subjects(
                extremity_output[subset_name][2][['Subject']],
                skeleton_output[subset_name][2][['Subject']],
                f"{subset_name} labels")

        if config.environment == "brainvisa" and config.checking:
            for subset_name in extremity_output.keys():
                compare_array_aims_files(extremity_output[subset_name][0],
                                        extremity_output[subset_name][1],
                                        config.data[reg].crop_dir)

    else:
        log.info("extremity data NOT requested. Extremity data NOT loaded")
        return None

    return extremity_output


def check_if_list_of_equal_dataframes(list_of_df, key):
    """Checks if it is a list of equal dataframes"""
    if len(list_of_df) > 1:
        df0 = list_of_df[0]
        for df in list_of_df[1:]:
            if not df0.equals(df):
                raise ValueError(
                    f"List of dataframes are not equal: {key}"
                    "First dataframe head:\n"
                    f"{df0.head()}\n"
                    "Other dataframe head:\n"
                    f"{df.head()}\n"    
                    f"length of first dataframe = {len(df0)}\n"
                    f"length of other dataframe = {len(df)}"               
                    )


def check_if_same_csv(csv_file_1, csv_file_2, key, header='infer'):
    """Checks if the two csv are identical"""
    csv1 = pd.read_csv(csv_file_1, header=header)
    csv2 = pd.read_csv(csv_file_2, header=header)
    if not csv1.equals(csv2):
        raise ValueError(
            f"Input {key} csv files are not equal"
            "First dataframe head:\n"
            f"{csv1.head()}\n"
            "Other dataframe head:\n"
            f"{csv2.head()}\n"
            f"length of first dataframe ({csv_file_1}) = {len(csv1)}\n"
            f"length of other dataframe ({csv_file_2}) = {len(csv2)}"
        )


def check_if_numpy_same_length(npy_file_1, npy_file_2, key):
    """Checks if the two numpy arrays have the same length"""
    arr1 = np.load(npy_file_1)
    arr2 = np.load(npy_file_2)
    if len(arr1) != len(arr2):
        raise ValueError(
            f"Input {key} numpy files don't have the same length"
        )


def create_sets_with_labels(config):
    """Creates train, validation and test sets when there are labels

    Args:
        config (Omegaconf dict): contains configuration parameters
        reg: region number
    Returns:
        train_dataset, val_dataset, test_datasetset, train_val_dataset (tuple)
    """

    skeleton_all = []
    foldlabel_all = []
    distbottom_all = []
    extremity_all = []
    
    # checks consistency among regions
    if len(config.data) > 1:
        for reg in range(len(config.data)-1):
            check_if_same_csv(config.data[0].subject_labels_file,
                              config.data[reg+1].subject_labels_file,
                              "subject_labels")         
            check_if_same_csv(config.data[0].subjects_all,
                              config.data[reg+1].subjects_all,
                              "subjects_all")
            if 'train_val_csv_file' in config.data[0].keys():
                check_if_same_csv(config.data[0].train_val_csv_file,
                                  config.data[reg+1].train_val_csv_file,
                                  "train_val_csv", header=None)
            else:
                check_if_same_csv(config.data[0].train_csv_file,
                                  config.data[reg+1].train_csv_file,
                                  "train_csv")                
            check_if_numpy_same_length(config.data[0].numpy_all,
                                       config.data[1].numpy_all,
                                       "numpy_all")
            if config.foldlabel or config.trimdepth or config.random_choice or config.mixed:
                check_if_numpy_same_length(config.data[0].foldlabel_all,
                                           config.data[1].foldlabel_all,
                                           "foldlabel_all")
            if config.trimdepth or config.random_choice or config.mixed:
                check_if_numpy_same_length(config.data[0].distbottom_all,
                                           config.data[1].distbottom_all,
                                           "distbottom_all")
            if config.random_choice or config.mixed:
                check_if_numpy_same_length(config.data[0].extremity_all,
                                           config.data[1].extremity_all,
                                           "extremity_all")

    for reg in range(len(config.data)):
        # Gets labels for all subjects
        # Column subject_column_name is renamed 'Subject'
        label_scaling = (None if 'label_scaling' not in config.keys()
                         else config.data[reg].label_scaling)
        #retrocompatibility 
        label_names = config.label_names if 'label_names' in config else config.data[0].label_names
        subject_labels = read_labels(
            config.data[reg].subject_labels_file,
            config.data[reg].subject_column_name,
            label_names,
            label_scaling)

        if config.environment == "brainvisa" and config.checking:
            quality_checks(config.data[reg].subjects_all,
                           config.data[reg].numpy_all,
                           config.data[reg].crop_dir, parallel=True)

        # Loads and separates in train_val/test skeleton crops
        skeleton_output = extract_data_with_labels(
            config.data[reg].numpy_all, subject_labels,
            config.data[reg].crop_dir, config, reg)
        
        log.info(f"[DEBUG] apply_augmentations={config.apply_augmentations} "
         f"foldlabel={config.foldlabel} trimdepth={config.trimdepth} "
         f"random_choice={config.random_choice} mixed={config.mixed} "
         f"types: {type(config.foldlabel)}, {type(config.mixed)}")
         
        # Loads and separates in train_val/test set foldlabels if requested
        if config.apply_augmentations and (config.foldlabel or config.trimdepth
                                           or config.random_choice or config.mixed):
            foldlabel_output = sanity_checks_foldlabel_with_labels(config,
                                                         skeleton_output,
                                                         subject_labels,
                                                         reg)
        else:
            foldlabel_output = None
            log.info("foldlabel data NOT requested. Foldlabel data NOT loaded")

        # same with distbottom
        if config.apply_augmentations and (config.trimdepth or config.random_choice or config.mixed):
            if config.with_labels :
               distbottom_output = sanity_checks_distbottoms_with_labels(config,
                                                            skeleton_output,
                                                            subject_labels,
                                                            reg)
            else:       
                distbottom_output = sanity_checks_distbottoms_without_labels(config,
                                                             skeleton_output,
                                                             reg)
        else:
            distbottom_output = None
            log.info("distbottom data NOT requested. Distbottom data NOT loaded")

        # same with extremity
        if config.apply_augmentations and (config.trimdepth or config.random_choice or config.mixed):
            if config.with_labels :
                extremity_output = sanity_checks_extremities_with_labels(config,
                                                            skeleton_output,
                                                            subject_labels,
                                                            reg)
            else:   
                extremity_output = sanity_checks_extremities_without_labels(config,
                                                            skeleton_output,
                                                            reg)
        else:
            extremity_output = None
            log.info("extremity data NOT requested. extremity data NOT loaded")
        
        skeleton_all.append(skeleton_output)
        foldlabel_all.append(foldlabel_output)
        distbottom_all.append(distbottom_output)
        extremity_all.append(extremity_output)


    # Creates the dataset from these data by doing some preprocessing
    datasets = {}
    for subset_name in skeleton_all[0].keys():
        log.debug(subset_name)
        # Concatenates filenames
        filenames = [skeleton_output[subset_name][0]
                     for skeleton_output in skeleton_all]
        # Concatenates arrays
        arrays = [skeleton_output[subset_name][1]
                  for skeleton_output in skeleton_all]

        # TODO: avoid copy/paste
        # Concatenates foldabel arrays
        foldlabel_arrays = []
        for foldlabel_output in foldlabel_all:
            # select the augmentation method
            if config.apply_augmentations:
                if config.trimdepth or config.random_choice or config.mixed or config.foldlabel:  # branch_clipping
                    foldlabel_array = foldlabel_output[subset_name][1]
                else:  # cutout
                    foldlabel_array = None  # no need of fold labels
            else:  # no augmentation
                foldlabel_array = None
            foldlabel_arrays.append(foldlabel_array)

        # Concatenates distbottom arrays
        distbottom_arrays = []
        for distbottom_output in distbottom_all:
            # select the augmentation method
            if config.apply_augmentations:
                if config.random_choice or config.mixed or config.trimdepth:  # trimdepth
                    distbottom_array = distbottom_output[subset_name][1]
                else:  # cutout
                    distbottom_array = None  # no need of fold labels
            else:  # no augmentation
                distbottom_array = None
            distbottom_arrays.append(distbottom_array)

        # Concatenates extremity arrays
        extremity_arrays = []
        for extremity_output in extremity_all:
            # select the augmentation method
            if config.apply_augmentations:
                if config.random_choice or config.mixed:  # trimdepth
                    extremity_array = extremity_output[subset_name][1]
                else:  # cutout
                    extremity_array = None  # no need of fold labels
            else:  # no augmentation
                extremity_array = None
            extremity_arrays.append(extremity_array)

        # Concatenates labels
        labels = [skeleton_output[subset_name][2]
                  for skeleton_output in skeleton_all]
        
        # Convert labels to long
        for label in label_names:
            if label in labels[0].columns:
                if config.label_type == 'continuous':
                   labels[0][label] = labels[0][label].to_numpy().astype(np.float32)
                elif config.label_type == 'binary':
                    labels[0][label] = labels[0][label].to_numpy().astype(np.int64)



        # Checks if equality of filenames and labels
        check_if_list_of_equal_dataframes(
            filenames,
            "filenames, " + subset_name)
        check_if_list_of_equal_dataframes(
            labels,
            "labels, " + subset_name)

        # Builds subset-name=train/val/test dataset
        datasets[subset_name] = ContrastiveDatasetFusion(
            filenames=filenames,
            arrays=arrays,
            foldlabel_arrays=foldlabel_arrays,
            distbottom_arrays=distbottom_arrays,
            extremity_arrays=extremity_arrays,
            labels=labels,
            config=config,
            apply_transform=config.apply_augmentations)

    return datasets
