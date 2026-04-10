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
Tools to create pytorch dataloaders
"""
import torch
import numpy as np
import os
import sparse

from contrastive.utils.logs import set_file_logger

from contrastive.data.transforms import \
    transform_foldlabel, transform_cutin, transform_cutout, \
    transform_nothing_done, transform_only_padding,\
    transform_trimdepth, transform_random, transform_mixed,\
    transform_cropresize

from contrastive.data.utils import convert_sparse_to_numpy

from contrastive.augmentations import PaddingTensor

log = set_file_logger(__file__)


def get_sample(arr, idx, type_el):
    """Returns sub-numpy torch tensors corresponding to array of indices idx.

    First axis of arr (numpy array) corresponds to subject nbs from 0 to N-1
    type_el is 'float32' for input, 'int32' for foldlabel
    """
    log.debug(f"idx (in get_sample) = {idx}")
    log.debug(f"shape of arr (in get_sample) = {arr.shape}")
    sample = arr[idx].astype(type_el)

    return torch.from_numpy(sample)


def get_filename(filenames, idx):
    """"Returns filenames corresponding to indices idx

    filenames: dataframe with column name 'ID'
    """
    filename = filenames.Subject[idx]
    log.debug(f"filenames[:5] = {filenames[:5]}")
    log.debug(f"len(filenames) = {len(filenames)}")
    log.debug(f"idx = {idx}, filename[idx] = {filename}")
    log.debug(f"{idx} in filename = {idx in filenames.index}")

    return filename


def get_label(labels, idx):
    """"Returns labels corresponding to indices idx

    labels: dataframe with column name 'Subject'
    """
    label = labels.drop(columns='Subject').values[idx]
    log.debug(f"idx = {idx}, labels[idx] = {label}")
    log.debug(f"{idx} in labels = {idx in labels.index}")

    return label


def check_consistency(filename, labels, idx):
    """Checks if filenames are identical"""
    filename_label = labels.Subject[idx]
    if filename_label != filename:
        raise ValueError("Filenames are not consistent between data and labels"
                         f"For idx = {idx}, filename = {filename}"
                         f"and filename_label = {filename_label}")


def padd_array(sample, input_size, fill_value=0):
    """Padds array according to input_size"""
    transfo = PaddingTensor(
        input_size,
        fill_value=fill_value)
    sample = transfo(sample)
    return sample


def check_equal_non_zero_voxels(sample1, sample2, name):
    b1 = (sample1 > 0)
    if name == "distbottom":
        b2 = (sample2 < 32500)
    else:
        b2 = (sample2 > 0)
    if torch.count_nonzero((b1!=b2) * b2) > 0:
        print(f"{name} volumes are not included in skeleton"
                 f"{torch.count_nonzero((b1!=b2) * b2)} voxels differ "
                 f"over {torch.count_nonzero(b1)} skeleton voxels")
    if torch.count_nonzero((b1!=b2) * b1) > (b2.numel()/10):
        print(f"Skeleton and {name} volumes do not have the same number "
                 "of non-zero voxels. "
                 f"{torch.count_nonzero((b1!=b2) * b1)} voxels differ "
                 f"over {torch.count_nonzero(b1)} skeleton voxels")


class ContrastiveDatasetFusion():
    """Custom dataset that includes image file paths.

    Applies different transformations to data depending on the type of input.
    """

    def __init__(self, filenames, config, apply_transform=True,
                 labels=None, arrays=None, foldlabel_arrays=None,
                 distbottom_arrays=None, extremity_arrays=None,
                 coords_arrays_dirs=None, skeleton_arrays_dirs=None,
                 foldlabel_arrays_dirs=None, distbottom_arrays_dirs=None,
                 extremity_arrays_dirs=None):
        """
        Every data argument is a list over regions

        Args:
            data_tensor (tensor): contains MRIs as numpy arrays
            filenames (list of strings): list of subjects' IDs
            config (Omegaconf dict): contains configuration information
        """
        self.labels=labels
        self.arrs=arrays
        self.foldlabel_arrs=foldlabel_arrays
        self.distbottom_arrs=distbottom_arrays
        self.extremity_arrs=extremity_arrays
        self.nb_train=len(filenames[0])
        self.filenames=filenames
        self.config=config
        self.transform=apply_transform
        self.coords_arrs_dirs=coords_arrays_dirs
        self.skeleton_arrs_dirs=skeleton_arrays_dirs
        self.foldlabel_arrs_dirs=foldlabel_arrays_dirs
        self.distbottom_arrs_dirs=distbottom_arrays_dirs
        self.extremity_arrs_dirs=extremity_arrays_dirs

        log.debug(f"nb_train = {self.nb_train}")
        log.debug(f"filenames[:5] = {filenames[:5]}")
        if labels is not None and labels[0].shape[0] > 0:
            label0 = labels[0]
            log.debug(f"labels[:5] = {label0[:5]}")
            log.debug(f"There are {label0[label0[config.label_names[0]].isna()].shape[0]} NaN labels")
            log.debug(label0[label0[config.label_names[0]].isna()])

    def __len__(self):
        if self.config.multiregion_single_encoder:
            if self.arrs is not None:
                return (self.nb_train*len(self.arrs))
            else:
                return (self.nb_train*len(self.coords_arrs_dirs))
        else:
            return (self.nb_train)

    def __getitem__(self, idx):
        """Returns the two views corresponding to index idx

        The two views are generated on the fly.

        Returns:
            tuple of (views, subject ID)
        """
        if torch.is_tensor(idx):
            if self.transform:
                idx = idx.tolist()
            else:
                idx = idx.tolist(self.nb_train)
        if self.config.multiregion_single_encoder:
            idx_region, idx = idx // self.nb_train, idx%self.nb_train

        # Gets data corresponding to idx
        log.debug(f"length = {self.nb_train}")
        log.debug(f"filenames = {self.filenames[0]}")
        # get filenames
        if self.config.multiregion_single_encoder:
            filename = self.filenames[idx_region]
            filenames = [get_filename(filename, idx)]
        else:
            filenames = [get_filename(filename, idx)
                        for filename in self.filenames]
        # if arrays loaded
        if self.arrs is not None and self.arrs[0] is not None:
            if self.config.multiregion_single_encoder:
                # TODO: use same code as single region, but add condition in for loop ?
                # arr = [self.arrs[idx_region]] # 1 element list
                # apply a function which loops on the list
                arr = self.arrs[idx_region]
                samples = get_sample(arr, idx, 'float32')
                samples = [padd_array(samples,
                                    self.config.data[idx_region].input_size,
                                    fill_value=0)]
            else:
                # TODO: create a build sample function
                samples = [get_sample(arr, idx, 'float32')
                        for arr in self.arrs]
                samples = [padd_array(sample,
                                    self.config.data[reg].input_size,
                                    fill_value=0)
                            for reg, sample in enumerate(samples)]

        if self.foldlabel_arrs is not None and self.foldlabel_arrs[0] is not None:
            if self.config.multiregion_single_encoder:
                foldlabel_arr = self.foldlabel_arrs[idx_region]
                sample_foldlabels = get_sample(foldlabel_arr, idx, 'int32')
                sample_foldlabels = [padd_array(sample_foldlabels,
                                                self.config.data[idx_region].input_size,
                                                fill_value=0)]
            else:
                sample_foldlabels = [get_sample(foldlabel_arr, idx, 'int32')
                                    for foldlabel_arr in self.foldlabel_arrs]
                sample_foldlabels = [padd_array(sample_foldlabel,
                                                self.config.data[reg].input_size,
                                                fill_value=0)
                                    for reg, sample_foldlabel in enumerate(sample_foldlabels)]
            for s, f in zip(samples, sample_foldlabels):
                check_equal_non_zero_voxels(s, f, "foldlabel")

        if self.distbottom_arrs is not None and self.distbottom_arrs[0] is not None:
            if self.config.multiregion_single_encoder:
                distbottoms_arr = self.distbottom_arrs[idx_region]
                sample_distbottoms = get_sample(distbottoms_arr, idx, 'int32')
                sample_distbottoms = [padd_array(sample_distbottoms,
                                                 self.config.data[idx_region].input_size,
                                                 fill_value=32500)]
            else:
                sample_distbottoms = [get_sample(distbottom_arr, idx, 'int32')
                                    for distbottom_arr in self.distbottom_arrs]
                sample_distbottoms = [padd_array(sample_distbottom,
                                                    self.config.data[reg].input_size,
                                                    fill_value=32500)
                                    for reg, sample_distbottom in enumerate(sample_distbottoms)]
            for s, d in zip(samples, sample_distbottoms):
                check_equal_non_zero_voxels(s, d, "distbottom")
        
        if self.extremity_arrs is not None and self.extremity_arrs[0] is not None:
            if self.config.multiregion_single_encoder:
                extremity_arr = self.extremity_arrs[idx_region]
                sample_extremities = get_sample(extremity_arr, idx, 'int32')
                sample_extremities = [padd_array(sample_extremities,
                                                self.config.data[idx_region].input_size,
                                                fill_value=0)]
            else:
                sample_extremities = [get_sample(extremity_arr, idx, 'int32')
                                    for extremity_arr in self.extremity_arrs]
                sample_extremities = [padd_array(sample_extremity,
                                                self.config.data[reg].input_size,
                                                fill_value=0)
                                    for reg, sample_extremity in enumerate(sample_extremities)]
            #for s, f in zip(samples, sample_extremities):
            #    check_equal_non_zero_voxels(s, f, "extremity") # TODO: check inclusion only ?

        
        # if path given instead
        if self.coords_arrs_dirs is not None and self.coords_arrs_dirs[0] is not None:
            if self.config.multiregion_single_encoder:
                coords_arr_dir = self.coords_arrs_dirs[idx_region][idx]
                coords_arr = np.load(coords_arr_dir)
            else:
                coords_arr_dir = [arr[idx] for arr in self.coords_arrs_dirs]
                coords_arrs = [np.load(coords_dir) for coords_dir in coords_arr_dir]
        if self.skeleton_arrs_dirs is not None and self.skeleton_arrs_dirs[0] is not None:
            # TODO: build_sample_from_path ## nb: or from sparse ?
            # and copy the architecture from above.
            if self.config.multiregion_single_encoder:
                skeleton_arr_dir = self.skeleton_arrs_dirs[idx_region][idx]
                skeleton_arr = np.load(skeleton_arr_dir)
                samples = convert_sparse_to_numpy(skeleton_arr, coords_arr,
                                                  self.config.data[idx_region].input_size[1:], 'float32')
                samples = torch.from_numpy(samples)
                samples = [padd_array(samples,
                                    self.config.data[idx_region].input_size,
                                    fill_value=0)]
            else:
                skeleton_arr_dir = [arr[idx] for arr in self.skeleton_arrs_dirs]
                skeleton_arrs = [np.load(skeleton_dir) for skeleton_dir in skeleton_arr_dir]
                samples = [convert_sparse_to_numpy(skeleton_arr, coords_arr,
                                                  self.config.data[reg].input_size[1:], 'float32')
                                                  for reg, (skeleton_arr, coords_arr)
                                                  in enumerate(zip(skeleton_arrs, coords_arrs))]
                samples = [torch.from_numpy(sample) for sample in samples]
                samples = [padd_array(sample,
                                    self.config.data[reg].input_size,
                                    fill_value=0)
                           for reg, sample in enumerate(samples)]
        if self.foldlabel_arrs_dirs is not None and self.foldlabel_arrs_dirs[0] is not None:
            if self.config.multiregion_single_encoder:
                foldlabel_arr_dir = self.foldlabel_arrs_dirs[idx_region][idx]
                foldlabel_arr = np.load(foldlabel_arr_dir)
                sample_foldlabels = convert_sparse_to_numpy(foldlabel_arr, coords_arr,
                                                            self.config.data[idx_region].input_size[1:], 'int32')
                sample_foldlabels = torch.from_numpy(sample_foldlabels)
                sample_foldlabels = [padd_array(sample_foldlabels,
                                    self.config.data[idx_region].input_size,
                                    fill_value=0)]
            else:
                foldlabel_arr_dir = [arr[idx] for arr in self.foldlabel_arrs_dirs]
                foldlabel_arrs = [np.load(foldlabel_dir) for foldlabel_dir in foldlabel_arr_dir]
                sample_foldlabels = [convert_sparse_to_numpy(foldlabel_arr, coords_arr,
                                                  self.config.data[reg].input_size[1:], 'int32')
                                                  for reg, (foldlabel_arr, coords_arr)
                                                  in enumerate(zip(foldlabel_arrs, coords_arrs))]
                sample_foldlabels = [torch.from_numpy(sample_foldlabel) for sample_foldlabel in sample_foldlabels]
                sample_foldlabels = [padd_array(sample_foldlabel,
                                    self.config.data[reg].input_size,
                                    fill_value=0)
                           for reg, sample_foldlabel in enumerate(sample_foldlabels)]
            for s, f in zip(samples, sample_foldlabels):
                check_equal_non_zero_voxels(s, f, "foldlabel")
        if self.distbottom_arrs_dirs is not None and self.distbottom_arrs_dirs[0] is not None:
            if self.config.multiregion_single_encoder:
                distbottom_arr_dir = self.distbottom_arrs_dirs[idx_region][idx]
                distbottom_arr = np.load(distbottom_arr_dir)
                sample_distbottoms = convert_sparse_to_numpy(distbottom_arr, coords_arr,
                                                             self.config.data[idx_region].input_size[1:], 'int32')
                # sparse distbottoms had value 0 for out of skeleton voxels
                # and -1 for bottoms, they need to be reformated
                sample_distbottoms[sample_distbottoms==0]=32500
                sample_distbottoms[sample_distbottoms==-1]=0
                sample_distbottoms = torch.from_numpy(sample_distbottoms)
                sample_distbottoms = [padd_array(sample_distbottoms,
                                    self.config.data[idx_region].input_size,
                                    fill_value=32500)]
            else:
                distbottom_arr_dir = [arr[idx] for arr in self.distbottom_arrs_dirs]
                distbottom_arrs = [np.load(distbottom_dir) for distbottom_dir in distbottom_arr_dir]
                sample_distbottoms = [convert_sparse_to_numpy(distbottom_arr, coords_arr,
                                                  self.config.data[reg].input_size[1:], 'int32')
                                                  for reg, (distbottom_arr, coords_arr)
                                                  in enumerate(zip(distbottom_arrs, coords_arrs))]
                # sparse distbottoms had value 0 for out of skeleton voxels
                # and -1 for bottoms, they need to be reformated
                for reg, sample_distbottom in enumerate(sample_distbottoms):
                    sample_distbottom[sample_distbottom==0]=32500
                    sample_distbottom[sample_distbottom==-1]=0
                    sample_distbottoms[reg]=sample_distbottom
                sample_distbottoms = [torch.from_numpy(sample_distbottom) for sample_distbottom in sample_distbottoms]
                sample_distbottoms = [padd_array(sample_distbottom,
                                    self.config.data[reg].input_size,
                                    fill_value=32500)
                           for reg, sample_distbottom in enumerate(sample_distbottoms)]
            for s, d in zip(samples, sample_distbottoms):
                check_equal_non_zero_voxels(s, d, "distbottom")  
        if self.extremity_arrs_dirs is not None and self.extremity_arrs_dirs[0] is not None:
            if self.config.multiregion_single_encoder:
                extremity_arr_dir = self.extremity_arrs_dirs[idx_region][idx]
                extremity_arr = np.load(extremity_arr_dir)
                sample_extremities = convert_sparse_to_numpy(extremity_arr, coords_arr,
                                                            self.config.data[idx_region].input_size[1:], 'int32')
                sample_extremities[sample_extremities==-1]=0
                sample_extremities = torch.from_numpy(sample_extremities)
                sample_extremities = [padd_array(sample_extremities,
                                    self.config.data[idx_region].input_size,
                                    fill_value=0)]
            else:
                extremity_arr_dir = [arr[idx] for arr in self.extremity_arrs_dirs]
                extremity_arrs = [np.load(extremity_dir) for extremity_dir in extremity_arr_dir]
                sample_extremities = [convert_sparse_to_numpy(extremity_arr, coords_arr,
                                                  self.config.data[reg].input_size[1:], 'int32')
                                                  for reg, (extremity_arr, coords_arr)
                                                  in enumerate(zip(extremity_arrs, coords_arrs))]
                for reg, sample_extremity in enumerate(sample_extremities):
                    sample_extremity[sample_extremity==-1]=0
                    sample_extremities[reg]=sample_extremity
                sample_extremities = [torch.from_numpy(sample_extremity) for sample_extremity in sample_extremities]
                sample_extremities = [padd_array(sample_extremity,
                                    self.config.data[reg].input_size,
                                    fill_value=0)
                           for reg, sample_extremity in enumerate(sample_extremities)]
            #for s, f in zip(samples, sample_extremities):
            #    check_equal_non_zero_voxels(s, f, "extremity") TODO: check inclusion only ?
        

        if self.labels is not None:
            for reg in range(len(filenames)):
                check_consistency(filenames[reg], self.labels[reg], idx)
            labels = [get_label(label, idx) for label in self.labels]

        self.transform1 = []
        self.transform2 = []
        self.transform3 = []

        if self.config.multiregion_single_encoder:
            regs = [0]
            input_sizes = [self.config.data[idx_region].input_size]
            mask_paths = [self.config.data[idx_region].mask_path]
            flips = [self.config.data[idx_region].flip_dataset]
        else:
            regs = range(len(filenames))
            input_sizes = [self.config.data[reg].input_size for reg in regs]
            cutout_mask_paths = [self.config.data[reg].cutout_mask_path for reg in regs]
            cutin_mask_paths = [self.config.data[reg].cutin_mask_path for reg in regs]
            flips = [self.config.data[reg].flip_dataset for reg in regs]
        # compute the transforms
        for reg, cutout_mask_path, cutin_mask_path, input_size, flip in zip(regs, cutout_mask_paths, cutin_mask_paths, input_sizes, flips):
            if self.transform:
                # mix of branch clipping, cutout, cutin, trimdepth, and trimextremities
                if self.config.random_choice:
                    transform1 = transform_random(
                        sample_foldlabels[reg],
                        sample_distbottoms[reg],
                        sample_extremities[reg],
                        cutout_mask_path=cutout_mask_path,
                        cutin_mask_path=cutin_mask_path,
                        input_size=input_size,
                        flip_dataset=flip,
                        config=self.config)
                    transform2 = transform_random(
                        sample_foldlabels[reg],
                        sample_distbottoms[reg],
                        sample_extremities[reg],
                        cutout_mask_path=cutout_mask_path,
                        cutin_mask_path=cutin_mask_path,
                        input_size=input_size,
                        flip_dataset=flip,
                        config=self.config)
                elif self.config.mixed:
                    transform1 = transform_mixed(
                        sample_foldlabels[reg],
                        sample_distbottoms[reg],
                        sample_extremities[reg],
                        cutin_mask_path=cutin_mask_path,
                        input_size=input_size,
                        config=self.config)
                    transform2 = transform_mixed(
                        sample_foldlabels[reg],
                        sample_distbottoms[reg],
                        sample_extremities[reg],
                        cutin_mask_path=cutin_mask_path,
                        input_size=input_size,
                        config=self.config)
                # branch clipping
                elif self.config.foldlabel:
                    transform1 = transform_foldlabel(
                        sample_foldlabels[reg],
                        input_size,
                        self.config)
                    transform2 = transform_foldlabel(
                        sample_foldlabels[reg],
                        input_size,
                        self.config)
                # trimdepth
                elif self.config.trimdepth:
                        transform1 = transform_trimdepth(
                            sample_distbottoms[reg],
                            sample_foldlabels[reg],
                            input_size,
                            self.config)
                        transform2 = transform_trimdepth(
                            sample_distbottoms[reg],
                            sample_foldlabels[reg],
                            input_size,
                            self.config)
                # cropresize
                elif self.config.cropresize:
                    transform1 = transform_cropresize(
                        input_size, self.config)
                    transform2 = transform_cropresize(
                        input_size, self.config)
                # cutout with or without noise
                else:
                    transform1 = transform_cutout(
                        mask_path=mask_path,
                        input_size=input_size,
                        config=self.config)
                    transform2 = transform_cutin(
                        mask_path=mask_path,
                        input_size=input_size,
                        config=self.config)
                    
            else:
                transform1 = transform_only_padding(
                    input_size, flip, self.config)
                transform2 = transform_only_padding(
                    input_size, flip, self.config)
            self.transform1.append(transform1)
            self.transform2.append(transform2)

            if self.config.with_labels:
                if self.config.mode == "decoder":
                    transform3 = transform_only_padding(
                        input_size,
                        flip,
                        self.config)
                else:
                    transform3 = transform_nothing_done()
                    if not self.transform:
                        transform3 = transform_only_padding(
                            input_size,
                            flip,
                            self.config)
                self.transform3.append(transform3)

        # Computes the views
        view1 = []
        view2 = []
        for reg in range(len(filenames)):
            view1.append(self.transform1[reg](samples[reg]))
            view2.append(self.transform2[reg](samples[reg]))

        # Computes the outputs as tuples
        concatenated_tuple = ()
        # loop over input datasets
        for reg in range(len(filenames)):
            if self.config.mode == "decoder":
                view3 = self.transform3(samples[reg])
                views = torch.stack((view1, view2, view3), dim=0)
                if self.config.with_labels:
                    tuple_with_path = ((views, filenames[reg], labels),)
                elif self.config.multiregion_single_encoder and \
                    self.config.multiple_projection_heads:
                    tuple_with_path = ((views, filenames[reg], idx_region),) # does it make sens for decoder ?
                else:
                    tuple_with_path = ((views, filenames[reg]),)
            else:
                views = torch.stack((view1[reg], view2[reg]), dim=0)
                if self.config.with_labels:
                    view3 = self.transform3[reg](samples[reg])
                    tuple_with_path = (
                        (views, filenames[reg], labels[reg], view3),)
                elif self.config.multiregion_single_encoder or \
                    self.config.multiple_projection_heads:
                    tuple_with_path = ((views, filenames[reg], idx_region),)
                else:
                    tuple_with_path = ((views, filenames[reg]),)
            concatenated_tuple += tuple_with_path

        return concatenated_tuple
