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
Transforms used in dataset
"""

import torchvision.transforms as transforms
from skimage.morphology import ball
import numpy as np

from contrastive.augmentations import *


def transform_nothing_done():
    return \
        transforms.Compose([
            SimplifyTensor(),
            EndTensor()
        ])


def transform_only_padding(input_size, flip_dataset, config):
    if config.backbone_name != 'pointnet':
        transforms_list = [
                SimplifyTensor(),
                PaddingTensor(shape=input_size,
                              fill_value=config.fill_value),
                BinarizeTensor()]
        if flip_dataset:
            transforms_list.append(FlipFirstAxisTensor())
        transforms_list.append(EndTensor())
        return transforms.Compose(transforms_list)
    else:
        return \
            transforms.Compose([
                SimplifyTensor(),
                PaddingTensor(shape=input_size,
                              fill_value=config.fill_value),
                BinarizeTensor(),
                EndTensor(),
                ToPointnetTensor(n_max=config.n_max)
            ])


def transform_foldlabel(sample_foldlabel, input_size, config):
    transforms_list = [SimplifyTensor(),
                       PaddingTensor(shape=input_size,
                                     fill_value=config.fill_value),
                       RemoveRandomBranchTensor(
                            sample_foldlabel=sample_foldlabel,
                            percentage=config.percentage,
                            variable_percentage=config.variable_percentage,
                            input_size=input_size,
                            keep_extremity=config.keep_extremity),
                       BinarizeTensor(),
                       TrimCropEdges(max_n_voxel=config.vx_crop_edges,
                                     ignore_axis=config.ignore_axis_trim),
                       FlipTensor(ignore_axis=config.ignore_axis_flip,
                                     p=config.flip_proba),
                       TranslateTensor(config.max_translation)]
                       #RotateTensor(max_angle=config.max_angle)]
    if config.backbone_name == 'pointnet':
        transforms_list.append(ToPointnetTensor(n_max=config.n_max))
    if config.sigma_noise > 0:
        transforms_list.append(GaussianNoiseTensor(sigma=config.sigma_noise))
    
    return transforms.Compose(transforms_list)


# OBSOLETE
def transform_no_foldlabel(from_skeleton, input_size, config):
    transforms_list = [SimplifyTensor(),
                       PaddingTensor(shape=input_size,
                                     fill_value=config.fill_value),
                       PartialCutOutTensor_Roll(from_skeleton=from_skeleton,
                                                input_size=input_size,
                                                keep_extremity=config.keep_extremity,
                                                patch_size=config.patch_size),
                       BinarizeTensor(),
                       TrimCropEdges(max_n_voxel=config.vx_crop_edges,
                                     ignore_axis=config.ignore_axis_trim),
                       FlipTensor(ignore_axis=config.ignore_axis_flip,
                                     p=config.flip_proba),
                       TranslateTensor(config.max_translation)]
    if config.backbone_name == 'pointnet':
        transforms_list.append(ToPointnetTensor(n_max=config.n_max))
    if config.sigma_noise > 0:
        transforms_list.append(GaussianNoiseTensor(sigma=config.sigma_noise))
    
    return transforms.Compose(transforms_list)


def transform_cutout(sample_foldlabel, mask_path, input_size, flip_dataset, config):
    mask = np.load(mask_path)
    transforms_list = [SimplifyTensor(),
                       PaddingTensor(shape=input_size,
                                     fill_value=config.fill_value),
                       PartialCutOutTensor_Roll(sample_foldlabel,
                                                mask,
                                                mask_constraint=config.mask_constraint,
                                                from_skeleton=True,
                                                input_size=input_size,
                                                keep_extremity=config.keep_extremity,
                                                keep_proba_per_branch=config.keep_proba_per_branch_cutout,
                                                keep_proba_global=config.keep_proba_global_cutout,
                                                patch_size=config.patch_size_cutout),
                       BinarizeTensor(),
                       TrimCropEdges(max_n_voxel=config.vx_crop_edges,
                                     ignore_axis=config.ignore_axis_trim),
                       FlipTensor(ignore_axis=config.ignore_axis_flip,
                                     p=config.flip_proba),
                       TranslateTensor(config.max_translation)]
    if flip_dataset:
        transforms_list.append(FlipFirstAxisTensor())
    if config.backbone_name == 'pointnet':
        transforms_list.append(ToPointnetTensor(n_max=config.n_max))
    if config.sigma_noise > 0:
        transforms_list.append(GaussianNoiseTensor(sigma=config.sigma_noise))
    
    return transforms.Compose(transforms_list)


def transform_cutin(sample_foldlabel, mask_path, input_size, flip_dataset, config):
    mask = np.load(mask_path)
    transforms_list = [SimplifyTensor(),
                       PaddingTensor(shape=input_size,
                                     fill_value=config.fill_value),
                       PartialCutOutTensor_Roll(sample_foldlabel,
                                                mask,
                                                mask_constraint=config.mask_constraint,
                                                from_skeleton=False,
                                                input_size=input_size,
                                                keep_extremity=config.keep_extremity,
                                                keep_proba_per_branch=config.keep_proba_per_branch_cutin,
                                                keep_proba_global=config.keep_proba_global_cutin,
                                                patch_size=config.patch_size_cutin),
                       BinarizeTensor(),
                       TrimCropEdges(max_n_voxel=config.vx_crop_edges,
                                     ignore_axis=config.ignore_axis_trim),
                       FlipTensor(ignore_axis=config.ignore_axis_flip,
                                     p=config.flip_proba),
                       TranslateTensor(config.max_translation)]
    if flip_dataset:
        transforms_list.append(FlipFirstAxisTensor())
    if config.backbone_name == 'pointnet':
        transforms_list.append(ToPointnetTensor(n_max=config.n_max))
    if config.sigma_noise > 0:
        transforms_list.append(GaussianNoiseTensor(sigma=config.sigma_noise))
    
    return transforms.Compose(transforms_list)


def transform_multicutout(input_size, config):
    transforms_list = [SimplifyTensor(),
                       PaddingTensor(shape=input_size,
                                     fill_value=config.fill_value),
                       MultiCutoutTensor(patch_size=config.small_patch_size,
                                         input_size=input_size,
                                         number_patches=config.nb_patches),
                       BinarizeTensor(),
                       TrimCropEdges(max_n_voxel=config.vx_crop_edges,
                                     ignore_axis=config.ignore_axis_trim),
                       FlipTensor(ignore_axis=config.ignore_axis_flip,
                                     p=config.flip_proba),
                       TranslateTensor(config.max_translation)]
    if config.backbone_name == 'pointnet':
        transforms_list.append(ToPointnetTensor(n_max=config.n_max))
    if config.sigma_noise > 0:
        transforms_list.append(GaussianNoiseTensor(sigma=config.sigma_noise))
    
    return transforms.Compose(transforms_list)


def transform_trimdepth(sample_distbottom, sample_foldlabel,
                        input_size, flip_dataset, config):
    transforms_list = [SimplifyTensor(),
                       PaddingTensor(shape=input_size,
                                     fill_value=config.fill_value),
                       TrimDepthTensor(sample_distbottom=sample_distbottom,
                                       sample_foldlabel=sample_foldlabel,
                                       max_distance=config.max_distance,
                                       delta=config.trimdepth_delta,
                                       input_size=input_size,
                                       keep_extremity=config.keep_extremity_trimdepth,
                                       uniform=config.uniform_trim,
                                       binary=config.binary_trim,
                                       binary_proba=config.binary_proba_trim,
                                       pepper=config.proba_pepper_trimdepth,
                                       redefine_bottom=config.redefine_bottom),
                       BinarizeTensor(),
                       TrimCropEdges(max_n_voxel=config.vx_crop_edges,
                                     ignore_axis=config.ignore_axis_trim),
                       FlipTensor(ignore_axis=config.ignore_axis_flip,
                                     p=config.flip_proba),
                       TranslateTensor(config.max_translation)]
    if flip_dataset:
        transforms_list.append(FlipFirstAxisTensor())
    if config.backbone_name == 'pointnet':
        transforms_list.append(ToPointnetTensor(n_max=config.n_max))
    if config.sigma_noise > 0:
        transforms_list.append(GaussianNoiseTensor(sigma=config.sigma_noise))
    
    return transforms.Compose(transforms_list)


def transform_trimextremities(sample_extremities, sample_foldlabel,
                        input_size, config):
    transforms_list = [SimplifyTensor(),
                       PaddingTensor(shape=input_size,
                                     fill_value=config.fill_value),
                       TrimExtremitiesTensor(sample_extremities=sample_extremities,
                                             sample_foldlabel=sample_foldlabel,
                                             input_size=input_size,
                                             protective_structure=np.expand_dims(ball(config.ball_radius), axis=-1),
                                             p=config.proba_trimedges,
                                             keep_bottom=config.keep_bottom_extremities),
                       BinarizeTensor(),
                       TrimCropEdges(max_n_voxel=config.vx_crop_edges,
                                     ignore_axis=config.ignore_axis_trim),
                       FlipTensor(ignore_axis=config.ignore_axis_flip,
                                     p=config.flip_proba),
                       TranslateTensor(config.max_translation)]
    if config.backbone_name == 'pointnet':
        transforms_list.append(ToPointnetTensor(n_max=config.n_max))
    if config.sigma_noise > 0:
        transforms_list.append(GaussianNoiseTensor(sigma=config.sigma_noise))
    
    return transforms.Compose(transforms_list)

## no binarize tensor !! to keep the value 2.
def transform_highlightextremities(sample_extremities, sample_foldlabel,
                                   input_size, flip_dataset, config):
    transforms_list = [SimplifyTensor(),
                       PaddingTensor(shape=input_size,
                                     fill_value=config.fill_value),
                       HighlightExtremitiesTensor(sample_extremities=sample_extremities,
                                                sample_foldlabel=sample_foldlabel,
                                                input_size=input_size,
                                                protective_structure=np.expand_dims(ball(config.ball_radius), axis=-1),
                                                p=config.proba_trimedges,
                                                pepper=config.proba_pepper_trimedges),
                       TrimCropEdges(max_n_voxel=config.vx_crop_edges,
                                     ignore_axis=config.ignore_axis_trim),
                       FlipTensor(ignore_axis=config.ignore_axis_flip,
                                     p=config.flip_proba),
                       TranslateTensor(config.max_translation)]
    if flip_dataset:
        transforms_list.append(FlipFirstAxisTensor())
    if config.backbone_name == 'pointnet':
        transforms_list.append(ToPointnetTensor(n_max=config.n_max))
    if config.sigma_noise > 0:
        transforms_list.append(GaussianNoiseTensor(sigma=config.sigma_noise))
    
    return transforms.Compose(transforms_list)


def transform_elastic(input_size, config):
    transforms_list = [SimplifyTensor(),
                       PaddingTensor(shape=input_size,
                                     fill_value=config.fill_value),
                       BinarizeTensor(),
                       ElasticDeformTensor(sigma=config.sigma_elastic,
                                           points=config.size_elastic),
                       TrimCropEdges(max_n_voxel=config.vx_crop_edges,
                                     ignore_axis=config.ignore_axis_trim),
                       FlipTensor(ignore_axis=config.ignore_axis_flip,
                                     p=config.flip_proba),
                       TranslateTensor(config.max_translation)]
    if config.backbone_name == 'pointnet':
        transforms_list.append(ToPointnetTensor(n_max=config.n_max))
    if config.sigma_noise > 0:
        transforms_list.append(GaussianNoiseTensor(sigma=config.sigma_noise))

    return transforms.Compose(transforms_list)


def transform_addbranch(input_size, config):
    transforms_list = [SimplifyTensor(),
                       PaddingTensor(shape=input_size,
                                     fill_value=config.fill_value),
                       AddBranchTensor(branch_directory=config.data[0].branch_dir,
                                       nb_branches=config.data[0].nb_branches,
                                       input_size=input_size),
                       BinarizeTensor(),
                       TrimCropEdges(max_n_voxel=config.vx_crop_edges,
                                     ignore_axis=config.ignore_axis_trim),
                       FlipTensor(ignore_axis=config.ignore_axis_flip,
                                     p=config.flip_proba),
                       TranslateTensor(config.max_translation)]
    if config.backbone_name == 'pointnet':
        transforms_list.append(ToPointnetTensor(n_max=config.n_max))
    if config.sigma_noise > 0:
        transforms_list.append(GaussianNoiseTensor(sigma=config.sigma_noise))

    return transforms.Compose(transforms_list)


def transform_noisyedges(input_size, config):
    transforms_list = [SimplifyTensor(),
                       PaddingTensor(shape=input_size,
                                     fill_value=config.fill_value),
                       BinarizeTensor(),
                       TrimCropEdges(max_n_voxel=config.vx_crop_edges,
                                     ignore_axis=config.ignore_axis_trim),
                       FlipTensor(ignore_axis=config.ignore_axis_flip,
                                     p=config.flip_proba),
                       NoisyEdgesTensor(slope=config.slope_noise,
                                        offset=config.offset_noise),
                       TranslateTensor(config.max_translation)]
    if config.backbone_name == 'pointnet':
        transforms_list.append(ToPointnetTensor(n_max=config.n_max))
    if config.sigma_noise > 0:
        transforms_list.append(GaussianNoiseTensor(sigma=config.sigma_noise))
    
    return transforms.Compose(transforms_list)


def transform_translation(input_size, flip_dataset, config):
    transforms_list = [SimplifyTensor(),
                       PaddingTensor(shape=input_size,
                                     fill_value=config.fill_value),
                       BinarizeTensor(),
                       TrimCropEdges(max_n_voxel=config.vx_crop_edges,
                                     ignore_axis=config.ignore_axis_trim),
                       FlipTensor(ignore_axis=config.ignore_axis_flip,
                                     p=config.flip_proba),
                       TranslateTensor(config.max_translation)]
    if flip_dataset:
        transforms_list.append(FlipFirstAxisTensor())
    if config.backbone_name == 'pointnet':
        transforms_list.append(ToPointnetTensor(n_max=config.n_max))
    if config.sigma_noise > 0:
        transforms_list.append(GaussianNoiseTensor(sigma=config.sigma_noise))
    
    return transforms.Compose(transforms_list)


def transform_random(sample_foldlabel,
                     sample_distbottom, sample_extremities,
                     cutout_mask_path, cutin_mask_path, input_size, flip_dataset, config):
    np.random.seed()
    alpha = np.random.uniform()
    if alpha < config.distribution[0]:
        return transform_trimdepth(sample_distbottom,
                                   sample_foldlabel,
                                   input_size, flip_dataset, config)
    elif alpha < config.distribution[1]:
        return transform_cutout(sample_foldlabel, cutout_mask_path, input_size, flip_dataset, config)
    elif alpha < config.distribution[2]:
        return transform_cutin(sample_foldlabel, cutin_mask_path, input_size, flip_dataset, config)
    elif alpha < config.distribution[3]:
        return transform_highlightextremities(sample_extremities,
                                              sample_foldlabel,
                                              input_size, flip_dataset, config)
    else:
        return transform_translation(input_size, flip_dataset, config)
    

def transform_mixed(sample_foldlabel, sample_distbottom,
                    sample_extremities, cutin_mask_path, input_size, config):
    mask=np.load(cutin_mask_path)
    transforms_list = [SimplifyTensor(),
                       PaddingTensor(shape=input_size,
                                     fill_value=config.fill_value),
                       ConcatTensor(sample_foldlabel, sample_distbottom, sample_extremities, mask)]
    np.random.seed()
    r = np.random.uniform()
    if r < config.proba_trimdepth:
        transforms_list.append(
            TrimDepthTensor(
                max_distance=config.max_distance,
                delta=config.trimdepth_delta,
                input_size=input_size,
                keep_extremity=config.keep_extremity_trimdepth,
                uniform=config.uniform_trim,
                binary=config.binary_trim,
                binary_proba=config.binary_proba_trim,
                pepper=config.proba_pepper_trimdepth,
                redefine_bottom=config.redefine_bottom)
        )
    r = np.random.uniform()
    if r < config.proba_trimextremities:
        if config.ball_radius==0:
            protective_structure=None
        else:
            protective_structure=np.expand_dims(ball(config.ball_radius), axis=-1)
        transforms_list.append(
            HighlightExtremitiesTensor(
                                    input_size=input_size,
                                    protective_structure=protective_structure,
                                    p=config.proba_trimedges,
                                    pepper=config.proba_pepper_trimedges,
                                    keep_extremity=None)
        )
    r = np.random.uniform()    
    if r < config.proba_rotation:
        transforms_list.append(RotateTensor(config.max_angle))
    r = np.random.uniform()
    if r < config.proba_cutout + config.proba_cutin:
        r = np.random.uniform()
        # cutout and cutin are mutually exclusive
        if r < config.proba_cutout / (config.proba_cutout + config.proba_cutin):
            from_skeleton=True
            patch_size=config.patch_size_cutout
            keep_proba_per_branch=config.keep_proba_per_branch_cutout
            keep_proba_global=config.keep_proba_global_cutout
            mask_constraint=False
        else:
            from_skeleton=False
            patch_size=config.patch_size_cutin
            keep_proba_per_branch=config.keep_proba_per_branch_cutin
            keep_proba_global=config.keep_proba_global_cutin
            mask_constraint=config.mask_constraint
        transforms_list.append(
            PartialCutOutTensor_Roll(
                                    mask_constraint=mask_constraint,
                                    from_skeleton=from_skeleton,
                                    input_size=input_size,
                                    keep_extremity=config.keep_extremity,
                                    keep_proba_per_branch=keep_proba_per_branch,
                                    keep_proba_global=keep_proba_global,
                                    patch_size=patch_size)
        )

    ## Reduce to one modality
    transforms_list.append(ReduceTensor())
    transforms_list.append(BinarizeTensor())
    r = np.random.uniform()
    if r < config.proba_translation:
        transforms_list.append(TranslateTensor(config.max_translation))
    transforms_list.append(TransposeTensor())

    return transforms.Compose(transforms_list)



def transform_cropresize(input_size, config):
    transforms_list = [SimplifyTensor(),
                       PaddingTensor(shape=input_size,
                                     fill_value=config.fill_value)]
    np.random.seed()
    r = np.random.uniform()
    if r < config.proba_cropresize:
        transforms_list.append(CropResizeTensor(input_size=input_size,
                                                crop_ratio=config.crop_ratio))
    transforms_list.append(BinarizeTensor())
    r = np.random.uniform()
    if r < config.proba_translation:
        transforms_list.append(TranslateTensor(config.max_translation))
    transforms_list.append(TransposeTensor())

    return transforms.Compose(transforms_list)


# DEPRECATED
def transform_both(sample_foldlabel, from_skeleton,
                   input_size, config):
    if config.backbone_name != 'pointnet':
        return \
            transforms.Compose([
                SimplifyTensor(),
                PaddingTensor(shape=input_size,
                              fill_value=config.fill_value),
                RemoveRandomBranchTensor(
                    sample_foldlabel=sample_foldlabel,
                    percentage=config.percentage,
                    variable_percentage=config.variable_percentage,
                    input_size=input_size,
                    keep_bottom=config.keep_bottom),
                PartialCutOutTensor_Roll(from_skeleton=from_skeleton,
                                         input_size=input_size,
                                         keep_bottom=config.keep_bottom,
                                         patch_size=config.patch_size),
                BinarizeTensor(),
                RotateTensor(max_angle=config.max_angle)
            ])
    else:
        return \
            transforms.Compose([
                SimplifyTensor(),
                PaddingTensor(shape=input_size,
                              fill_value=config.fill_value),
                RemoveRandomBranchTensor(
                    sample_foldlabel=sample_foldlabel,
                    percentage=config.percentage,
                    variable_percentage=config.variable_percentage,
                    input_size=input_size,
                    keep_bottom=config.keep_bottom),
                PartialCutOutTensor_Roll(from_skeleton=from_skeleton,
                                         input_size=input_size,
                                         keep_bottom=config.keep_bottom,
                                         patch_size=config.patch_size),
                RotateTensor(max_angle=config.max_angle),
                BinarizeTensor(),
                ToPointnetTensor(n_max=config.n_max)
            ])


def transform_foldlabel_resize(sample_foldlabel,
                               resize_ratio, input_size, config):
    if config.backbone_name != 'pointnet':
        return \
            transforms.Compose([
                SimplifyTensor(),
                RemoveRandomBranchTensor(
                    sample_foldlabel=sample_foldlabel,
                    percentage=config.percentage,
                    variable_percentage=config.variable_percentage,
                    input_size=input_size,
                    keep_bottom=config.keep_bottom),
                BinarizeTensor(),
                ResizeTensor(resize_ratio),
                RotateTensor(max_angle=config.max_angle)
            ])
    else:
        return \
            transforms.Compose([
                SimplifyTensor(),
                RemoveRandomBranchTensor(
                    sample_foldlabel=sample_foldlabel,
                    percentage=config.percentage,
                    variable_percentage=config.variable_percentage,
                    input_size=input_size,
                    keep_bottom=config.keep_bottom),
                RotateTensor(max_angle=config.max_angle),
                BinarizeTensor(),
                ResizeTensor(resize_ratio),
                ToPointnetTensor(n_max=config.n_max)
            ])
