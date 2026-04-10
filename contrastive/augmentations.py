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
# knowledge of the CeCILL license version 2 and that you ac8
import numbers
from collections import namedtuple

import os
import numpy as np
import torch
from scipy.ndimage import rotate, zoom, binary_erosion
from sklearn.preprocessing import OneHotEncoder
from kornia.augmentation import RandomRotation3D
#import elasticdeform

from contrastive.utils import logs
from contrastive.utils.test_timeit import timeit
from contrastive.data.utils import zero_padding, repeat_padding, pad, convert_sparse_to_numpy

log = logs.set_file_logger(__file__)


def rotate_list(l_list):
    "Rotates list by -1"
    return l_list[1:] + l_list[:1]


def checkerboard(shape, tile_size):
    return (np.indices(shape) // tile_size).sum(axis=0) % 2


def mask_array_with_skeleton(array, skeleton, cval):
    array[skeleton==0]=cval
    return array


class PaddingTensor(object):
    """A class to pad a tensor"""

    def __init__(self, shape, nb_channels=1, fill_value=0):
        """ Initialize the instance.
        Parameters
        ----------
        shape: list of int
            the desired shape.
        nb_channels: int, default 1
            the number of channels.
        fill_value: int or list of int, default 0
            the value used to fill the array, if a list is given, use the
            specified value on each channel.
        """
        self.shape = rotate_list(shape)
        self.nb_channels = nb_channels
        self.fill_value = fill_value
        if self.nb_channels > 1 and not isinstance(self.fill_value, list):
            self.fill_value = [self.fill_value] * self.nb_channels
        elif isinstance(self.fill_value, list):
            assert len(self.fill_value) == self.nb_channels()

    def __call__(self, tensor):
        """ Fill a tensor to fit the desired shape.
        Parameters
        ----------
        tensor: torch.tensor
            an input tensor.
        Returns
        -------
        fill_tensor: torch.tensor
            the fill_value padded tensor.
        """
        if len(tensor.shape) - len(self.shape) == 1:
            data = []
            for _tensor, _fill_value in zip(tensor, self.fill_value):
                data.append(self._apply_padding(_tensor, _fill_value))
            return torch.from_numpy(np.asarray(data))
        elif len(tensor.shape) - len(self.shape) == 0:
            return self._apply_padding(tensor, self.fill_value)
        else:
            raise ValueError("Wrong input shape specified!")

    def _apply_padding(self, tensor, fill_value):
        """ See Padding.__call__().
        """
        arr = tensor.numpy()
        orig_shape = arr.shape
        padding = []
        # print(f"SHAPES: {orig_shape} - {self.shape}")
        for orig_i, final_i in zip(orig_shape, self.shape):
            shape_i = final_i - orig_i
            half_shape_i = shape_i // 2
            if shape_i % 2 == 0:
                padding.append((half_shape_i, half_shape_i))
            else:
                padding.append((half_shape_i, half_shape_i + 1))
        for cnt in range(len(arr.shape) - len(padding)):
            padding.append((0, 0))

        fill_arr = np.pad(arr, padding, mode="constant",
                          constant_values=fill_value)

        # fill_arr = np.reshape(fill_arr, (1,) + fill_arr.shape[:-1])

        return torch.from_numpy(fill_arr)


class EndTensor(object):
    """Puts all internal and external values to background value 0
    """

    def __init__(self):
        None

    def __call__(self, tensor):
        arr = tensor.numpy()
        arr = np.reshape(arr, (1,) + arr.shape[:-1])
        return torch.from_numpy(arr)


class SimplifyTensor(object):
    """Puts all internal and external values to background value 0
    """

    def __init__(self):
        None

    def __call__(self, tensor):
        arr = tensor.numpy()
        arr[arr == 11] = 0
        return torch.from_numpy(arr)


class OnlyBottomTensor(object):
    """Keeps only bottom '30' values, puts everything else to '0'
    """

    def __init__(self):
        None

    def __call__(self, tensor):
        arr = tensor.numpy()
        arr = arr * (arr == 30)
        return torch.from_numpy(arr)


class BinarizeTensor(object):
    """Puts non-zero values to 1
    """

    def __init__(self):
        None

    def __call__(self, tensor):
        arr = tensor.numpy()
        arr[arr > 0] = 1
        return torch.from_numpy(arr)
    

class ReduceTensor(object):
    """Keep only the first modality (skeleton)
    """

    def __init__(self):
        None

    def __call__(self, tensor):
        arr = tensor.numpy()
        arr_skel = arr[0]
        return torch.from_numpy(arr_skel)
    

class ConcatTensor(object):
    """Concat skeleton with foldlabel, distbottom and edges.
    """

    def __init__(self, sample_foldlabel, sample_distbottom, sample_extremities, mask):
        self.sample_foldlabel = sample_foldlabel
        self.sample_distbottom = sample_distbottom
        self.sample_extremities = sample_extremities
        self.mask = mask

    def __call__(self, tensor):
        arr = tensor.numpy()
        arr_foldlabel = self.sample_foldlabel.numpy()
        arr_distbottom = self.sample_distbottom.numpy()
        arr_extremities = self.sample_extremities.numpy()
        arr_mask = self.mask
        arr_concat = np.stack((arr, arr_foldlabel, arr_distbottom, arr_extremities, arr_mask))

        return torch.from_numpy(arr_concat)

    

class FlipTensor(object):
    """
    Flip one axis randomly with probability p.
    ignore_axis : None or int
    """

    def __init__(self, ignore_axis=None, p=0.):
        self.ignore_axis = ignore_axis
        self.p = p
    
    def __call__(self, tensor):
        arr = tensor.numpy()
        flipped_arr = arr.copy()
        np.random.seed()
        r = np.random.uniform()
        if r >= self.p:
            return torch.from_numpy(flipped_arr)
        else:
            axes = [0,1,2]
            if self.ignore_axis is not None:
                axes.remove(self.ignore_axis)
            ax = np.random.choice(axes)
            slc = [slice(None) for _ in range(4)]
            slc[ax] = slice(None, None, -1)
            flipped_arr = flipped_arr[tuple(slc)]

            arr_flipped = flipped_arr.copy()
            return torch.from_numpy(arr_flipped)

class FlipFirstAxisTensor(object):
    """
    Flip first axis.
    """
    def __init__(self):
        pass

    def __call__(self, tensor):
        arr = tensor.numpy()
        flipped_arr = arr.copy()
        slc = [slice(None) for _ in range(4)]
        slc[0] = slice(None, None, -1) # first axis
        flipped_arr = flipped_arr[tuple(slc)]
        arr_flipped = flipped_arr.copy()
        return torch.from_numpy(arr_flipped)

def count_non_null(arr):
    return (arr != 0).sum()


def remove_branch(arr_foldlabel, arr_skel, selected_branch):
    """It masks the selected branch in arr_skel
    """
    mask = ((arr_foldlabel != 0) & (arr_foldlabel != selected_branch))
    #mask = arr_foldlabel != selected_branch
    mask = mask.astype(int)
    return arr_skel * mask


def intersection_skeleton_foldlabel(arr_foldlabel, arr_skel):
    """It returns the intersection between skeleton and foldlabel
    """
    mask = ((arr_foldlabel != 0)).astype(int)
    intersec = arr_skel*mask
    count_intersec = count_non_null(intersec)
    count_skel = count_non_null(arr_skel)
    if count_intersec != count_skel:
        raise ValueError("Probably misaligned skeleton and foldlabel\n"
                         f"Intersection between skeleton and foldlabel "
                         f"has {count_intersec} non-null elements.\n"
                         f"Skeleton has {count_skel} non-null elements")
    return count_intersec


def remove_bottom_branches(a, arr_skel):
    """Removes bottom branches from foldlabel.

    Bottom branches are numerated between 7000 and 7999"""
    return a*((a < 7000) | (a >= 8000)).astype(int)
    #return a*(arr_skel!=30)

def remove_top_branches(a):
    """Removes top branches from foldlabel.

    Top branches are numerated between 6000 and 6999"""
    return a*((a < 6000) | (a >= 7000)).astype(int)


def remove_branches_up_to_percent(arr_foldlabel, arr_skel,
                                  percentage, keep_extremity):
    """Removes from arr_skel random branches up to percentage of pixels
    If percentage==0, no pixel is deleted
    If percentage==100, all pixels are deleted
    """
    # We make some initial checks
    # of alignments between skeletons and foldlabels
    # TODO: remove voxels from foldlabel, based on skel. If trimdepth applied earlier, skel lost voxels, so they need to be removed from foldlabel INUTILE SI STEP 2 ?
    # TODO 2: remove_bottom_branches (+ top) : à enlerver ! Comme ça on en enlève toute la branche, puis à la fin on superpose le masque des bottoms, attention à pas additionner 30+30 ? Mais que 30+0. Prendre le max des deux arrays ?
    #NB: il faut bien enlever les bottoms à foldlabel, pour éviter que la branche puisse être sélectionner en dessous ! anciens bottoms + les nouveaux ! (0 et 30)
    # pas besoin d'ajouter les bottoms à la fin car ils n'auront pas pu être enlevé.
    total_pixels_after = intersection_skeleton_foldlabel(arr_foldlabel,
                                                         arr_skel)

    if keep_extremity=='bottom':
        arr_foldlabel = remove_bottom_branches(arr_foldlabel, arr_skel) # TODO: mask la valeur 30 de arr_skel (les 0 sont déjà masqués dans la ft principale)
    elif keep_extremity=='top':
        arr_foldlabel = remove_top_branches(arr_foldlabel)
    # if keep_bottom:
    #     arr_foldlabel_without_bottom = remove_bottom_branches(arr_foldlabel)
    #     branches, counts = np.unique(arr_foldlabel_without_bottom,
    #                                  return_counts=True)
    # else:
    #     branches, counts = np.unique(arr_foldlabel,
    #                                  return_counts=True)

    branches, counts = np.unique(arr_foldlabel,
                                 return_counts=True)

    total_pixels = count_non_null(arr_skel)
    # We take as index branches indexes that are not 0
    log.debug(f"Number of branches = {branches.size}")
    log.debug(f"Histogram of size of branches = {counts}")
    indexes = np.arange(branches.size-1) + 1

    log.debug(f"total_pixels = {total_pixels}")
    log.debug(f"skel shape = {arr_skel.shape}")
    log.debug(f"foldlabel shape = {arr_foldlabel.shape}")

    # We take random branches
    np.random.seed()
    np.random.shuffle(indexes)
    arr_skel_without_branches = arr_skel

    # We loop over shuffled indexes until enough branches have been removed
    for index in indexes:
        if total_pixels_after <= total_pixels*(100.-percentage)/100.:
            # TODO: éviter cas pathologique où l'on supprime un long paracingulaire car c'est la première branche de la liste ? Peut arriver quelque soit la proportion imposée.
            # Limiter la proportion avant d'enlever la branche !! explique pk auc diminue dans le temps sur PCS ?
            # est-ce souhaitable de supprimer les gros sillons ?
            # vu qu'on garde les bottoms c'est peut-être pas grave, mais ça explique peut-être pk la proportion importe peu.
            # choisir entre pourcentage des branches et pourcentage des voxels.
            # FOR NOW, KEEP LIKE THIS
            break
        arr_skel_without_branches = \
            remove_branch(arr_foldlabel,
                          arr_skel_without_branches,
                          branches[index])
        total_pixels_after = (arr_skel_without_branches != 0).sum()
        log.debug(f"total_pixels_after (iteration) = {total_pixels_after}")
    log.debug(f"total_pixels_after (final) = {total_pixels_after}")
    percent_pixels_removed = (total_pixels-total_pixels_after)/total_pixels*100
    log.debug(f"Minimum expected % removed pixels = {percentage}")
    log.debug(f"% removed pixels = {percent_pixels_removed}")

    assert (total_pixels == 0 or percent_pixels_removed >= percentage), \
        f"{percent_pixels_removed} >= {percentage}, total_pixels : {total_pixels}"

    return arr_skel_without_branches


class RemoveRandomBranchTensor(object):
    """Removes randomly branches up to percent
    """

    def __init__(self, sample_foldlabel,
                 percentage, input_size,
                 keep_extremity, variable_percentage):
        self.sample_foldlabel = sample_foldlabel
        self.percentage = percentage
        self.variable_percentage = variable_percentage
        self.input_size = input_size
        if keep_extremity=='random':
            np.random.seed()
            r = np.random.randint(3)
            if r == 0:
                self.keep_extremity='top'
            elif r==1:
                self.keep_extremity='bottom'
            else:
                self.keep_extremity=None
        else:
            self.keep_extremity = keep_extremity

    def __call__(self, tensor_skel):
        log.debug(f"Shape of tensor_skel = {tensor_skel.shape}")
        arr_skel = tensor_skel.numpy()
        arr_foldlabel = self.sample_foldlabel.numpy()

        # log.debug(f"arr_skel.shape = {arr_skel.shape}")
        # log.debug(f"arr_foldlabel.shape = {arr_foldlabel.shape}")
        assert (arr_skel.shape == arr_foldlabel.shape)
        assert (self.percentage >= 0)
        #assert not (self.keep_bottom and self.keep_top), "Choose either keep_bottom or keep_top."

        if self.variable_percentage:
            percentage = np.random.uniform(0, self.percentage)
        else:
            percentage = self.percentage
        log.debug("expected percentage "
                  f"(RemoveRandomBranchTensor) = {percentage}")

        arr_skel_without_branches = np.zeros(arr_skel.shape)
        log.debug("Shape of arr_skel before calling transform: "
                  f"{arr_skel_without_branches.shape}")

        # mask foldlabel with skeleton, as other augmentations might have shrunk skeleton
        arr_foldlabel = mask_array_with_skeleton(arr_foldlabel, arr_skel, cval=0)

        # Checks if it is only one image or a batch of images
        if len(arr_skel.shape) == len(self.input_size)+1:
            for num_img in np.arange(arr_skel.shape[0]):
                arr_skel_without_branches[num_img, ...] = \
                    remove_branches_up_to_percent(arr_foldlabel[num_img, ...],
                                                  arr_skel[num_img, ...],
                                                  percentage,
                                                  self.keep_extremity)
        elif len(arr_skel.shape) == len(self.input_size):
            arr_skel_without_branches = \
                remove_branches_up_to_percent(arr_foldlabel,
                                              arr_skel,
                                              percentage,
                                              self.keep_extremity)
        else:
            raise RuntimeError(
                f"Unexpected skeleton shape."
                f"Compare arr_skel shape {arr_skel.shape} "
                f"with input_size shape {self.input_size.shape}")

        arr_skel_without_branches = arr_skel_without_branches.astype('float32')

        return torch.from_numpy(arr_skel_without_branches)


class RotateTensor(object):
    """Apply a random rotation on the images
    """

    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, tensor):
        arr = tensor.numpy()
        log.debug("Shapes before rotation", tensor.shape, arr.shape)
        rot_array = np.copy(arr)

        for axes in (1, 2), (1, 3), (2, 3):
            np.random.seed()
            angle = np.random.uniform(-self.max_angle, self.max_angle)
            log.debug(axes, angle)
            rot_array = rotate(rot_array,
                               angle=angle,
                               axes=axes,
                               order=0,
                               reshape=False,
                               mode='constant',
                               cval=0)

        #rot_array = np.expand_dims(rot_array[..., 0], axis=0)

        log.debug("Values in the array after rotation", np.unique(rot_array))

        return torch.from_numpy(rot_array)


class PartialCutOutTensor_Roll(object):
    """Apply a rolling cutout on the images and puts only bottom value
    inside the cutout
    cf. Improved Regularization of Convolutional Neural Networks with Cutout,
    arXiv, 2017
    We assume that the rectangle to be cut is inside the image.
    """

    def __init__(self, mask_constraint=False, from_skeleton=True, input_size=None,
                 keep_extremity='bottom', keep_proba_per_branch=1., keep_proba_global=1., patch_size=None,
                 random_size=False, localization=None):
        """[summary]
        If from_skeleton==True,
            takes skeleton image, cuts it out and fills with bottom_only image
        If from_skeleton==False,
            takes bottom_only image, cuts it out and fills with skeleton image
        Args:
            mask (bool array): the mask of the ROI limits where
                the center of the cutout can be.
            mask_constraint (bool): whether mask is used or not.
            from_skeleton (bool, optional): Defaults to True.
            patch_size (either int or list of int): Defaults to None.
                if int, percentage of the volume to cutout.
                Defines a rectangle with same proportions as input.
            random_size (bool, optional): Defaults to False.
            inplace (bool, optional): Defaults to False.
            localization ([type], optional): Defaults to None.
        """

        if isinstance(patch_size, int):
            self.patch_size = patch_size
        elif len(patch_size)==2: # a range is given, select a random size in the range
            self.patch_size = np.random.randint(low=patch_size[0], high=patch_size[1])
        else: # a crop size is given
            self.patch_size = rotate_list(patch_size)
        self.input_size = input_size
        self.random_size = random_size
        self.localization = localization
        self.from_skeleton = from_skeleton
        self.mask_constraint = mask_constraint
        self.keep_proba_per_branch = keep_proba_per_branch
        self.keep_proba_global = keep_proba_global
        if keep_extremity=='random':
            np.random.seed()
            r = np.random.randint(3)
            if r == 0:
                self.keep_extremity='top'
            elif r==1:
                self.keep_extremity='bottom'
            else:
                self.keep_extremity=None
        else:
            np.random.seed()
            r = np.random.uniform()
            # don't keep bottom/top with given probability
            if r > self.keep_proba_global:
                keep_extremity=None
            self.keep_extremity = keep_extremity

    def __call__(self, tensor):

        arr_all = tensor.numpy()
        arr = arr_all[0]
        arr_foldlabel = arr_all[1]
        arr_foldlabel = mask_array_with_skeleton(arr_foldlabel, arr, cval=0)

        img_shape = np.array(arr.shape)
        if isinstance(self.patch_size, int):
            proportion = (1/100*self.patch_size)**(1/(len(img_shape)-1))
            size = rotate_list(self.input_size)
            size = proportion*np.array(size)
            size = np.round(size).astype(int)
            size[-1]=1
            
        else:
            size = np.copy(self.patch_size)
        assert len(size) == len(img_shape), f"Incorrect patch dimension : {size}"
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.random_size:
                size[ndim] = np.random.randint(0, size[ndim])
        if self.localization is not None:
            start_cutout = []
            for ndim in range(len(img_shape)):
                delta_before = max(
                    self.localization[ndim] - size[ndim] // 2, 0)
                start_cutout.append(delta_before)
        else:
            np.random.seed()
            if self.mask_constraint:
                #boolean = True
                # loop until the center of the crop is inside the mask
                #while boolean or not self.mask[tuple(middle_cutout)]:
                #    boolean = False
                #    start_cutout = []
                #    middle_cutout = []
                #    for ndim in range(len(img_shape)):
                #        delta_before = np.random.randint(0, img_shape[ndim])
                #        start_cutout.append(delta_before)
                        # define middle of cutout, taking roll into account
                #        middle_pos = int((delta_before + size[ndim] // 2)%img_shape[ndim])
                #        middle_cutout.append(middle_pos)
                # alt : use mask as proba sampling # TODO : implement properly and distinguish cutin and cutout
                # normalize the mask
                mask = arr_all[4]
                mask = mask / np.sum(mask)
                i = np.random.choice(np.arange(mask.size), p=mask.ravel())
                middle_pos = np.unravel_index(i, mask.shape)
                start_cutout = [(middle_pos[ndim] - size[ndim] // 2)%img_shape[ndim] for ndim in range(len(img_shape))]
            else:
                start_cutout = []
                for ndim in range(len(img_shape)):
                    delta_before = np.random.randint(0, img_shape[ndim])
                    start_cutout.append(delta_before)

        # Creates rolling mask cutout
        mask_roll = np.zeros(img_shape).astype('float32')

        indexes = []
        for ndim in range(len(img_shape)):
            indexes.append(slice(0, int(size[ndim])))
        mask_roll[tuple(indexes)] = 1

        for ndim in range(len(img_shape)):
            mask_roll = np.roll(mask_roll, start_cutout[ndim], axis=ndim)

        # Determines part of the array inside and outside the cutout
        arr_inside = arr * mask_roll
        arr_outside = arr * (1 - mask_roll)

        if self.keep_proba_per_branch < 1.:
            # keep bottom with proba p for each branch
            indexed_branches = np.mod(arr_foldlabel,
                            np.full(arr_foldlabel.shape, fill_value=1000))
            indexes =  np.unique(indexed_branches)
            assert (len(indexes)>1), 'No branch in foldlabel'
            indexes = indexes[1:] # remove background
            select = np.random.rand(indexes.size) < self.keep_proba_per_branch
            selected_indexes = indexes[select]
            selected_branches = np.isin(indexed_branches, selected_indexes)
            #print(np.sum(selected_branches!=0), np.sum(arr_foldlabel!=0), np.sum(arr!=0), np.sum(np.logical_and(arr!=0, selected_branches!=0)) / np.sum(arr!=0))

        # If self.from_skeleton == True:
        # This keeps the whole skeleton outside the cutout
        # and keeps only bottom value inside the cutout
        if self.from_skeleton:
            if self.keep_extremity=='top':
                arr_inside = arr_inside * (arr_inside == 35)
            elif self.keep_extremity=='bottom':
                arr_inside = arr_inside * (arr_inside == 30)
            elif self.keep_extremity=='bottom_top':
                arr_inside = arr_inside * (np.logical_or(arr_inside == 30, arr_inside == 35))
            elif self.keep_extremity=='all': # protect whole branch !
                arr_inside = arr_inside != 0
            else:
                arr_inside = arr_inside * (arr_inside == 0)
            if self.keep_proba_per_branch < 1.:
                arr_inside = arr_inside * selected_branches

        # If self.from_skeleton == False:
        # This keeps only bottom value outside the cutout
        # and keeps the whole skeleton inside the cutout
        else:
            if self.keep_extremity=='top':
                arr_outside = arr_outside * (arr_outside == 35)
            elif self.keep_extremity=='bottom':
                arr_outside = arr_outside * (arr_outside == 30)
            elif self.keep_extremity=='bottom_top':
                arr_outside = arr_outside * (np.logical_or(arr_outside == 30, arr_outside == 35))
            elif self.keep_extremity=='all': # protect whole branch !
                arr_outside = arr_outside != 0
            else:
                arr_outside = arr_outside * (arr_outside == 0)
            if self.keep_proba_per_branch < 1.:
                arr_outside = arr_outside * selected_branches
        
        trimmed_arr = arr_inside + arr_outside

        #log.info(f"{self.from_skeleton},{np.sum(arr!=0)},{np.sum(trimmed_arr!=0)},{np.sum(np.logical_and(arr!=0, arr!=30))},{np.sum(np.logical_and(trimmed_arr!=0,trimmed_arr!=30))}")
        #np.save('/volatile2/jl277509/visu_augmentations/sub_new/skel_trimdepth_extremities_cutout.npy', trimmed_arr)

        trimmed_arr = np.expand_dims(trimmed_arr, axis=0)
        arr_all = np.vstack((trimmed_arr, arr_all[1:]))

        arr_all = arr_all.astype('float32')

        return torch.from_numpy(arr_all)


class CheckerboardTensor(object):
    """Apply a checkerboard noise
    """

    def __init__(self, checkerboard_size):
        """[summary]
        Args:
        """
        self.checkerboard_size = checkerboard_size

    def __call__(self, tensor):

        arr = tensor.numpy()
        img_shape = np.array(arr.shape)

        if isinstance(self.checkerboard_size, int):
            size = [self.checkerboard_size for _ in range(len(img_shape))]
        else:
            size = np.copy(self.checkerboard_size)
        assert len(size) == len(img_shape), "Incorrect patch dimension."

        start_cutout = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            np.random.seed()
            delta_before = np.random.randint(0, size[ndim])
            start_cutout.append(delta_before)

        # Creates checkerboard mask
        mask = checkerboard(
            img_shape,
            self.checkerboard_size).astype('float32')

        for ndim in range(len(img_shape)):
            mask = np.roll(mask, start_cutout[ndim], axis=ndim)

        return torch.from_numpy(arr * mask)


# OBSOLETE, INCOMPATIBILITIES
class PartialCutOutTensor(object):
    """Apply a cutout on the images and puts only bottom value inside
    cf. Improved Regularization of Convolutional Neural Networks with Cutout,
    arXiv, 2017
    We assume that the rectangle to be cut is inside the image.
    """

    def __init__(self, from_skeleton=True, patch_size=None, random_size=False,
                 inplace=False, localization=None):
        """[summary]
        If from_skeleton==True,
            takes skeleton image, cuts it out and fills with bottom_only image
        If from_skeleton==False,
            takes bottom_only image, cuts it out and fills with skeleton image
        Args:
            from_skeleton (bool, optional): Defaults to True.
            patch_size (either int or list of int): Defaults to None.
            random_size (bool, optional): Defaults to False.
            inplace (bool, optional): Defaults to False.
            localization ([type], optional): Defaults to None.
        """
        self.patch_size = rotate_list(patch_size)
        self.random_size = random_size
        self.inplace = inplace
        self.localization = localization
        self.from_skeleton = from_skeleton

    def __call__(self, tensor):

        arr = tensor.numpy()
        img_shape = np.array(arr.shape)
        if isinstance(self.patch_size, int):
            size = [self.patch_size for _ in range(len(img_shape))]
        else:
            size = np.copy(self.patch_size)
        assert len(size) == len(img_shape), "Incorrect patch dimension."
        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.random_size:
                size[ndim] = np.random.randint(0, size[ndim])
            if self.localization is not None:
                delta_before = max(
                    self.localization[ndim] - size[ndim] // 2, 0)
            else:
                np.random.seed()
                delta_before = np.random.randint(
                    0, img_shape[ndim] - size[ndim] + 1)
            indexes.append(slice(int(delta_before),
                                 int(delta_before + size[ndim])))
        if self.from_skeleton:
            if self.inplace:
                arr_cut = arr[tuple(indexes)]
                arr[tuple(indexes)] = arr_cut * (arr_cut == 30)
                return torch.from_numpy(arr)
            else:
                arr_copy = np.copy(arr)
                arr_cut = arr_copy[tuple(indexes)]
                arr_copy[tuple(indexes)] = arr_cut * (arr_cut == 30)
                return torch.from_numpy(arr_copy)
        else:
            arr_bottom = arr * (arr == 30)
            arr_cut = arr[tuple(indexes)]
            arr_bottom[tuple(indexes)] = np.copy(arr_cut)
            return torch.from_numpy(arr_bottom)


# OBSOLETE, INCOMPATIBILITIES
class CutoutTensor(object):
    """Apply a cutout on the images
    cf. Improved Regularization of Convolutional Neural Networks with Cutout,
    arXiv, 2017
    We assume that the cube to be cut is inside the image.
    """

    def __init__(self, patch_size=None, value=0, random_size=False,
                 inplace=False, localization=None):
        self.patch_size = patch_size
        self.value = value
        self.random_size = random_size
        self.inplace = inplace
        self.localization = localization

    def __call__(self, arr):

        img_shape = np.array(arr.shape)
        if isinstance(self.patch_size, int):
            size = [self.patch_size for _ in range(len(img_shape))]
        else:
            size = np.copy(self.patch_size)
        assert len(size) == len(img_shape), "Incorrect patch dimension."
        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.random_size:
                size[ndim] = np.random.randint(0, size[ndim])
            if self.localization is not None:
                delta_before = max(
                    self.localization[ndim] - size[ndim] // 2, 0)
            else:
                delta_before = np.random.randint(
                    0, img_shape[ndim] - size[ndim] + 1)
            indexes.append(slice(int(delta_before),
                                 int(delta_before + size[ndim])))

        if self.inplace:
            arr[tuple(indexes)] = self.value
            return torch.from_numpy(arr)
        else:
            arr_cut = np.copy(arr)
            arr_cut[tuple(indexes)] = self.value
            return torch.from_numpy(arr_cut)


class ToPointnetTensor(object):

    def __init__(self, padding_method=repeat_padding, n_max=None):
        self.padding_method = padding_method
        self.n_max = n_max

    def __call__(self, tensor):
        arr = tensor.numpy()

        clouds = []
        for i in range(arr.shape[0]):  # loop over batch elements
            point_cloud = np.array(arr[i].nonzero()[:3])
            clouds.append(point_cloud)

        padded_clouds = pad(clouds, padding_method=self.padding_method,
                            n_max=self.n_max)

        return torch.from_numpy(padded_clouds)


def interval(obj, lower=None):
    """ Listify an object.
    Parameters
o   ----------
    obj: 2-uplet or number
        the object used to build the interval.
    lower: number, default None
        the lower bound of the interval. If not specified, a symetric
        interval is generated.
    Returns
    -------
    interval: 2-uplet
        an interval.
    """
    if isinstance(obj, numbers.Number):
        if obj < 0:
            raise ValueError("Specified interval value must be positive.")
        if lower is None:
            lower = -obj
        return (lower, obj)
    if len(obj) != 2:
        raise ValueError("Interval must be specified with 2 values.")
    min_val, max_val = obj
    if min_val > max_val:
        raise ValueError("Wrong interval boudaries.")
    return tuple(obj)


class Transformer(object):
    """ Class that can be used to register a sequence of transformations.
    """
    Transform = namedtuple("Transform", ["transform", "probability"])

    def __init__(self):
        """ Initialize the class.
        """
        self.transforms = []

    def register(self, transform, probability=1):
        """ Register a new transformation.
        Parameters
        ----------
        transform: callable
            the transformation object.
        probability: float, default 1
            the transform is applied with the specified probability.
        """
        trf = self.Transform(transform=transform, probability=probability, )
        self.transforms.append(trf)

    def __call__(self, arr):
        """ Apply the registered transformations.
        """
        transformed = arr.copy()
        for trf in self.transforms:
            if np.random.rand() < trf.probability:
                transformed = trf.transform(transformed)
        return transformed

    def __str__(self):
        if len(self.transforms) == 0:
            return '(Empty Transformer)'
        s = 'Composition of:'
        for trf in self.transforms:
            s += '\n\t- ' + trf.__str__()
        return s


class ResizeTensor(object):
    """Apply resize to a 3D image
    """

    def __init__(self, resize_ratio):
        self.resize_ratio = resize_ratio

    def __call__(self, tensor):
        arr = tensor.numpy()
        resized_arr = np.copy(arr)
        log.debug(f"Resize Ratio: {self.resize_ratio}")
        resized_arr = zoom(resized_arr,
                           self.resize_ratio,
                           order=0)

        return torch.from_numpy(resized_arr)
    

class CropResizeTensor(object):
    """Apply crop then resize to a 3D image.
    the crop ratio is given by a range and is chosen uniformly.
    """

    def __init__(self, input_size, crop_ratio):
        self.input_size = input_size
        if isinstance(crop_ratio, int):
            self.crop_ratio = crop_ratio
        elif len(crop_ratio)==2: # a range is given, select a random size in the range
            self.crop_ratio = np.random.randint(low=crop_ratio[0], high=crop_ratio[1])

    def __call__(self, tensor):

        arr = tensor.numpy()
        img_shape = np.array(arr.shape)

        # define crop size
        proportion = (1/100*self.crop_ratio)**(1/(len(img_shape)-1))
        size = rotate_list(self.input_size)
        size = proportion*np.array(size)
        size = np.round(size).astype(int)
        size[-1]=1

        # define cropped_arr
        start_cutout = []
        for ndim in range(len(img_shape)):
            max_offset = img_shape[ndim] - size[ndim]
            if max_offset <= 0:
                delta_before = 0
            else:
                np.random.seed()
                delta_before = np.random.randint(0, img_shape[ndim]-size[ndim])
            start_cutout.append(delta_before)
        cropped_arr = arr.copy()
        cropped_arr = cropped_arr[tuple([slice(int(delta_before),
                                                int(delta_before + s)) for delta_before, s in zip(start_cutout, size)])]

        # define resize_ratio
        resize_ratio = np.array(img_shape) / np.array(size)
        resized_arr = zoom(cropped_arr,
                           resize_ratio,
                           order=0)

        return torch.from_numpy(resized_arr)


class GaussianNoiseTensor(object):
    """Add gaussian noise to a 3D image."""

    def __init__(self, sigma):
        self.sigma = sigma
    
    def __call__(self, tensor):
        noise = torch.randn(tensor.shape)
        return tensor + self.sigma * noise
    
class TransposeTensor(object):
    """
    Permute first and last dimension.
    """

    def __init__(self):
        None
    def __call__(self, tensor):
        arr = tensor.numpy()
        arr_t = np.transpose(arr, (3, 0, 1, 2))
        arr_t = arr_t.astype('float32')

        return(torch.from_numpy(arr_t))
    

class TranslateTensor(object):
    """
    Apply a random slicing of up to n_voxel in every direction and pads
    to perform translation while keeping original dimension.
    """

    def __init__(self, n_voxel):
        self.n_voxel = n_voxel
    
    def __call__(self, tensor):
        arr = tensor.numpy()
        translated_arr = arr.copy()
        if isinstance(self.n_voxel, int):
            absolute_translation_xyz = np.random.randint(self.n_voxel+1, size=3)
        else:
            absolute_translation_xyz = np.array([np.random.randint(n_vx_dim+1) for n_vx_dim in self.n_voxel])
        sign_translation = np.random.randint(2, size=3)
        slc = [slice(None) if (translation==0) else slice(translation, None) if sign else slice(-translation)
               for sign, translation in zip(sign_translation, absolute_translation_xyz)]
        translated_arr = translated_arr[tuple(slc)]
        pad_width = [(0, translation) if sign else (translation, 0) for sign, translation in zip(sign_translation, absolute_translation_xyz)] + [(0,0)]
        translated_arr = np.pad(translated_arr, pad_width, mode='constant', constant_values=0)
        #translated_arr = np.expand_dims(translated_arr[..., 0], axis=0)

        return torch.from_numpy(translated_arr)
    
"""
class RotateTensor(object):
    ""
    Apply rotation using Kornia.
    ""

    def __init__(self, degrees):
        self.degrees = tuple(degrees)

    def __call__(self, tensor):
        aug = RandomRotation3D(self.degrees, p=1., resample='nearest', keepdim=True)
        rotated_tensor = aug(tensor)

        return rotated_tensor
"""

class NoisyEdgesTensor(object):

    """
    Add pepper noise to the voxels at the edges of the crop.
    The probability increases towards the edges.
    """

    def __init__(self, slope, offset):
        self.slope = slope
        self.offset = offset

    def __call__(self, tensor):
        arr = tensor.numpy()
        img_shape = arr.shape
        masked_arr = arr.copy()

        k_max = int((1-self.offset) / self.slope)
        mask = np.ones(arr.shape)
        for k in range(k_max):
            outer_layer_k = np.array([[[(i<=k) or (i>=img_shape[2]-1-k) or (j<=k) or (j>=img_shape[1]-1-k) or (l<=k) or (l>=img_shape[0]-1-k)
                                        for i in range(img_shape[2])] for j in range(img_shape[1])] for l in range(img_shape[0])])
            outer_layer_k = np.expand_dims(outer_layer_k, axis=-1)
            outer_layer_k = outer_layer_k.astype('float64')
            mask -= self.slope*outer_layer_k

        # generate random noise
        noise = np.random.uniform(size=arr.shape)
        # threshold based on probability mask
        binary_noise = noise < mask
        masked_arr[np.where(binary_noise==0)]=0

        return torch.from_numpy(masked_arr)
    

class CropFixedSizeTensor(object):

    """
    Given a fixed nb of vx to remove on each axis,
    defines a random fixed size crop.
    """

    def __init__(self, n_vx_to_remove):
        self.n_vx_to_remove = n_vx_to_remove

    def __call__(self, tensor):
        arr = tensor.numpy()
        cropped_arr = arr.copy()
        img_size = arr.shape[:3]
        indexes = []
        for img_dim, to_remove in zip(img_size, self.n_vx_to_remove):
            delta_before = np.random.randint(0, to_remove+1)
            indexes.append(slice(int(delta_before),
                                 int(delta_before + img_dim - to_remove)))
        indexes.append(slice(1))
        cropped_arr = cropped_arr[tuple(indexes)]
        cropped_arr = np.expand_dims(cropped_arr[..., 0], axis=0)

        return torch.from_numpy(cropped_arr)
    

class TrimDepthTensor(object):
    """
    Trim depth based on distbottom.
    Set max_distance to 0 to remove bottom only.
    Set max_distance to -1 to remove nothing.
    Then the scale is 100 = 2mm.
    """

    def __init__(self, max_distance, delta,
                 input_size, keep_extremity, uniform, binary, binary_proba=0.5,
                 pepper=0.5, redefine_bottom=False):
        self.max_distance = max_distance
        self.delta = delta
        self.input_size = input_size
        self.uniform=uniform
        self.binary=binary
        self.binary_proba=binary_proba
        self.pepper=pepper
        self.redefine_bottom=redefine_bottom
        if keep_extremity=='random':
            np.random.seed()
            r = np.random.randint(2)
            if r == 0:
                self.keep_extremity='top'
            else:
                self.keep_extremity=None
        else:
            self.keep_extremity = keep_extremity
    
    def __call__(self, tensor_skel):
        log.debug(f"Shape of tensor_skel = {tensor_skel.shape}")
        arr_all = tensor_skel.numpy()
        arr_skel = arr_all[0]
        #np.save('/volatile2/jl277509/visu_augmentations/sub_new/full_skel.npy', arr_skel)
        arr_distbottom = arr_all[2]
        arr_foldlabel = arr_all[1]

        assert (self.max_distance >= -1)

        # mask foldlabel and distbottom with skeleton
        # in case another augmentation was applied before
        arr_foldlabel = mask_array_with_skeleton(arr_foldlabel, arr_skel, cval=0)
        arr_distbottom = mask_array_with_skeleton(arr_distbottom, arr_skel, cval=32501)

        if self.uniform: # OBSOLETE, need to rewrite bottom values
            arr_trimmed = arr_skel.copy()
            # get random threshold
            threshold = np.random.randint(-1, self.max_distance+1)
            # mask skel with thresholded distbottom
            if self.keep_extremity=='top':
                arr_trimmed[np.logical_and(arr_distbottom<=threshold, arr_skel!=35)]=0
            else:
                arr_trimmed[arr_distbottom<=threshold]=0
        else:
            # select a threshold for each branch
            arr_trimmed_branches = np.zeros(arr_skel.shape)
            indexed_branches = np.mod(arr_foldlabel,
                                      np.full(arr_foldlabel.shape, fill_value=1000))
            indexes =  np.unique(indexed_branches)
            assert (len(indexes)>1), 'No branch in foldlabel'
            for index in indexes[1:]:
                arr_trimmed = arr_skel.copy()
                mask_branch = indexed_branches==index
                if self.binary:
                    r = np.random.uniform()
                    if r > self.binary_proba:
                        threshold = -1
                    else:
                        threshold = self.max_distance
                else: # OBSOLETE, is scaling by 100 correct ?
                    #threshold = np.random.randint(-1, (self.max_distance+1)//100)*100
                    threshold = np.random.choice(np.array([0, 71, 100]))
                if self.keep_extremity=='top':
                    arr_trimmed[np.logical_and(arr_distbottom<=threshold, arr_skel!=35)]=0
                else:
                    arr_trimmed[arr_distbottom<=threshold]=0
                arr_trimmed_branch = arr_trimmed * mask_branch
                if threshold != -1: # if so no need to redefine the bottoms.
                    # define distbottom pour la branche !
                    arr_distbottom_branch = arr_distbottom * (arr_trimmed_branch != 0)
                    # get smallest distbottom value + delta margin and replace topological value
                    if self.redefine_bottom:
                        arr_trimmed_branch[(arr_distbottom_branch <= threshold + self.delta)&(arr_distbottom_branch > threshold)]=30
                arr_trimmed_branches += arr_trimmed_branch
            arr_trimmed = arr_trimmed_branches.copy()

        trimmed_vx = np.logical_xor(arr_trimmed!=0, arr_skel!=0)
        pepper = (np.random.rand(*trimmed_vx.shape) > self.pepper).astype('float64')
        # add topological values to pepper before adding to trimmed skel
        pepper = np.multiply(pepper, arr_skel)
        arr_trimmed += np.multiply(trimmed_vx, pepper)
        
        #arr_trimmed = arr_trimmed.astype('float32')
        #np.save('/volatile2/jl277509/visu_augmentations/sub_new/skel_trimdepth.npy', arr_trimmed)
        arr_trimmed = np.expand_dims(arr_trimmed, axis=0)
        arr_all = np.vstack((arr_trimmed, arr_all[1:]))

        arr_all = arr_all.astype('float32')

        return torch.from_numpy(arr_all)
    
"""
class ElasticDeformTensor(object):

    def __init__(self, sigma, points):
        self.sigma=sigma
        self.points=points
    
    def __call__(self, tensor):
        arr = tensor.numpy()
        arr = arr.reshape(arr.shape[:-1])
        deformed_arr = elasticdeform.deform_random_grid(arr, sigma=self.sigma, points=self.points, order=0)
        shape = list(arr.shape) + [1]
        deformed_arr = deformed_arr.reshape(shape)

        deformed_arr = deformed_arr.astype('float32')

        return torch.from_numpy(deformed_arr)
"""

class TrimExtremitiesTensor(object):
    """
    Trim the lateral edges of the folds based on sample_extremities.
    Parameters
    ----------
    p: probability to trim each branch (i.e. proportion of trimmed branches)
    protective structure: object such as morphology.ball(n). The object
    shape must be odd (so it has an int center).
    arr_extremities : binary mask of the trimmed skeleton voxels.
    """

    def __init__(self, sample_extremities, sample_foldlabel,
                 input_size, protective_structure, p=0.5, keep_bottom=False):
        self.input_size = input_size
        self.protective_structure = protective_structure
        self.p = p
        self.sample_foldlabel = sample_foldlabel
        self.sample_extremities = sample_extremities
        self.keep_bottom = keep_bottom
    
    def __call__(self, tensor_skel):
        log.debug(f"Shape of tensor_skel = {tensor_skel.shape}")
        arr_skel = tensor_skel.numpy()
        arr_foldlabel = self.sample_foldlabel.numpy()
        arr_extremities = self.sample_extremities.numpy()

        assert (self.p >= 0)

        arr_foldlabel = mask_array_with_skeleton(arr_foldlabel, arr_skel, cval=0)
        arr_extremities = mask_array_with_skeleton(arr_extremities, arr_skel, cval=0)

        arr_trimmed_branches = np.zeros(arr_skel.shape)
        indexed_branches = np.mod(arr_foldlabel,
                                np.full(arr_foldlabel.shape, fill_value=1000))
        indexes =  np.unique(indexed_branches)
        assert (len(indexes)>1), 'No branch in foldlabel'
        # loop over branches
        for index in indexes[1:]:
            mask_branch = indexed_branches==index
            branch = arr_skel * mask_branch
            r = np.random.uniform()
            if r < self.p:
                trimmed_branch = (1-arr_extremities) * branch
                if np.array_equal(branch!=0, trimmed_branch!=0): # nothing to trim
                    pass
                else:
                    # find mass center
                    coords = np.nonzero(branch)
                    center = [np.mean(coords[i]) for i in range(len(coords)-1)]
                    center = (np.round(center)).astype(int)
                    # branch center is protected using given structure
                    mask_protection = np.zeros(branch.shape)
                    slc = [slice(max(0, c-s//2),
                                 min(arr_skel.shape[i], c+s//2 +1))
                           for i,(c,s) in enumerate(zip(center, self.protective_structure.shape))]
                    slc.append(slice(1))
                    # need to slice the protective structure if it reaches an edge
                    # the slice depends on which edge is reached
                    # the mass center and the protective structure center must remain aligned
                    slc_struct = []
                    for i,(c,s) in enumerate(zip(center, self.protective_structure.shape)):
                        if c-s//2 < 0:
                            sl = slice(-(c-s//2), None)
                        elif c+s//2+1 > arr_skel.shape[i]:
                            sl = slice(None, -(c+s//2+1 - arr_skel.shape[i]))
                        else:
                            sl = slice(None)
                        slc_struct.append(sl)
                    slc_struct.append(slice(1))
                    mask_protection[tuple(slc)]=self.protective_structure[tuple(slc_struct)]
                    trimmed_branch = branch * np.logical_or(mask_protection, 1-arr_extremities)

                arr_trimmed_branches += trimmed_branch
            else:
                arr_trimmed_branches += branch
        arr_trimmed = arr_trimmed_branches.copy()

        if self.keep_bottom:
            # add the bottoms which were trimmed
            trimmed_vx = np.logical_xor(arr_skel, arr_trimmed)
            trimmed_bottoms = np.logical_and(arr_skel==30, trimmed_vx)
            arr_trimmed = arr_trimmed + 30*trimmed_bottoms
        
        arr_trimmed = arr_trimmed.astype('float32')

        return torch.from_numpy(arr_trimmed)
    

class HighlightExtremitiesTensor(object):
    """
    Highlight the lateral edges of the folds based on sample_extremities
    with a specific topological value.
    Parameters
    ----------
    p: probability to trim each branch (i.e. proportion of trimmed branches)
    pepper: proba to erase each voxel in trimmed branch
    protective structure: object such as morphology.ball(n). The object
    shape must be odd (so it has an int center).
    arr_extremities : binary mask of the trimmed skeleton voxels.
    """

    def __init__(self,
                 input_size, protective_structure, p=0.5, pepper=0.5, keep_extremity=None):
        self.input_size = input_size
        self.protective_structure = protective_structure
        self.p = p
        self.pepper=pepper
        self.keep_extremity=keep_extremity
    
    def __call__(self, tensor_skel):
        log.debug(f"Shape of tensor_skel = {tensor_skel.shape}")
        arr_all = tensor_skel.numpy()
        arr_skel = arr_all[0]
        arr_extremities = arr_all[3]
        arr_foldlabel = arr_all[1]

        assert (self.p >= 0)

        arr_foldlabel = mask_array_with_skeleton(arr_foldlabel, arr_skel, cval=0)
        arr_extremities = mask_array_with_skeleton(arr_extremities, arr_skel, cval=0)

        arr_trimmed_branches = np.zeros(arr_skel.shape)
        indexed_branches = np.mod(arr_foldlabel,
                                np.full(arr_foldlabel.shape, fill_value=1000))
        indexes =  np.unique(indexed_branches)
        assert (len(indexes)>1), 'No branch in foldlabel'
        # loop over branches
        # TODO: compute intersection between arr_extremities and foldlabel
        # here to loop only on selected idxs.
        # this would avoid computing arr_equal too many times
        # find unique indexes in np.multiply(indexed_branches, arr_extremities)
        for index in indexes[1:]:
            mask_branch = indexed_branches==index
            branch = arr_skel * mask_branch
            r = np.random.uniform()
            if r < self.p:
                trimmed_branch = (1-arr_extremities) * branch
                if self.protective_structure is not None:
                    if np.array_equal(branch!=0, trimmed_branch!=0): # nothing to trim
                        pass
                    else:
                        # find mass center
                        coords = np.nonzero(branch)
                        center = [np.mean(coords[i]) for i in range(len(coords)-1)]
                        center = (np.round(center)).astype(int)
                        # branch center is protected using given structure
                        mask_protection = np.zeros(branch.shape)
                        slc = [slice(max(0, c-s//2),
                                    min(arr_skel.shape[i], c+s//2 +1))
                            for i,(c,s) in enumerate(zip(center, self.protective_structure.shape))]
                        slc.append(slice(1))
                        # need to slice the protective structure if it reaches an edge
                        # the slice depends on which edge is reached
                        # the mass center and the protective structure center must remain aligned
                        slc_struct = []
                        for i,(c,s) in enumerate(zip(center, self.protective_structure.shape)):
                            if c-s//2 < 0:
                                sl = slice(-(c-s//2), None)
                            elif c+s//2+1 > arr_skel.shape[i]:
                                sl = slice(None, -(c+s//2+1 - arr_skel.shape[i]))
                            else:
                                sl = slice(None)
                            slc_struct.append(sl)
                        slc_struct.append(slice(1))
                        mask_protection[tuple(slc)]=self.protective_structure[tuple(slc_struct)]
                        trimmed_branch = branch * np.logical_or(mask_protection, 1-arr_extremities)

                arr_trimmed_branches += trimmed_branch
            else:
                arr_trimmed_branches += branch

        arr_trimmed = (arr_trimmed_branches != 0).astype('float64')
        trimmed_vx = (np.logical_xor(arr_skel, arr_trimmed)).astype('float64')

        #pepper = np.random.randint(0, 2, size=trimmed_vx.shape).astype('float64')
        pepper = (np.random.rand(*trimmed_vx.shape) > self.pepper).astype('float64')

        arr_trimmed += np.multiply(trimmed_vx, pepper)

        # add bottoms / top if keep
        if self.keep_extremity is None:
            extremity = np.zeros(arr_skel.shape)
        elif self.keep_extremity=='bottom':
            extremity = arr_skel==30
        elif self.keep_extremity=='top':
            extremity = arr_skel==35
        elif self.keep_extremity=='bottom_top':
            extremity = np.logical_or(arr_skel==30, arr_skel==35)
            
        
        arr_trimmed = np.logical_or(arr_trimmed, extremity)

        # multiply by the topological values
        arr_trimmed = np.multiply(arr_trimmed, arr_skel)
        
        #arr_trimmed = arr_trimmed.astype('float32')

        arr_trimmed = np.expand_dims(arr_trimmed, axis=0)

        arr_all = np.vstack((arr_trimmed, arr_all[1:]))

        arr_all = arr_all.astype('float32')

        #np.save('/volatile2/jl277509/visu_augmentations/sub_new/skel_trimdepth_extremities.npy', arr_trimmed)
        return torch.from_numpy(arr_all)


class TrimCropEdges(object):

    """
    Mask each edge of the crop independently.
    """

    def __init__(self, max_n_voxel, ignore_axis=None, value=0):
        self.max_n_voxel = max_n_voxel
        self.ignore_axis = ignore_axis
        self.value = value
    
    def __call__(self, tensor):

        arr = tensor.numpy()
        arr_trimmed = np.full(arr.shape, fill_value=self.value)

        # TODO: take min(max_n_voxel, 15% of dim) for each dim

        vx_to_trim_left = np.random.randint(self.max_n_voxel+1, size=3)
        vx_to_trim_right = np.random.randint(self.max_n_voxel+1, size=3)
        if self.ignore_axis is not None:
            slc = [slice(None) if axis==self.ignore_axis
                   else slice(vx_l, n-vx_r) for axis, (n, vx_l, vx_r) in enumerate(zip(arr.shape, vx_to_trim_left, vx_to_trim_right))]
        else:
            slc = [slice(vx_l, n-vx_r) for n, vx_l, vx_r in zip(arr.shape, vx_to_trim_left, vx_to_trim_right)]
        slc.append(slice(1))
        arr_trimmed[tuple(slc)]=arr[tuple(slc)]

        arr_trimmed = arr_trimmed.astype('float32')

        return torch.from_numpy(arr_trimmed)
    

class MultiCutoutTensor(object):
    """
    Cutout performed multiple times (meant to erase multiple small areas).
    Since the areas cut are small, center the crops on non-zero voxels.
    NB: the bottoms are not kept.
    """

    def __init__(self, patch_size, input_size, number_patches=1, value=0):
        self.patch_size = patch_size
        self.number_patches = number_patches
        self.input_size = input_size
        self.value = value
        
    def __call__(self, tensor):

        arr = tensor.numpy()
        img_shape = np.array(arr.shape)
        if isinstance(self.patch_size, int): # percentage of input size
            proportion = (1/100*self.patch_size)**(1/(len(img_shape)-1))
            size = rotate_list(self.input_size)
            size = proportion*np.array(size)
            size = np.round(size).astype(int)
            size[-1]=1
        else:
            size = np.copy(self.patch_size)
        assert len(size) == len(img_shape), "Incorrect patch dimension."

        # get the non zero coordinates
        coords_x, coords_y, coords_z = np.nonzero(arr[:,:,:,0])

        # apply masks in a loop
        for _ in range(self.number_patches):
            # pick random center coordinates among coords
            rd_idx = np.random.randint(0, len(coords_x))
            center = [coords_x[rd_idx], coords_y[rd_idx], coords_z[rd_idx]]
            indexes = []
            for dim, (center_dim, size_dim) in enumerate(zip(center, size)):
                slc = slice(max(0, center_dim - int(size_dim//2)), min(img_shape[dim], center_dim + int(size_dim//2)+ size_dim%2))
                indexes.append(slc)
            arr[tuple(indexes)] = self.value
        
        return torch.from_numpy(arr)
    

class AddBranchTensor(object):

    """
    Add one random branch to the skeleton, from a pool of plausible branches.
    """

    def __init__(self, branch_directory, nb_branches, input_size):
        self.branch_directory = branch_directory
        self.nb_branches = int(nb_branches)
        self.input_size = input_size
    
    def __call__(self, tensor):
        arr = tensor.numpy()
        np.random.seed()
        idx = np.random.randint(0, self.nb_branches)
        coords_branch = np.load(os.path.join(self.branch_directory, f'branch_{idx}.npy'))
        nb_coords = coords_branch.shape[1]
        data = np.ones(nb_coords)
        branch = convert_sparse_to_numpy(data, coords_branch, self.input_size[1:], 'float32')
        arr = arr + branch

        arr = arr.astype('float32')

        return torch.from_numpy(arr)
    

class ContourTensor(object):

    """
    Keep solely the contours of the folds.
    """

    def __init__(self, sample_foldlabel):
        self.sample_foldlabel = sample_foldlabel
    
    def __call__(self, tensor):
        arr = tensor.numpy()
        contours = np.logical_or(arr==30, arr==35) # 79%
        #contours = arr != 60 # terrible
        #arr_foldlabel = self.sample_foldlabel.numpy()
        #contours = np.logical_and(arr_foldlabel>=6000, arr_foldlabel<8000)
        arr = arr * contours

        arr = arr.astype('float32')

        return torch.from_numpy(arr)

