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
Some helper functions are taken from:
https://learnopencv.com/tensorboard-with-pytorch-lightning

"""
import os
import wandb
import json
import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.manifold import TSNE
from collections import OrderedDict
import pandas as pd
import torch.nn.functional as F
import os, re

from sklearn.metrics import roc_auc_score, r2_score
from contrastive.utils.kernel_plotting import wrap_loss_with_kernel_plotting
from contrastive.augmentations import ToPointnetTensor
from contrastive.backbones.densenet import DenseNet
from contrastive.backbones.convnet import ConvNet
from contrastive.backbones.resnet import ResNet, BasicBlock
from contrastive.backbones.convnext import ConvNeXt
#from contrastive.backbones.pointnet import PointNetCls
from contrastive.backbones.projection_heads import *
from contrastive.data.utils import change_list_device
from contrastive.models.tabular_resnet_mlp import TabularResNetMLPEncoder
from contrastive.evaluation.auc_score import regression_roc_auc_score
from contrastive.models.models_utils import *
from contrastive.losses import *
from contrastive.utils.plots.visualize_images import plot_bucket, \
    plot_histogram, plot_histogram_weights, plot_scatter_matrix, \
    plot_scatter_matrix_with_labels
from contrastive.utils.plots.visualize_tsne import plot_tsne
from contrastive.utils.test_timeit import timeit

try:
    from contrastive.utils.plots.visualize_anatomist import Visu_Anatomist
except ImportError:
    print("INFO: you are probably not in a brainvisa env. Probably OK.")

from contrastive.utils.logs import set_root_logger_level, set_file_logger
log = set_file_logger(__file__)



class ContrastiveLearnerFusion(pl.LightningModule):
    
        # =========================
    # CLIPTabular helpers
    # =========================
    def _load_pca_table(self, csv_path: str, id_col: str):
        """
        Loads PCA CSV once into dict: participant_id(str) -> torch.FloatTensor[tab_dim]
        Keys are normalized to match _pid_from_filename(), i.e. WITHOUT "sub-" prefix.
        Example: "sub-123456" or "123456" -> "123456"
        """
        df = pd.read_csv(csv_path)

        if id_col not in df.columns:
            raise ValueError(
                f"[CLIPTabular] id_col='{id_col}' not found in PCA CSV columns. "
                f"First columns: {list(df.columns)[:20]}"
            )

        feat_cols = [c for c in df.columns if c != id_col]
        if len(feat_cols) == 0:
            raise ValueError("[CLIPTabular] PCA CSV has no feature columns (only id_col).")

        # ---- normalize id column to match _pid_from_filename() ----
        ids = df[id_col].astype(str)
        ids = ids.str.replace(r"\.0$", "", regex=True)  # handle float read -> "1001.0"
        ids = ids.str.strip()
        ids = ids.str.replace(r"^sub-", "", regex=True)  # strip "sub-" if present
        ids = ids.str.strip("_-")  # be safe (match old code behavior)

        df[id_col] = ids

        table = {}
        for _, row in df.iterrows():
            pid = row[id_col]  # normalized, e.g. "100206"
            vec = torch.tensor(row[feat_cols].to_numpy(dtype=np.float32))
            table[pid] = vec

        return table, feat_cols




    def _pid_from_filename(self, fn: str) -> str:
        """
        Return the participant id in the SAME format as the old working code:
        - if path contains "sub-XXXX" -> returns "XXXX" (without "sub-")
        - else if digits exist -> returns the longest digit token
        - else -> returns basename without extension
        """
        s = str(fn)
        base = os.path.splitext(os.path.basename(s))[0]

        # Prefer explicit BIDS token if present anywhere in path
        m = re.search(r"sub-([0-9A-Za-z]+)", s)  # capture without "sub-"
        if m:
            return m.group(1).strip("_-")

        # Fallback: if only digits exist, return the longest digit token
        digits = re.findall(r"\d+", s)
        if digits:
            return max(digits, key=len)

        # Final fallback: basename without extension (old behavior)
        return base



    def _tabular_from_filenames(self, filenames):
        """
        returns:
        tab: [B_kept, tab_dim]
        keep_mask: [B] bool (True if pid found)
        kept_pids, missing_pids: debug lists
        """
        vecs = []
        keep = []
        kept_pids = []
        missing_pids = []

        for fn in filenames:
            pid = self._pid_from_filename(fn)
            if pid in self.pca_table:
                vecs.append(self.pca_table[pid])
                keep.append(True)
                kept_pids.append(pid)
            else:
                keep.append(False)
                missing_pids.append(pid)

        keep_mask = torch.tensor(keep, dtype=torch.bool)

        if len(vecs) == 0:
            tab = torch.empty((0, self.tabular_dim), dtype=torch.float32)
        else:
            tab = torch.stack(vecs, dim=0)

        # warn but don't crash
        if len(missing_pids) > 0 and (not hasattr(self, "_clip_missing_warned")):
            log.warning(f"[CLIPTabular] Missing {len(missing_pids)} ids in PCA table (showing 10): {missing_pids[:10]}")
            self._clip_missing_warned = True

        return tab, keep_mask, kept_pids, missing_pids




    def __init__(self, config, sample_data, with_labels=False):
        super(ContrastiveLearnerFusion, self).__init__()
        
        

        if config.multiregion_single_encoder:
            n_datasets = 1
            n_regions = len(config.data)
            log.info("n_datasets 1 because a single encoder is used for multiple regions")
        else:
            n_datasets = len(config.data)
            log.info(f"n_datasets {n_datasets}")

        # define the encoder structure
        self.backbones = nn.ModuleList()
        if config.backbone_name == 'densenet':
            for i in range(n_datasets):
                self.backbones.append(DenseNet(
                    growth_rate=config.growth_rate,
                    block_config=config.block_config,
                    num_init_features=config.num_init_features,
                    num_representation_features=config.backbone_output_size,
                    drop_rate=config.drop_rate,
                    in_shape=config.data[i].input_size))
        elif config.backbone_name == "convnet":
            for i in range(n_datasets):
                self.backbones.append(ConvNet(
                    encoder_depth=config.encoder_depth,
                    filters=config.filters,
                    block_depth=config.block_depth,
                    initial_kernel_size=config.initial_kernel_size,
                    initial_stride=config.initial_stride,
                    max_pool=config.max_pool,
                    num_representation_features=config.backbone_output_size,
                    linear = config.linear_in_backbone,
                    adaptive_pooling=config.adaptive_pooling,
                    drop_rate=config.drop_rate,
                    in_shape=config.data[i].input_size))
        elif config.backbone_name == 'resnet':
            for i in range(n_datasets):
                self.backbones.append(ResNet(
                    block=BasicBlock,
                    layers=config.layers,
                    channels=config.channels,
                    in_channels=1,
                    num_classes=config.backbone_output_size,
                    zero_init_residual=config.zero_init_residual,
                    dropout_rate=config.drop_rate,
                    out_block=None,
                    prediction_bias=False,
                    initial_kernel_size=config.initial_kernel_size,
                    initial_stride=config.initial_stride,
                    adaptive_pooling=config.adaptive_pooling,
                    linear_in_backbone=config.linear_in_backbone))
        elif config.backbone_name == 'convnext':
            for i in range(n_datasets):
                self.backbones.append(ConvNeXt(
                    in_chans=1,
                    num_classes=config.backbone_output_size,
                    nb_blocks=config.nb_blocks,
                    depths=config.depth,
                    dims=config.dims,
                    initial_stride=config.initial_stride,
                    initial_kernel_size=config.initial_kernel_size,
                    kernel_size=config.kernel_size,
                    adaptive_pooling=config.adaptive_pooling))
        # elif config.backbone_name == 'pointnet':
        #     self.backbone = PointDataModule_LearningFalse)
        else:
            raise ValueError(f"No underlying backbone with backbone name {config.backbone_name}")
        
        # freeze the backbone weights if required
        if 'freeze_encoders' in config.keys() and config.freeze_encoders:
            for backbone in self.backbones:
                backbone.freeze()
            log.info("The model's encoders weights are frozen. Set 'freeze_encoders' \
                      in the config to False to unfreeze them.")

        # rename variables
        concat_latent_spaces_size = config.backbone_output_size * n_datasets

        # build converter (if required) and set the latent space size according to it
        converter, num_representation_features = build_converter(config, concat_latent_spaces_size)
        self.converter = converter

        # set up the projection head layers shapes
        layers_shapes = get_projection_head_shape(config, num_representation_features)
        output_shape = layers_shapes[-1]

        # set projection head activation
        activation = config.projection_head_name
        log.debug(f"activation = {activation}")

        if config.multiple_projection_heads:
            # Evaluation: need to initialize the right number of projection heads for weight mapping
            n_regions = len(config.data)
            self.projection_head = nn.ModuleList()
            for reg in range(n_regions):
                if config.linear_in_backbone:
                    self.projection_head.append(ProjectionHead(
                    num_representation_features=num_representation_features,
                    layers_shapes=layers_shapes,
                    activation=activation,
                    drop_rate=config.ph_drop_rate))
                else:
                    # add a variable size linear layer to each projection head
                    layers_shapes_including_variable = layers_shapes.copy()
                    # TODO: make it compatible with ResNet !
                    backbone_output_shape = [config.data[reg].input_size[1] // 2**config.encoder_depth,
                                            config.data[reg].input_size[2] // 2**config.encoder_depth,
                                            config.data[reg].input_size[3] // 2**config.encoder_depth]
                    backbone_output_shape = config.filters[-1]*np.prod(backbone_output_shape)
                    layers_shapes_including_variable = [backbone_output_shape] + layers_shapes_including_variable
                    self.projection_head.append(ProjectionHead(
                        num_representation_features=num_representation_features,
                        layers_shapes=layers_shapes_including_variable,
                        activation=activation,
                        drop_rate=config.ph_drop_rate))
        else:
            self.projection_head = ProjectionHead(
                num_representation_features=num_representation_features,
                layers_shapes=layers_shapes,
                activation=activation,
                drop_rate=config.ph_drop_rate)

        # set up class keywords
        self.config = config
        self.with_labels = with_labels
        self.n_datasets = n_datasets
        if self.config.multiregion_single_encoder:
            self.n_regions = n_regions
        self.sample_data = sample_data
        self.sample_i = np.array([])
        self.sample_j = np.array([])
        self.sample_k = np.array([])
        self.sample_filenames = []
        self.num_representation_features = num_representation_features
        self.output_shape = output_shape
        self.lr = self.config.lr
        if self.config.environment == "brainvisa":
            self.visu_anatomist = Visu_Anatomist()

        if 'class_weights' in config.keys():
            self.class_weights = torch.Tensor(config.class_weights).to(device=config.device)
        else:
            self.class_weights = None

        # Keeps track of losses
        self.training_step_outputs = []
        self.validation_step_outputs = []
        if self.config.multiple_projection_heads or self.config.multiregion_single_encoder:
            self.training_step_idxs_region = [] 
            self.validation_step_idxs_region = []
        if self.config.mode == "encoder" and self.config.contrastive_model=='BarlowTwins':
            self.training_step_loss_inv = []
            self.training_step_loss_redund = []
            self.validation_step_loss_inv = []
            self.validation_step_loss_redund = []

        # Output of intermediate layer of ProjectionHead
        self.activation={}
        self.loss_fn = None
        if self.config.contrastive_model == "Yaware":
            self.loss_fn = GeneralizedSupervisedNTXenLoss(
                label_csv=config.data[0].subject_labels_file,
                label_cols=config.label_names,
                kernel='rbf',
                temperature=config.temperature,
                sigma=config.sigma_labels,
                return_logits=True,
                sigma_factor=config.sigma_factor,
                sigma_mode=config.get("sigma_mode", "diag"),
                standardize_y=config.get("standardize_y", False),
                shrinkage=config.get("shrinkage", 0.1),
                cosine_temperature=config.get("cosine_temperature", None),
            )

        
        # -------------------------
        # CLIPTabular setup (Y-aware-like: tab dim comes from batch labels)
        # -------------------------
        self.is_clip_tabular = (self.config.contrastive_model == "CLIPTabular")
        if self.is_clip_tabular:
            # We DO NOT load PCA here. We will use the batch `labels` like Y-aware does.
            self.tabular_dim = None  # will be inferred from labels on first batch

            self.tabular_encoder = None  # lazy init
            self.clip_loss_fn = CLIPLoss(
                temperature=config.clip_temperature,
                gather_ddp=config.get("gather_ddp", True),
                return_logits=True,
            )

            log.info(
                f"[CLIPTabular] Lazy tabular encoder init. "
                f"tab_dim will be inferred from batch labels. embed_dim={self.output_shape} "
                f"temp={self.config.get('clip_temperature', 0.07)}"
            )
      


    def forward(self, x, idx_region=None):
        # log.info(f"x shape: {x.shape}")
        embeddings = []
        for i in range(self.n_datasets):
            embedding = self.backbones[i].forward(x[i])
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings, dim=1)
        embeddings = self.converter.forward(embeddings)
        if idx_region is not None:
            out = self.projection_head[idx_region].forward(embeddings)
        else:
            out = self.projection_head.forward(embeddings)
        return out
    
    def get_full_inputs_from_batch_with_region_idx(self, batch):
        full_inputs = []
        for (inputs, filenames, idx_region) in batch:  # loop over datasets
            if self.config.backbone_name == 'pointnet':
                inputs = torch.squeeze(inputs).to(torch.float)
            full_inputs.append(inputs)
        
        inputs = full_inputs
        idx_region = idx_region.detach().cpu().numpy()[0]
        return (inputs, filenames, idx_region)


    def get_full_inputs_from_batch(self, batch):
        full_inputs = []
        for (inputs, filenames) in batch:  # loop over datasets
            if self.config.backbone_name == 'pointnet':
                inputs = torch.squeeze(inputs).to(torch.float)
            full_inputs.append(inputs)
        
        inputs = full_inputs
        return (inputs, filenames)
    
    def get_full_inputs_from_batch_with_labels(self, batch):
        #print("A-T-ON ENCORE BESOIN DE LA VIEW3 ?")
        full_inputs = []
        full_view3 = []
        for (inputs, filenames, labels, view3) in batch: # loop over datasets
            if self.config.backbone_name == 'pointnet':
                inputs = torch.squeeze(inputs).to(torch.float)
            full_inputs.append(inputs)
            full_view3.append(view3)
        
        inputs = full_inputs
        view3 = full_view3
        return (inputs, filenames, labels, view3)


    def load_pretrained_model(self, pretrained_model_path, encoder_only=False,
                              convolutions_only=False, freeze_loaded_layers=False,
                              freeze_bias=False):
        """Load weights stored in a state_dict at pretrained_model_path
        """

        pretrained_state_dict = torch.load(pretrained_model_path)['state_dict']
        if convolutions_only:
            pretrained_state_dict = OrderedDict(
                {k: v for k, v in pretrained_state_dict.items()
                 if 'encoder' in k and
                 ('conv' in k or 'norm' in k)})
        elif encoder_only:
            pretrained_state_dict = OrderedDict(
                {k: v for k, v in pretrained_state_dict.items()
                 if 'encoder' in k})

        model_dict = self.state_dict()

        loaded_layers = []
        for n, p in pretrained_state_dict.items():
            if n in model_dict:
                loaded_layers.append(n)
                model_dict[n] = p

        self.load_state_dict(model_dict)

        not_loaded_layers = [
            key for key in model_dict.keys() if key not in loaded_layers]
        # print(f"Loaded layers = {loaded_layers}")
        log.info(f"Layers not loaded = {not_loaded_layers}")

        # freeze loaded layers
        if freeze_loaded_layers:
            for name, para in self.named_parameters():
                if name in loaded_layers:
                    para.requires_grad = False
                if 'bias' in name and not freeze_bias:
                    para.requires_grad = True

        
    def _ensure_tabular_encoder(self, labels: torch.Tensor):
            """Initialize tabular encoder lazily using labels dimension (Y-aware style)."""
            if self.tabular_encoder is not None:
                return

            if labels is None:
                raise ValueError("[CLIPTabular] labels is None but CLIPTabular requires labels.")

            if labels.ndim == 1:
                labels = labels.view(-1, 1)

            tab_dim = int(labels.shape[1])
            self.tabular_dim = tab_dim

            self.tabular_encoder = TabularResNetMLPEncoder(
                in_dim=self.tabular_dim,
                out_dim=self.output_shape,
                width=self.config.get("tab_width", 256),
                depth=self.config.get("tab_depth", 3),
                dropout=self.config.get("tab_dropout", 0.0),
                hidden_mult=self.config.get("tab_hidden_mult", 2.0),
                in_norm=self.config.get("tab_in_norm", True),
            ).to(self.device)

            log.info(f"[CLIPTabular] Initialized tabular encoder with in_dim={self.tabular_dim}") 

    def custom_histogram_adder(self):
        """Builds histogram for each model parameter.
        """
        # iterating through all parameters
        for name, params in self.named_parameters():
            self.loggers[0].experiment.add_histogram(
                name,
                params,
                self.current_epoch)


    def plot_histograms(self):
        """Plots all zii, zjj, zij and weights histograms"""

        # Computes histogram of sim_zii
        histogram_sim_zii = plot_histogram(self.sim_zii, buffer=True)
        self.loggers[0].experiment.add_image(
            'histo_sim_zii', histogram_sim_zii, self.current_epoch)

        # Computes histogram of sim_zjj
        histogram_sim_zjj = plot_histogram(self.sim_zjj, buffer=True)
        self.loggers[0].experiment.add_image(
            'histo_sim_zjj', histogram_sim_zjj, self.current_epoch)

        # Computes histogram of sim_zij
        histogram_sim_zij = plot_histogram(self.sim_zij, buffer=True)
        self.loggers[0].experiment.add_image(
            'histo_sim_zij', histogram_sim_zij, self.current_epoch)

        # Computes histogram of weights
        # histogram_weights = plot_histogram_weights(self.weights,
        #                                            buffer=True)
        # self.loggers[0].experiment.add_image(
        #     'histo_weights', histogram_weights, self.current_epoch)


    def plot_scatter_matrices(self, dataloader, key):
        """Plots scatter matrices of output and representations spaces"""
        
        funcs = {'outputs': self.compute_outputs_skeletons,
                 'representations': self.compute_outputs_skeletons}
        
        for name, func in funcs.items():
            r = func(dataloader)
            X = r[0]  # get inputs

            if self.with_labels:
                labels = r[2]
                scatter_matrix_with_labels = \
                    plot_scatter_matrix_with_labels(X, labels, buffer=True)
                self.loggers[0].experiment.add_image(
                    f'scatter_matrix_{name}_with_labels_' + key,
                    scatter_matrix_with_labels,
                    self.current_epoch)
            else:
                scatter_matrix = plot_scatter_matrix(X, buffer=True)
                self.loggers[0].experiment.add_image(
                    f'scatter_matrix_{name}',
                    scatter_matrix,
                    self.current_epoch)
            
            if (self.config.mode == "regresser") and (name =='output'):
                score = r2_score(labels, X)
            else:
                score = 0

        return score


    def plot_views(self):
        """Plots different 3D views"""
        image_input_i = plot_bucket(self.sample_i, buffer=True)
        self.loggers[0].experiment.add_image(
            'input_i', image_input_i, self.current_epoch)
        image_input_j = plot_bucket(self.sample_j, buffer=True)
        self.loggers[0].experiment.add_image(
            'input_j', image_input_j, self.current_epoch)

        # Plots view using anatomist
        if self.config.environment == "brainvisa":
            image_input_i = self.visu_anatomist.plot_bucket(
                self.sample_i, buffer=True)
            self.loggers[0].experiment.add_image(
                'input_ana_i: ',
                image_input_i, self.current_epoch)
            # self.loggers[0].experiment.add_text(
            #     'filename: ',self.sample_filenames[0], self.current_epoch)
            image_input_j = self.visu_anatomist.plot_bucket(
                self.sample_j, buffer=True)
            self.loggers[0].experiment.add_image(
                'input_ana_j: ',
                image_input_j, self.current_epoch)
            if len(self.sample_k) != 0:
                image_input_k = self.visu_anatomist.plot_bucket(
                    self.sample_k, buffer=True)
                self.loggers[0].experiment.add_image(
                    'input_ana_k: ',
                    image_input_k, self.current_epoch)

    def configure_optimizers(self):
        """Adam optimizer"""
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay)
        return_dict = {"optimizer": optimizer}

        if 'scheduler' in self.config.keys() and self.config.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma)
            return_dict["lr_scheduler"] = {"scheduler": scheduler,
                                           "interval": "epoch"}


        """
        if 'scheduler' in self.config.keys() and self.config.scheduler:  
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',           # We want to minimize the loss
                factor=self.config.factor, # Divide the learning rate by 3
                patience=self.config.step_size, # Wait for 10 epochs without improvement
                threshold=self.config.threshold_plateau, # Minimum loss reduction of 10% to be considered an improvement
                threshold_mode='rel', # Relative threshold, i.e., 10% relative decrease in loss
            )
            return_dict["lr_scheduler"] = {"scheduler": scheduler,
                                        "monitor": 'val_loss',
                                        "interval": "epoch",
                                        "frequency": 1}
        """
        return return_dict
    
    
    def barlow_twins_loss(self, z_i, z_j):
        "Loss function for SSL (BarlowTwins)"
        loss = BarlowTwinsLoss(lambda_param=self.config.lambda_BT,
                               correlation=self.config.BT_correlation,
                               device=self.config.device)
        return loss.forward(z_i, z_j)
    
    def vic_reg_loss(self, z_i, z_j):
        "Loss function for SSL (VicReg)"
        loss = VicRegLoss(device=self.config.device,
                          lmbd=self.config.lambda_VR / float(self.config.backbone_output_size),
                          u=1,v=1,
                          epsilon=self.config.epsilon_VR)
        return loss.forward(z_i, z_j)

    def nt_xen_loss(self, z_i, z_j):
        """Loss function for contrastive (SimCLR)"""
        loss = NTXenLoss(temperature=self.config.temperature,
                         return_logits=True)
        return loss.forward(z_i, z_j)
    
    # def generalized_supervised_nt_xen_loss(self, z_i, z_j, labels):
    #     """Loss function for supervised contrastive"""
    #     loss_fn = GeneralizedSupervisedNTXenLoss(
    #         kernel='rbf',
    #         temperature=self.config.temperature,
    #         sigma=self.config.sigma_labels,
    #         return_logits=True)

    #     return loss_fn(z_i, z_j, labels)
    
    def generalized_supervised_nt_xen_loss(self, z_i, z_j, labels, step=None):
        return self.loss_fn(z_i, z_j, labels, step=step)


    def cross_entropy_loss_classification(self, output_i, output_j, labels):
        """Loss function for decoder"""
        loss = CrossEntropyLoss_Classification(device=self.device,
                                               class_weights=self.class_weights)
        return loss.forward(output_i, output_j, labels)

    def mse_loss_regression(self, output_i, output_j, labels):
        """Loss function for decoder"""
        loss = MSELoss_Regression(device=self.device)
        return loss.forward(output_i, output_j, labels)
    



    def compute_clip_loss(self, z_img, labels, return_sims: bool = False):
        """
        z_img: [B, D]
        labels: [B, tab_dim] (aligned by the dataset, like Y-aware)
        """
        self._ensure_tabular_encoder(labels)

        tab = labels.float().to(z_img.device)
        z_tab = self.tabular_encoder(tab)  # [B, D]

        z_img = F.normalize(z_img, dim=1)
        z_tab = F.normalize(z_tab, dim=1)

        out = self.clip_loss_fn(z_img, z_tab)
        loss = out[0] if isinstance(out, (tuple, list)) else out

        if not return_sims:
            return loss

        sim_zij = (z_img @ z_tab.t()).detach()
        sim_zii = (z_img @ z_img.t()).detach()
        sim_zjj = (z_tab @ z_tab.t()).detach()
        return loss, sim_zij, sim_zii, sim_zjj







    def training_step(self, train_batch, batch_idx):
        """Training step.
        """
        if self.config.with_labels:
            inputs, filenames, labels, view3 = \
                self.get_full_inputs_from_batch_with_labels(train_batch)
        elif self.config.multiple_projection_heads or self.config.multiregion_single_encoder:
            inputs, filenames, idx_region = self.get_full_inputs_from_batch_with_region_idx(train_batch)
        else:
            inputs, filenames = self.get_full_inputs_from_batch(train_batch)

        # print("TRAINING STEP", inputs.shape)
        input_i = [inputs[i][:, 0, ...] for i in range(self.n_datasets)]
        input_j = [inputs[i][:, 1, ...] for i in range(self.n_datasets)]
        if self.config.multiple_projection_heads:
            z_i = self.forward(input_i, idx_region=idx_region)
            z_j = self.forward(input_j, idx_region=idx_region)
        else:
            z_i = self.forward(input_i)
            z_j = self.forward(input_j)

        # compute the right loss depending on the learning mode
        if self.config.mode == "decoder":
            sample = inputs[:, 2, :]
            batch_loss = self.cross_entropy_loss(sample, z_i, z_j)
        elif self.config.mode == "classifier":
            batch_loss = self.cross_entropy_loss_classification(
                z_i, z_j, labels)
            batch_label_loss = torch.tensor(0.)
        elif self.config.mode == "regresser":
            batch_loss = self.mse_loss_regression(z_i, z_j, labels)
            batch_label_loss = torch.tensor(0.)
        # elif self.config.proportion_pure_contrastive != 1:
        #     batch_loss, batch_label_loss, \
        #         sim_zij, sim_zii, sim_zjj, correct_pair, weights = \
        #         self.generalized_supervised_nt_xen_loss(z_i, z_j, labels)
        elif self.config.contrastive_model=='SimCLR':
            batch_loss, sim_zij, sim_zii, sim_zjj = self.nt_xen_loss(z_i, z_j)
        elif self.config.contrastive_model=='BarlowTwins':
            batch_loss, loss_invariance, loss_redundancy = self.barlow_twins_loss(z_i,z_j)
        elif self.config.contrastive_model=='VicReg':
            batch_loss = self.vic_reg_loss(z_i,z_j)
        elif self.config.contrastive_model == 'Yaware':
            batch_loss, sim_zij, sim_zii, sim_zjj = self.generalized_supervised_nt_xen_loss(z_i, z_j, labels, step=self.global_step)
        ##################################################################################################
        elif self.config.contrastive_model == "CLIPTabular":
            if batch_idx == 0:
                loss_i, sim_zij, sim_zii, sim_zjj = self.compute_clip_loss(z_i, labels, return_sims=True)
            else:
                loss_i = self.compute_clip_loss(z_i, labels, return_sims=False)

            loss_j = self.compute_clip_loss(z_j, labels, return_sims=False)
            batch_loss = 0.5 * (loss_i + loss_j)





        ####################################################################################################

        #TODO: add error if None of these names
        #encoder peut être du contrastive supervisé !! gérer ce cas là...

        # # Only computes graph on first step
        # if self.global_step == 1:
        #     self.loggers[0].experiment.add_graph(self, [input_i])

        # Records sample for first batch of each epoch
        ## OBSOLETE
        if batch_idx == 0:
            self.sample_i = change_list_device(input_i, 'cpu')
            self.sample_j = change_list_device(input_j, 'cpu')
            self.sample_filenames = filenames
            if self.config.with_labels:
                self.sample_k = change_list_device(view3, 'cpu')
                self.sample_labels = labels
            if self.config.mode == "encoder" and self.config.contrastive_model=='SimCLR':
                self.sim_zij = sim_zij * self.config.temperature
                self.sim_zii = sim_zii * self.config.temperature
                self.sim_zjj = sim_zjj * self.config.temperature
            if self.config.mode == "encoder" and self.config.contrastive_model=='Yaware':
                self.sim_zij = sim_zij * self.config.temperature
                self.sim_zii = sim_zii * self.config.temperature
                self.sim_zjj = sim_zjj * self.config.temperature
            ################################################################################
            if self.config.mode == "encoder" and self.config.contrastive_model == "CLIPTabular" and batch_idx == 0:
                self.sim_zij = sim_zij
                self.sim_zii = sim_zii
                self.sim_zjj = sim_zjj


            # ################################################################################        
            if self.config.environment == 'brainvisa' and self.config.checking:
                bv_checks(self, filenames)  # untested
        
        # logs - a dictionary
        #self.log('Loss/Train', float(batch_loss), on_epoch=True)
        logs = {"train_loss": float(batch_loss)}
        if self.config.contrastive_model=='BarlowTwins':
            logs["train_loss_inv"] = float(loss_invariance)
            logs["train_loss_redund"] = float(loss_redundancy)

        self.training_step_outputs.append(batch_loss)
        if self.config.contrastive_model=='BarlowTwins':
            # decompose loss in invariance and redundancy term
            self.training_step_loss_inv.append(loss_invariance)
            self.training_step_loss_redund.append(loss_redundancy)
        if self.config.multiple_projection_heads or self.config.multiregion_single_encoder:
            self.training_step_idxs_region.append(idx_region)

        batch_dictionary = {
            # REQUIRED: It is required for us to return "loss"
            "loss": batch_loss,
            # optional for batch logging purposes
            "log": logs}

        if self.config.scheduler:
            batch_dictionary['learning_rate'] = self.optimizers().param_groups[0]['lr']

        if self.config.with_labels and self.config.mode == 'encoder'\
           and self.config.proportion_pure_contrastive != 1:
            # add label_loss (a part of the loss) to log
            self.log('train_label_loss', float(batch_label_loss))
            logs['train_label_loss'] = float(batch_label_loss)
            batch_dictionary['label_loss'] = batch_label_loss

        # if required, save views from first batch of first epoch (first dataset)
        if self.config.save_views and self.current_epoch==0 and batch_idx==0:
            sample_i = change_list_device(input_i, 'cpu')
            sample_j = change_list_device(input_j, 'cpu')
            sample_i = sample_i[0].numpy()
            sample_j = sample_j[0].numpy()
            print('saving augmented views')
            dir_to_save = './logs/views/'
            if not os.path.isdir(dir_to_save):
                os.mkdir(dir_to_save)
            np.save(os.path.join(dir_to_save, 'view1.npy'), sample_i)
            np.save(os.path.join(dir_to_save, 'view2.npy'), sample_j)

        return batch_dictionary

    def compute_outputs_skeletons(self, loader):
        """Computes the outputs of the model for each crop.

        This includes the projection head"""

        # Initialization
        X = torch.zeros([0, self.output_shape]).cuda()
        labels_all = torch.zeros(
            [0, len(self.config.label_names)]).cuda()
        filenames_list = []
        transform = ToPointnetTensor()

        # Computes embeddings without computing gradient
        with torch.no_grad():
            for batch in loader:
                if self.config.with_labels:
                    inputs, filenames, labels, _ = \
                        self.get_full_inputs_from_batch_with_labels(batch)
                elif self.config.multiple_projection_heads:
                    inputs, filenames, idx_region = self.get_full_inputs_from_batch_with_region_idx(batch)
                else:
                    inputs, filenames = self.get_full_inputs_from_batch(batch)
                
                # First views of the whole batch
                inputs = change_list_device(inputs, 'cuda')
                # model = self.cuda()
                input_i = [inputs[i][:, 0, ...] for i in range(self.n_datasets)]
                input_j = [inputs[i][:, 1, ...] for i in range(self.n_datasets)]
                if self.config.backbone_name == 'pointnet':
                    input_i = transform(input_i.cpu()).cuda().to(torch.float)
                    input_j = transform(input_j.cpu()).cuda().to(torch.float)
                if self.config.multiple_projection_heads:
                    X_i = self.forward(input_i, idx_region=idx_region)
                    X_j = self.forward(input_j, idx_region=idx_region)
                else:
                    X_i = self.forward(input_i)
                    # Second views of the whole batch
                    X_j = self.forward(input_j)

                # First views and second views are put side by side
                X_reordered = torch.cat([X_i, X_j], dim=-1)
                X_reordered = X_reordered.view(-1, X_i.shape[-1])
                X = torch.cat((X, X_reordered.cuda()), dim=0)

                # concat filenames
                filenames_duplicate = [item
                                       for item in filenames
                                       for repetitions in range(2)]
                filenames_list = filenames_list + filenames_duplicate
                del inputs

                # deal with labels if required
                if self.config.with_labels:
                    # We now concatenate the labels
                    labels_reordered = torch.cat([labels, labels], dim=-1)
                    labels_reordered = labels_reordered.view(-1, labels.shape[-1])
                    # At the end, labels are concatenated
                    labels_all = torch.cat((labels_all, labels_reordered.cuda()),
                                        dim=0)
        
        if self.config.with_labels:
            return X.cpu(), filenames_list, labels_all.cpu()
        else:
            return X.cpu(), filenames_list
    
    def compute_output_probabilities(self, loader):
        """Only available in classifier mode.
        Gets the output of the model and convert it to probabilities thanks to softmax."""
        if self.config.mode == 'classifier':
            X, filenames_list, labels_all = self.compute_outputs_skeletons(
                loader)
            # compute the mean of the two views' outputs
            X = (X[::2, ...] + X[1::2, ...]) / 2
            # remove the doubleing of labels
            labels_all = labels_all[::2]
            filenames_list = filenames_list[::2]
            X = nn.functional.softmax(X, dim=1)
            return X, labels_all, filenames_list
        else:
            raise ValueError(
                "The config.mode is not 'classifier'. "
                "You shouldn't compute probabilities with another mode.")

    def compute_output_auc(self, loader):
        """Only available in classifier and regresser modes.
        Computes the auc from the outputs of the model and the associated labels."""
        # we don't apply transforms for the AUC computation
        loader.dataset.transform = False

        X, _, labels = self.compute_outputs_skeletons(loader)
        # compute the mean of the two views' outputs
        X = (X[::2, ...] + X[1::2, ...]) / 2
        # remove the doubleing of labels
        labels = labels[::2]
        if self.config.mode == "regresser":
            auc = regression_roc_auc_score(labels, X[:, 0])
        else:
            X = nn.functional.softmax(X, dim=1)
            if self.config.nb_classes==2:
                auc = roc_auc_score(labels, X[:, 1])
            else:
                auc = roc_auc_score(labels, X, multi_class='ovr', average='weighted')

        
        # put augmentations back to normal
        loader.dataset.transform = self.config.apply_augmentations

        return auc


    def compute_decoder_outputs_skeletons(self, loader):
        """Computes the outputs of the model for each crop,
        but for decoder mode.

        This includes the projection head"""

        # Initialization
        X = torch.zeros([0, 2, 20, 40, 40]).cpu()
        filenames_list = []

        # Computes embeddings without computing gradient
        with torch.no_grad():
            for batch in loader:
                if self.config.with_labels:
                    (inputs, filenames, _, _) = self.get_full_inputs_from_batch_with_labels(batch)
                elif self.config.multiple_projection_heads:
                    (inputs, filenames, _) = self.get_full_inputs_from_batch_with_region_idx(batch)
                else:
                    (inputs, filenames) = self.get_full_inputs_from_batch(batch)
                # First views of the whole batch
                inputs = change_list_device(inputs, 'cuda')
                model = self.cuda()
                X_i = model.forward(inputs[:, 0, :])
                print(f"shape X and X_i: {X.shape}, {X_i.shape}")
                # First views re put side by side
                X = torch.cat((X, X_i.cpu()), dim=0)
                filenames_duplicate = [item
                                        for item in filenames]
                filenames_list = filenames_list + filenames_duplicate
                del inputs

        return X, filenames_list
    

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook


    def compute_representations(self, loader):
        """Computes representations for each crop.

        Representation are before the projection head,
        Or after the first projection head layer when
        the linear layer is not in the backbone."""

        # Initialization
        X = torch.zeros(
            [0, self.num_representation_features]).cuda()
        labels_all = torch.zeros(
            [0, len(self.config.label_names)]).cuda()
        filenames_list = []

        # Computes representation (without gradient computation)
        with torch.no_grad():
            for batch in loader:
                if self.config.with_labels:
                    (inputs, filenames, labels, _) = self.get_full_inputs_from_batch_with_labels(batch)
                else:
                    inputs, filenames = self.get_full_inputs_from_batch(batch)
                
               # deal with devices
                if self.config.device != 'cpu':
                    inputs = change_list_device(inputs, 'cuda')
                else:
                    inputs = change_list_device(inputs, 'cpu')
                if self.config.device != 'cpu':
                    model = self.cuda()
                else:
                    model = self.cpu()
                # deal with pointnet
                if self.config.backbone_name == 'pointnet':
                    inputs = torch.squeeze(inputs).to(torch.float)
                
                input_i = [inputs[k][:, 0, ...] for k in range(self.n_datasets)]
                input_j = [inputs[k][:, 1, ...] for k in range(self.n_datasets)]

                # First views of the whole batch               
                X_i = []
                for k in range(self.n_datasets):
                    embedding = model.backbones[k].forward(input_i[k])
                    X_i.append(embedding)
                X_i = torch.cat(X_i, dim=1)
                X_i = self.converter.forward(X_i)
                if self.config.multiple_projection_heads and not self.config.linear_in_backbone:
                    self.projection_head[0].layers.Linear0.register_forward_hook(self.get_activation('Linear0'))
                    self.projection_head[0].forward(X_i)
                    X_i = self.activation['Linear0']

                # Second views of the whole batch
                X_j = []
                for k in range(self.n_datasets):
                    embedding = model.backbones[k].forward(input_j[k])
                    X_j.append(embedding)
                X_j = torch.cat(X_j, dim=1)
                X_j = self.converter.forward(X_j)
                if self.config.multiple_projection_heads and not self.config.linear_in_backbone:
                    self.projection_head[0].layers.Linear0.register_forward_hook(self.get_activation('Linear0'))
                    self.projection_head[0].forward(X_j)
                    X_j = self.activation['Linear0']

                # First views and second views are put side by side
                X_reordered = torch.cat([X_i, X_j], dim=-1)
                X_reordered = X_reordered.view(-1, X_i.shape[-1])
                X = torch.cat((X, X_reordered.cuda()), dim=0)
                # print(f"filenames = {filenames}")
                filenames_duplicate = [
                    item for item in filenames
                    for repetitions in range(2)]
                filenames_list = filenames_list + filenames_duplicate
                del inputs

                # deal with labels if required
                if self.config.with_labels:
                    # We now concatenate the labels
                    labels_reordered = torch.cat([labels, labels], dim=-1)
                    labels_reordered = labels_reordered.view(-1, labels.shape[-1])
                    # At the end, labels are concatenated
                    labels_all = torch.cat((labels_all, labels_reordered.cuda()),
                                        dim=0)
        
        if self.config.with_labels:
            return X.cpu(), filenames_list, labels_all.cpu()
        else:
            return X.cpu(), filenames_list


    def plotting_now(self):
        """Tells if it is the right epoch to plot the tSNE."""
        if self.config.nb_epochs_per_tSNE <= 0:
            return False
        elif self.current_epoch % self.config.nb_epochs_per_tSNE == 0 \
                or self.current_epoch >= self.config.max_epochs:
            return True
        else:
            return False

    def plotting_matrices_now(self):
        if self.config.nb_epochs_per_matrix_plot <= 0:
            return False
        elif (self.current_epoch % self.config.nb_epochs_per_matrix_plot == 0)\
                or (self.current_epoch >= self.config.max_epochs):
            return True
        else:
            return False
        
    def save_model_weights(self):
        """Tells if it is the right epoch to save model weights."""
        if self.config.nb_epochs_per_weight_save <= 0:
            return False
        elif self.current_epoch % self.config.nb_epochs_per_weight_save == 0 \
                or self.current_epoch >= self.config.max_epochs:
            return True
        else:
            return False


    def compute_tsne(self, loader, register):
        """Computes t-SNE.

        It is computed either in the representation
        or in the output space"""

        if register == "output":
            func = self.compute_outputs_skeletons
        elif register == "representation":
            func = self.compute_representations
        else:
            raise ValueError(
                "Argument register must be either 'output' or 'representation'")

        if self.config.with_labels:
            X, _, labels = func(loader)
        else:
            X, _ = func(loader)

        tsne = TSNE(n_components=2, perplexity=5, init='pca', random_state=50)

        Y = X.detach().numpy()

        # Makes the t-SNE fit
        X_tsne = tsne.fit_transform(Y)

        # Returns tsne embeddings
        if self.config.with_labels:
            return X_tsne, labels
        else:
            return X_tsne
    



       

    def save_best_auc_model(self, current_val_auc, current_train_auc, save_path='./logs/'):
        """Saves aucs and model weights if best val auc"""
        if self.current_epoch == 0:
            best_val_auc = 0
            best_train_auc = 0
        elif self.current_epoch > 0:
            with open(save_path + "best_model_params.json", 'r') as file:
                best_model_params = json.load(file)
                best_val_auc = best_model_params['best_val_auc']
                best_train_auc = best_model_params['best_train_auc']

        if current_val_auc > best_val_auc:
            torch.save({'state_dict': self.state_dict()},
                       save_path + 'best_model_weights.pt')
            best_model_params = {
                'epoch': self.current_epoch,
                'best_val_auc': current_val_auc,
                'best_train_auc': current_train_auc}
            with open(save_path + "best_model_params.json", 'w') as file:
                json.dump(best_model_params, file)
            best_val_auc = current_val_auc
            best_train_auc = current_train_auc
        
        # log the best AUC for wandb (if used)
        if self.config.wandb.grid_search:
            self.loggers[1].log_metrics({'AUC/Val_best': best_val_auc},
                                        step=self.current_epoch)
            self.loggers[1].log_metrics({'AUC/Train_best': best_train_auc},
                                        step=self.current_epoch)
        return best_val_auc, best_train_auc
    
    def save_best_criterion_model(self, current_val_auc, current_train_auc, save_path='./logs/'):
        """Saves best parameters if best criterion"""
        if self.current_epoch <= 5: # takes best model only after epoch 5
            best_val_auc = 0
            best_train_auc = 0
            best_criterion = 0
        elif self.current_epoch > 5:
            with open(save_path + "best_model_params.json", 'r') as file:
                best_model_params = json.load(file)
                best_val_auc = best_model_params['best_val_auc']
                best_train_auc = best_model_params['best_train_auc']
                best_criterion = best_model_params['best_criterion']

        current_criterion = compute_grid_search_criterion(
                                current_train_auc,
                                current_val_auc,
                                lambda_gs_crit=self.config.wandb.lambda_gs_crit)

        if ((current_criterion < best_criterion) or (best_criterion == 0)):
            torch.save({'state_dict': self.state_dict()},
                       save_path + 'best_model_weights.pt')
            best_model_params = {
                'epoch': self.current_epoch,
                'best_val_auc': current_val_auc,
                'best_train_auc': current_train_auc,
                'best_criterion': current_criterion}
            with open(save_path + "best_model_params.json", 'w') as file:
                json.dump(best_model_params, file)
            best_val_auc = current_val_auc
            best_train_auc = current_train_auc
            best_criterion = current_criterion
        
        # log the best AUC for wandb (if used)
        if self.config.wandb.grid_search:
            self.loggers[1].log_metrics({'AUC/Val_best': best_val_auc},
                                        step=self.current_epoch)
            self.loggers[1].log_metrics({'AUC/Train_best': best_train_auc},
                                        step=self.current_epoch)
            
        return best_val_auc, best_train_auc

    def on_train_epoch_end(self):
        """Computation done at the end of the epoch"""

        # score = 0
        if self.config.mode in ["encoder", "regresser"]:
            # Computes t-SNE both in representation and output space
            if self.plotting_now():
                print("Computing tsne\n")
                X_tsne = self.compute_tsne(
                    self.sample_data.train_dataloader(), "output")
                image_TSNE = plot_tsne(X_tsne, buffer=True)
                self.loggers[0].experiment.add_image(
                    'TSNE output image', image_TSNE, self.current_epoch)
                X_tsne = self.compute_tsne(
                    self.sample_data.train_dataloader(), "representation")
                image_TSNE = plot_tsne(X_tsne, buffer=True)
                self.loggers[0].experiment.add_image(
                    'TSNE representation image',
                    image_TSNE, self.current_epoch)
            
            # Plots scatter matrices
            if self.plotting_matrices_now() and (self.config.contrastive_model=='SimCLR'):
                # Plots zxx and weights histograms
                if (self.config.mode == "encoder"):
                    self.plot_histograms()

                # Plots scatter matrices
                self.plot_scatter_matrices(
                     self.sample_data.train_dataloader(),
                     "train",
                )

                # # Plots scatter matrices with label values
                # score = self.plot_scatter_matrices_with_labels(
                #     self.sample_data.train_dataloader(),
                #     "train",
                #     self.config.mode)
                # Computes histogram of sim_zij
                histogram_sim_zij = plot_histogram(self.sim_zij, buffer=True)
                self.loggers[0].experiment.add_image(
                    'histo_sim_zij', histogram_sim_zij, self.current_epoch)
                
            if self.save_model_weights():
                print('saving model weights')
                dir_to_save = './logs/model_weights_evolution/'
                if not os.path.isdir(dir_to_save):
                    os.mkdir(dir_to_save)
                torch.save({'state_dict': self.state_dict()},
                           dir_to_save + f'model_weights_epoch{self.current_epoch}.pt')

        if self.config.mode in ['classifier', 'regresser']:
            train_auc = self.compute_output_auc(
                self.sample_data.train_dataloader())
            
            # log train auc for tensorboard
            self.loggers[0].experiment.add_scalar(
                "AUC/Train",
                train_auc,
                self.current_epoch)
            
            # log train auc for wandb (if used)
            if self.config.wandb.grid_search:
                self.loggers[1].log_metrics({'AUC/Train': train_auc}, step=self.current_epoch)

            # save train_auc to use it during validation end step
            auc_dict = {'train_auc': train_auc}
            save_path = './' + self.loggers[0].experiment.log_dir + '/train_auc.json'
            with open(save_path, 'w') as file:
                json.dump(auc_dict, file)


        if self.plotting_matrices_now():
            # logs histograms
            self.custom_histogram_adder()
            # Plots views
            self.plot_views()

        # calculates average loss
        avg_loss = torch.stack([x for x in self.training_step_outputs]).mean()

        # logging using tensorboard logger
        self.loggers[0].experiment.add_scalar(
            "Loss/Train",
            avg_loss,
            self.current_epoch)
        
        if self.config.contrastive_model=='BarlowTwins':
            # visu the two loss components on tensorboard
            avg_loss_inv = torch.stack([x for x in self.training_step_loss_inv]).mean()
            avg_loss_redund = torch.stack([x for x in self.training_step_loss_redund]).mean()
            self.loggers[0].experiment.add_scalar(
                "LossInv/Train",
                avg_loss_inv,
                self.current_epoch)
            self.loggers[0].experiment.add_scalar(
                "LossRedund/Train",
                avg_loss_redund,
                self.current_epoch)

        # if multiregion, train loss for each region
        if self.config.multiregion_single_encoder:
            for region in range(self.n_regions):
                regional_loss = [x for x, idx in zip(self.training_step_outputs, self.training_step_idxs_region)
                                if idx==region]
                if len(regional_loss) > 0:
                    regional_loss = torch.stack(regional_loss).mean()
                    self.loggers[0].experiment.add_scalar(
                    f"LossRegion{region}/Train",
                    regional_loss,
                    self.current_epoch)

        if self.config.scheduler:
            self.loggers[0].experiment.add_scalar(
                "Learning rate",
                self.optimizers().param_groups[0]['lr'],
                self.current_epoch)

        # logging using wandb logger (if used)
        if self.config.wandb.grid_search:
            self.loggers[1].log_metrics({'Loss/Train': avg_loss}, 
                                        step=self.current_epoch)
        # if score != 0:
        #     self.loggers[0].experiment.add_scalar(
        #         "Score/Train",
        #         score,
        #         self.current_epoch)
        if self.config.mode == "encoder" and self.config.contrastive_model=='BarlowTwins':
            avg_loss_inv = avg_loss_inv.detach().cpu().item()
            avg_loss_redund = avg_loss_redund.detach().cpu().item()

        self.training_step_outputs.clear()  # free memory
        if self.config.multiple_projection_heads or self.config.multiregion_single_encoder:
            self.training_step_idxs_region.clear()
        if self.config.mode == "encoder" and self.config.contrastive_model=='BarlowTwins':
            self.training_step_loss_inv.clear()
            self.training_step_loss_redund.clear()


    def validation_step(self, val_batch, batch_idx):
        """Validation step"""
        # if self.config.with_labels:
        #     (inputs, _, labels, _) = \
        #         self.get_full_inputs_from_batch_with_labels(val_batch)
        log.info(f"[VAL] mode={self.config.mode} contrastive_model={self.config.contrastive_model}")

        if self.config.with_labels:
            (inputs, filenames, labels, _) = \
                self.get_full_inputs_from_batch_with_labels(val_batch)

        elif self.config.multiple_projection_heads or self.config.multiregion_single_encoder:
            (inputs, _, idx_region) = self.get_full_inputs_from_batch_with_region_idx(val_batch)
        else:
            inputs, filenames = self.get_full_inputs_from_batch(val_batch)
        
        input_i = [inputs[i][:, 0, ...] for i in range(self.n_datasets)]
        input_j = [inputs[i][:, 1, ...] for i in range(self.n_datasets)]
        if self.config.multiple_projection_heads:
            z_i = self.forward(input_i, idx_region=idx_region)
            z_j = self.forward(input_j, idx_region=idx_region)
        else:
            z_i = self.forward(input_i)
            z_j = self.forward(input_j)

        # compute the right loss depending on the learning mode
        if self.config.mode == "decoder":
            sample = inputs[:, 2, :]
            batch_loss = self.cross_entropy_loss(sample, z_i, z_j)
        elif self.config.mode == "classifier":
            batch_loss = self.cross_entropy_loss_classification(
                z_i, z_j, labels)
            batch_label_loss = torch.tensor(0.)
        elif self.config.mode == "regresser":
            batch_loss = self.mse_loss_regression(z_i, z_j, labels)
            batch_label_loss = torch.tensor(0.)
        # elif self.config.proportion_pure_contrastive != 1:
        #     batch_loss, batch_label_loss, _ = \
        #         self.generalized_supervised_nt_xen_loss(z_i, z_j, labels)
        elif self.config.contrastive_model=='SimCLR':
            batch_loss, sim_zij, sim_zii, sim_zjj = self.nt_xen_loss(z_i, z_j)
        elif self.config.contrastive_model=='BarlowTwins':
            batch_loss, loss_invariance, loss_redundancy = self.barlow_twins_loss(z_i,z_j)
        elif self.config.contrastive_model=='VicReg':
            batch_loss = self.vic_reg_loss(z_i,z_j)
        elif self.config.contrastive_model == 'Yaware':
            batch_loss, sim_zij, sim_zii, sim_zjj = self.generalized_supervised_nt_xen_loss(z_i, z_j, labels, step=self.global_step)

        ##################################################################################################################
        elif self.config.contrastive_model == "CLIPTabular":
            loss_i = self.compute_clip_loss(z_i, labels, return_sims=False)
            loss_j = self.compute_clip_loss(z_j, labels, return_sims=False)
            batch_loss = 0.5 * (loss_i + loss_j)



        ####################################################################################################################

        #TODO: add error if None of these names
        
        # values useful for early stoppings
        self.log('val_loss', float(batch_loss), on_epoch=True)
        if self.config.mode in ['classifier', 'regresser']:
            self.log('diff_auc', float(0))
        # logs- a dictionary
        logs = {"val_loss": float(batch_loss)}
        if self.config.contrastive_model=='BarlowTwins':
            logs["val_loss_inv"] = float(loss_invariance)
            logs["val_loss_redund"] = float(loss_redundancy)
        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "val_loss": batch_loss,
            # optional for batch logging purposes
            "log": logs}
        self.validation_step_outputs.append(batch_loss)
        if self.config.contrastive_model=='BarlowTwins':
            # decompose loss in invariance and redundancy term
            self.validation_step_loss_inv.append(loss_invariance)
            self.validation_step_loss_redund.append(loss_redundancy)
        if self.config.multiple_projection_heads or self.config.multiregion_single_encoder:
            self.validation_step_idxs_region.append(idx_region)

        if self.config.with_labels and self.config.mode == 'encoder' \
        and self.config.proportion_pure_contrastive != 1:
            # add label_loss (a part of the loss) to log
            self.log('val_label_loss', float(batch_label_loss))
            logs['val_label_loss'] = float(batch_label_loss)
            batch_dictionary['val_label_loss'] = batch_label_loss

        return batch_dictionary


    def on_validation_epoch_end(self):
        """Computation done at the end of each validation epoch"""

        # score = 0
        # Computes t-SNE
        if self.config.mode in ["encoder", "regresser"]:
            if self.plotting_now():
                log.info("Computing tsne\n")
                X_tsne = self.compute_tsne(
                    self.sample_data.val_dataloader(), "output")
                image_TSNE = plot_tsne(X_tsne, buffer=True)
                self.loggers[0].experiment.add_image(
                    'TSNE output validation image', image_TSNE, self.current_epoch)
                X_tsne = self.compute_tsne(
                    self.sample_data.val_dataloader(),
                    "representation")
                image_TSNE = plot_tsne(X_tsne, buffer=True)
                self.loggers[0].experiment.add_image(
                    'TSNE representation validation image',
                    image_TSNE,
                    self.current_epoch)
        
            # # Plots scatter matrices
            # if self.plotting_matrices_now():
            #     score = self.plot_scatter_matrices_with_labels(
            #         self.sample_data.val_dataloader(),
            #         "val",
            #         self.config.mode)
        
        # compute val auc
        if self.config.mode in ['classifier', 'regresser']:
            val_auc = self.compute_output_auc(
                self.sample_data.val_dataloader())
            # log val auc for tensorboard
            self.loggers[0].experiment.add_scalar(
                "AUC/Val",
                val_auc,
                self.current_epoch)

            # compute overfitting early stopping relevant value
            if self.current_epoch > 0:
                # load train_auc
                save_path = './' + self.loggers[0].experiment.log_dir + '/train_auc.json'
                with open(save_path, 'r') as file:
                    train_auc = json.load(file)['train_auc']
                self.log('diff_auc', float(train_auc - val_auc), on_epoch=True)
            else:
                train_auc = 0.5
            
            # log val auc and grid search criterion for wandb (if used)
            if self.config.wandb.grid_search:
                self.loggers[1].log_metrics({'AUC/Val': val_auc},
                                        step=self.current_epoch)
                if self.current_epoch > 0:
                    gs_crit = compute_grid_search_criterion(
                        train_auc,
                        val_auc,
                        lambda_gs_crit=self.config.wandb.lambda_gs_crit)
                    self.loggers[1].log_metrics({'gs_criterion': gs_crit},
                                                step=self.current_epoch)
                    self.loggers[0].experiment.add_scalar('gs_criterion',
                                                          gs_crit,
                                                          self.current_epoch)
                
            # save the model that has the best val auc during train
            #best_val_auc, best_train_auc = self.save_best_criterion_model(val_auc, train_auc, save_path='./logs/')
            best_val_auc, best_train_auc = self.save_best_auc_model(val_auc, train_auc, save_path='./logs/')

            # log best grid search criterion for wandb (if used)
            if self.config.wandb.grid_search:
                if self.current_epoch > 0:
                    gs_crit_best = compute_grid_search_criterion(
                        best_train_auc,
                        best_val_auc,
                        lambda_gs_crit=self.config.wandb.lambda_gs_crit)
                    self.loggers[1].log_metrics({'gs_criterion_best': gs_crit_best},
                                                step=self.current_epoch)
                    self.loggers[0].experiment.add_scalar('gs_criterion_best',
                                                          gs_crit_best,
                                                          self.current_epoch)

        # calculates average loss
        avg_loss = torch.stack([x for x in self.validation_step_outputs]).mean()

        # logs losses using tensorboard logger
        self.loggers[0].experiment.add_scalar(
            "Loss/Val",
            avg_loss,
            self.current_epoch)
        
        if self.config.contrastive_model=='BarlowTwins':
            # visu the two loss components on tensorboard
            avg_loss_inv = torch.stack([x for x in self.validation_step_loss_inv]).mean()
            avg_loss_redund = torch.stack([x for x in self.validation_step_loss_redund]).mean()
            self.loggers[0].experiment.add_scalar(
                "LossInv/Val",
                avg_loss_inv,
                self.current_epoch)
            self.loggers[0].experiment.add_scalar(
                "LossRedund/Val",
                avg_loss_redund,
                self.current_epoch)
        
        # if multiregion, val loss for each region
        if self.config.multiregion_single_encoder:
            for region in range(self.n_regions):
                regional_loss = [x for x, idx in zip(self.validation_step_outputs, self.validation_step_idxs_region)
                                if idx==region]
                if len(regional_loss) > 0:
                    regional_loss = torch.stack(regional_loss).mean()
                    self.loggers[0].experiment.add_scalar(
                    f"LossRegion{region}/Val",
                    regional_loss,
                    self.current_epoch)

        # use wandb logger (if present)
        if self.config.wandb.grid_search:
            self.loggers[1].log_metrics({'Loss/Val': avg_loss},
                                        step=self.current_epoch)
        # if score != 0:
        #     self.loggers[0].experiment.add_scalar(
        #         "score/Validation",
        #         score,
        #         self.current_epoch)

        # save best model by loss if no auc to do so
        if not self.config.with_labels:
            # save model if best validation loss
            save_path = './logs/'
            if self.current_epoch == 0:
                best_loss = np.inf
            elif self.current_epoch > 0:
                # load the current best loss
                with open(save_path+"best_model_params.json", 'r') as file:
                    best_model_params = json.load(file)
                    best_loss = best_model_params['best_loss']

            # compare to the current loss and replace the best if necessary
            avg_loss = avg_loss.cpu().item()
            if avg_loss < best_loss:
                torch.save({'state_dict': self.state_dict()},
                        save_path+'best_model_weights.pt')
                best_model_params = {
                    'epoch': self.current_epoch, 'best_loss': avg_loss}
                with open(save_path+"best_model_params.json", 'w') as file:
                    json.dump(best_model_params, file)
            if self.config.mode == "encoder" and self.config.contrastive_model=='BarlowTwins':
                avg_loss_inv = avg_loss_inv.detach().cpu().item()
                avg_loss_redund = avg_loss_redund.detach().cpu().item()

        self.validation_step_outputs.clear()  # free memory
        if self.config.multiple_projection_heads or self.config.multiregion_single_encoder:
            self.validation_step_idxs_region.clear()
        if self.config.mode == "encoder" and self.config.contrastive_model=='BarlowTwins':
            self.validation_step_loss_inv.clear()
            self.validation_step_loss_redund.clear()
