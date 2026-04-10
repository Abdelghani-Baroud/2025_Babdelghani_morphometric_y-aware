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
""" Training contrastive on skeleton images
"""
######################################################################
# Imports and global variables definitions
######################################################################
import os
# os.environ['MPLCONFIGDIR'] = os.getcwd()+'/.config_mpl'

import hydra
import numpy.random as rd
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb
import omegaconf
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from contrastive.data.datamodule import DataModule_Learning
from contrastive.models.contrastive_learner_fusion import ContrastiveLearnerFusion

from contrastive.utils.config import create_accessible_config, process_config, \
    get_config_diff
from contrastive.utils.logs import set_root_logger_level, \
    set_file_log_handler, set_file_logger

tb_logger = pl_loggers.TensorBoardLogger('logs')
writer = SummaryWriter()

import logging
log = logging.getLogger(__name__)

"""
We use the following definitions:
- embedding or representation, the space before the projection head.
  The elements of the space are features
- output, the space after the projection head.
  The elements are called output vectors
"""


def get_train_seed():
    """Get a random seed for training to avoid collisions when using wandb."""
    train_seed = rd.randint(256)
    return train_seed


@hydra.main(config_name='config', version_base="1.1", config_path="configs")
def train(config):
    config = process_config(config)

    # -------------------------
    # SEED HANDLING (train_seed preferred)
    # -------------------------
    # Priority:
    #   1) config.train_seed if provided (not None)
    #   2) otherwise draw random with get_train_seed()
    # Also mirror it into config.seed for compatibility with existing code/logs.
    train_seed_cfg = getattr(config, "train_seed", None)

    if train_seed_cfg is None:
        train_seed = int(get_train_seed())
        log.info(f"[SEED] train_seed is None/missing -> sampled train_seed={train_seed}")
    else:
        train_seed = int(train_seed_cfg)
        log.info(f"[SEED] using provided train_seed={train_seed}")

    config.train_seed = train_seed
    config.seed = train_seed

    # Apply deterministic seeding (PL seeds python/numpy/torch + dataloader workers)
    pl.seed_everything(train_seed, workers=True)

    # set the number of working cpus
    available_cpus = len(os.sched_getaffinity(0))
    log.debug('Available working cpus:', available_cpus)
    n_cpus = min(available_cpus, config.num_cpu_workers)
    os.environ["NUMEXPR_MAX_THREADS"] = str(n_cpus)
    log.debug('NUMEXPR_MAX_THREADS', n_cpus)

    set_root_logger_level(config.verbose)
    # Sets handler for logger
    set_file_log_handler(file_dir=os.getcwd(),
                         suffix='output')
    log.info("Logger correctly initialized in train()")
    log.debug(f"current directory = {os.getcwd()}")

    # copies some of the config parameters in a yaml file easily accessible
    keys_to_keep = ['datasets', 'nb_subjects', 'model', 'with_labels',
                    'input_size', 'temperature_initial', 'temperature', 'lambda_BT',
                    'sigma', 'drop_rate', 'mode', 'foldlabel',
                    'trimdepth', 'random_choice', 'mixed', 'distribution', 'patch_size', 'max_angle',
                    'max_distance', 'max_translation', 'checkerboard_size',
                    'keep_extremity', 'uniform_trim', 'binary_trim', 'growth_rate',
                    'block_config', 'num_init_features', 'backbone_output_size',
                    'fusioned_latent_space_size', 'num_outputs',
                    'environment', 'batch_size', 'pin_mem', 'partition',
                    'lr', 'gamma', 'weight_decay', 'max_epochs',
                    'early_stopping_patience', 'random_state', 'seed', 'train_seed',
                    'backbone_name', 'sigma_mode', 'standardize_y', 'shrinkage', 'cosine_temperature','contrastive_model',
                    'sigma_labels', 'label_names', 'label_file_name',
                    'proportion_pure_contrastive', 'percentage', 'label_type',
                    'projection_head_name', 'sigma_noise', 'pretrained_model_path',
                    'freeze_encoders', 'converter_activation']

    create_accessible_config(keys_to_keep, os.getcwd() + "/.hydra/config.yaml")

    # create a csv file where the parameters changing between runs are stored
    get_config_diff(os.getcwd() + '/..', whole_config=False, save=True)

    data_module = DataModule_Learning(config)

    model = ContrastiveLearnerFusion(config,
                                     sample_data=data_module)

    # load pretrained model's weights if in config
    if config.pretrained_model_path is not None:
        log.info(f"Load weigths stored at {config.pretrained_model_path}")
        model.load_pretrained_model(config.pretrained_model_path,
                                    encoder_only=config.load_encoder_only,
                                    convolutions_only=config.load_convolutions_only,
                                    freeze_loaded_layers=config.freeze_loaded_layers,
                                    freeze_bias=config.freeze_bias)

    input_size = tuple([1] + list(config.data[0].input_size))
    if config.backbone_name != 'pointnet':
        if (len(config.dataset.keys()) == 1):  # if one region
            print(config.data[0].input_size)
            summary(model, input_data=input_size, batch_dim=0, device=config.device, depth=6)
        else:
            summary(model, device=config.device, depth=6)  # TODO : why 16 ?
    else:
        summary(model, device='cpu')

    # define the early stoppings
    early_stop_callback = \
        EarlyStopping(monitor="val_loss",
                      patience=config.early_stopping_patience)

    early_stop_overfitting = \
        EarlyStopping(monitor="diff_auc",
                      divergence_threshold=config.diff_auc_threshold,
                      patience=config.max_epochs)

    # callbacks = [early_stop_callback]
    # NOTE: original code uses callbacks.append but callbacks is commented out.
    # Keeping the original behavior (no callbacks passed) to avoid changing training logic.
    # If you want callbacks, uncomment the callbacks list above and pass it to Trainer.

    # choose the logger
    loggers = [tb_logger]
    if config.wandb.grid_search:
        # add Wandb logger
        wandb.config = omegaconf.OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True)
        wandb.init(entity=config.wandb.entity, project=config.wandb.project,
                   dir=os.getcwd())
        wandb_logger = pl.loggers.WandbLogger(project=config.wandb.project,
                                              save_dir=os.getcwd())
        loggers.append(wandb_logger)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=config.max_epochs,
        # callbacks=callbacks,
        logger=loggers,
        # flush_logs_every_n_steps=config.nb_steps_per_flush_logs,
        log_every_n_steps=config.log_every_n_steps,
        # auto_lr_find=True
        accumulate_grad_batches=config.accumulate_grad_batches
    )

    # start training
    trainer.fit(model, data_module, ckpt_path=config.checkpoint_path)
    log.info("Fitting is done")

    print(f"End of training for model {os.path.abspath('./')}")


if __name__ == "__main__":
    omegaconf.OmegaConf.register_new_resolver("get_train_seed", get_train_seed)
    train()
