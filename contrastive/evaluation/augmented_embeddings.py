"""

The way this function works is that it creates a SimCLR structure
based on the provided config,
then loads the weights of the target model (in the config).

Once this is done, generate the embeddings of the target dataset
(in the config) in inference mode.

This generation methods is highly dependent of the parameters config,
so I suggest either to run it right
after the training is complete,
or to use evaluation/embeddings_pipeline.py to generate the embeddings
(it handles the needed modifications in order to load the right model).

This method is also relying on the current DataModule
and ContrastiveLearner implementations, which means
its retro compatibility leaves a lot to be desired.


"""


import hydra
import torch
import pandas as pd
import os
import glob
import yaml
import json
import omegaconf
from tqdm import tqdm

from contrastive.utils.config import process_config
from contrastive.data.datamodule import DataModule_Evaluation
from contrastive.evaluation.utils_pipelines import save_used_datasets
from contrastive.models.contrastive_learner_fusion import \
    ContrastiveLearnerFusion
from utils_pipelines import get_save_folder_name, change_config_datasets,\
                            change_config_label, change_config_dataset_localization

def preprocess_config(sub_dir, dataset_localization, datasets, idx_region_evaluation,
                      folder_name, epoch, verbose=False):
    """Loads the associated config of the given model and changes what has to be done,
    mainly the datasets, the classifier type and a few other keywords.
    
    Arguments:
        - sub_dir: str. Path to the directory containing the saved model.
        - datasets: list of str. List of the datasets to be used for the results generation.
        - label: str. Name of the label to be used for evaluation.
        - folder_name: str. Name of the directory where to store both embeddings and aucs.
        - classifier_name: str. Should correspond to a classifier yaml file's name 
        (currently either 'svm' or 'neural_network').
        - epoch: int. Specifies the epoch used for inference. Set to None to use the last epoch.
        - verbose: bool. Verbose.
        
    Output:
        - cfg: the config as an omegaconf object."""

    if verbose:
        print(os.getcwd())
    cfg = omegaconf.OmegaConf.load(sub_dir+'/.hydra/config.yaml')

    # replace the datasets
    change_config_datasets(cfg, datasets)
    # replace the dataset localizatyion
    change_config_dataset_localization(cfg, dataset_localization)

    # replace the possibly incorrect config parameters
    cfg.model_path = sub_dir
    cfg.embeddings_save_path = \
        sub_dir + f"/{folder_name}_augmented_embeddings"
    # TODO: modify augmentation policy here !!
    cfg.multiregion_single_encoder = False
    cfg.load_sparse = False

    # add epoch to config if specified
    if epoch is not None:
        cfg.epoch = epoch

    # in multi head case
    if idx_region_evaluation is not None:
        cfg.idx_region_evaluation=idx_region_evaluation

    # change config partition to avoid errors
    cfg.partition = [0.9,0.1]

    return cfg


def embeddings_to_pandas(embeddings, csv_path=None, verbose=False):
    """Homogenize column names and saves to pandas.

    Args:
        embeddings: Output of the compute_representations function
        csv_path: Path where to save the csv.
                  Set to None if you want to return the df
    """
    columns_names = ['dim'+str(i+1) for i in range(embeddings[0].shape[1])]
    values = pd.DataFrame(embeddings[0].numpy(), columns=columns_names)
    filenames = embeddings[1]
    filenames = pd.DataFrame(filenames, columns=['ID'])
    df_embeddings = pd.concat([filenames, values], axis=1)

    # remove one copy each ID
    df_embeddings = \
        df_embeddings.groupby('ID').mean()

    if verbose:
        print("embeddings:", df_embeddings.iloc[:10, :])
        print("nb of elements:", df_embeddings.shape[0])

    # Solves the case in which index type is tensor
    if len(df_embeddings.index) > 0:  # avoid cases where empty df
        if df_embeddings.index.dtype==int:
            df_embeddings.index = df_embeddings.index.astype(str)
        if type(df_embeddings.index[0]) != str:
            index = [idx.item() for idx in df_embeddings.index]
            index_name = df_embeddings.index.name
            df_embeddings.index = index
            df_embeddings.index.names = [index_name]

    if csv_path:
        df_embeddings.to_csv(csv_path)
    else:
        return df_embeddings


#@hydra.main(config_name='config_no_save', config_path="../configs")
def compute_augmented_embeddings(config, nb_iterations):
    """Compute the embeddings (= output of the backbone(s)) for a given model. 
    It relies on the hydra config framework, especially the backbone, datasets 
    and model parts.
    
    It saves csv files for each subset of the datasets (train, val, test_intra, 
    test) and one with all subjects."""
    
    config = process_config(config)

    # TODO: right augmentation config here ?
    #config.apply_augmentations = False
    config.with_labels = False

    # create new models in mode visualisation
    data_module = DataModule_Evaluation(config)
    data_module.setup(stage='validate')

    # create a new instance of the current model version,
    # then load hydra weights.
    print("No trained_model.pt saved. Create a new instance and load weights.")

    model = ContrastiveLearnerFusion(config, sample_data=data_module)
    # fetch and load weights
    if 'epoch' in config.keys():
        ckpt_path = config.model_path+\
                    f"/logs/model_weights_evolution/model_weights_epoch{config.epoch}.pt"
        #assert os.path.isfile(ckpt_path), f"No weights for selected epoch {config.epoch}"
        valid_path = os.path.isfile(ckpt_path) # check if weights exist for selected epoch
    else:
        paths = config.model_path+"/*logs/*/version_0/checkpoints"+r'/*.ckpt'
        if 'use_best_model' in config.keys():
            paths = config.model_path+"/logs/best_model_weights.pt"
        files = glob.glob(paths)
        #print("model_weights:", files[0])
        ckpt_path = files[0]
        valid_path=True

    if not valid_path:
        pass
    else:
        print(f"weights loaded from: {ckpt_path}")
        checkpoint = torch.load(
            ckpt_path, map_location=torch.device(config.device))
        
        # TODO : load projection head only if linear in projection head
        # otherwise load backbone only

        # remove keys not matching (when multiple projection heads, select one).
        if 'idx_region_evaluation' in config.keys():
            model_dict = model.state_dict()
            state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'projection_head' not in k or f'projection_head.{config.idx_region_evaluation}' in k}
            # rename projection head keys : remove projection head idx
            new_keys_list = []
            old_keys_list = []
            for k, v in state_dict.items():
                if 'projection_head' in k:
                    old_keys_list.append(k)
                    l = k.split('.')
                    l[1]=str(0)
                    k = '.'.join(l)
                    new_keys_list.append(k)
            for newk, oldk in zip(new_keys_list, old_keys_list):
                state_dict[newk] = state_dict.pop(oldk)
            model_dict.update(state_dict) 
            model.load_state_dict(state_dict)
        #else:
        #    model.load_state_dict(checkpoint['state_dict'])
        ## Do not load projection head
        else:
            model_dict = model.state_dict()
            state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'projection_head' not in k}
            for name, param in state_dict.items():
                model_dict[name].copy_(param)

        model.eval()

        # create folder where to save the embeddings
        embeddings_path = config.embeddings_save_path
        if not os.path.exists(embeddings_path):
            os.makedirs(embeddings_path)

        print(f'Generating {nb_iterations} augmented embeddings for each subject')
        for iter in tqdm(range(nb_iterations)):
            # calculate embeddings for training set
            train_embeddings = model.compute_representations(
                data_module.train_dataloader())
            train_embeddings_df = embeddings_to_pandas(train_embeddings)
            val_embeddings = model.compute_representations(
                data_module.val_dataloader())
            val_embeddings_df = embeddings_to_pandas(val_embeddings)
            test_embeddings = model.compute_representations(
                data_module.test_dataloader())
            test_embeddings_df = embeddings_to_pandas(test_embeddings)
            # same thing for test_intra if it exists
            try:
                test_intra_embeddings = model.compute_representations(
                    data_module.test_intra_dataloader())
                test_intra_embeddings_df = embeddings_to_pandas(test_intra_embeddings)
            except:
                pass
            # same thing on the entire dataset
            try:
                full_df = pd.concat([train_embeddings_df,
                                    val_embeddings_df,
                                    test_intra_embeddings_df,
                                    test_embeddings_df],
                                    axis=0)
            except:
                full_df = pd.concat([train_embeddings_df,
                                    val_embeddings_df,
                                    test_embeddings_df],
                                    axis=0)

            full_df = full_df.sort_values(by='ID')
            full_df.to_csv(embeddings_path+f"/full_embeddings_{iter}.csv")

        print("ALL EMBEDDINGS GENERATED: OK")
        save_used_datasets(embeddings_path, config.dataset.keys())

    return(valid_path)


def augmented_embeddings(dir_path, dataset_localization, datasets, nb_iterations,
                         idx_region_evaluation, short_name, overwrite, epochs, verbose=False):

        # walks recursively through the subfolders
    for name in os.listdir(dir_path):
        sub_dir = dir_path + '/' + name
        # checks if directory
        if os.path.isdir(sub_dir):
            # check if directory associated to a model
            if os.path.exists(sub_dir+'/.hydra/config.yaml'):
                print("\nTreating", sub_dir)

                # check if embeddings and ROC already computed
                # if already computed and don't want to overwrite, then pass
                # else apply the normal process
                folder_name = get_save_folder_name(datasets=datasets, short_name=short_name)
                if (
                    os.path.exists(sub_dir + f"/{folder_name}_augmented_embeddings")
                    and (not overwrite)
                ):
                    print("Model already treated "
                          "(existing folder with embeddings). "
                          "Set overwrite to True if you still want "
                          "to compute them.")

                elif '#' in sub_dir:
                    print(
                        "Model with an incompatible structure "
                        "with the current one. Pass.")

                else:
                    print("Start post processing")
                    # get the config and correct it
                    for epoch in epochs:
                        if epoch is not None:
                            f_name = folder_name + f'_epoch{epoch}'
                        else:
                            f_name = folder_name
                        cfg = preprocess_config(sub_dir,
                                                dataset_localization=dataset_localization,
                                                datasets=datasets,
                                                idx_region_evaluation=idx_region_evaluation,
                                                folder_name=f_name,
                                                epoch=epoch,
                                                verbose=verbose)
                        if verbose:
                            print("CONFIG FILE", type(cfg))
                            print(json.dumps(omegaconf.OmegaConf.to_container(
                                cfg, resolve=True), indent=4, sort_keys=True))
                        # save the modified config next to the real one
                        with open(sub_dir+'/.hydra/config_augmented_embeddings.yaml', 'w') \
                                as file:
                            yaml.dump(omegaconf.OmegaConf.to_yaml(cfg), file)
                        valid_path = compute_augmented_embeddings(cfg, nb_iterations)
                        if not valid_path:
                            print('Invalid epoch number, skipped')

if __name__ == "__main__":

    """
    augmented_embeddings("/neurospin/dico/jlaval/Output/test_augmented_embeddings",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/orbital_left_hcp_custom"],
                        nb_iterations=50,
                        idx_region_evaluation=None,
                        short_name='troiani',
                        overwrite=True,
                        epochs=[None],
                        verbose=False)
    """
    augmented_embeddings("/neurospin/dico/jlaval/Output/test_augmented_embeddings_cing",
                    dataset_localization="neurospin",
                    datasets=["julien/MICCAI_2024/training/cingulate_40k_right"],
                    nb_iterations=50,
                    idx_region_evaluation=None,
                    short_name='ukb',
                    overwrite=True,
                    epochs=[None],
                    verbose=False)
    # TODO: create an eval augmentation policy ?