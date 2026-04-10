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

from contrastive.utils.config import process_config
from contrastive.data.datamodule import DataModule_Evaluation
from contrastive.evaluation.utils_pipelines import save_used_datasets
from contrastive.models.contrastive_learner_fusion import \
    ContrastiveLearnerFusion


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
def compute_embeddings(config, subsets=None):
    """Compute the embeddings (= output of the backbone(s)) for a given model. 
    It relies on the hydra config framework, especially the backbone, datasets 
    and model parts.
    
    It saves csv files for each subset of the datasets (train, val, test_intra, 
    test) and one with all subjects."""
    
    config = process_config(config)

    config.apply_augmentations = False
    config.with_labels = False

    # create new models in mode visualisation
    data_module = DataModule_Evaluation(config)
    data_module.setup(stage='validate')

    # create a new instance of the current model version,
    # then load hydra weights.
    print("No trained_model.pt saved. Create a new instance and load weights.")
    config.eval_only = True
    config.mode = "embeddings"
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
            state_dict = {
                    k: v for k, v in checkpoint['state_dict'].items()
                    if ('projection_head' not in k) and ('tabular_encoder' not in k)
                }
            for name, param in state_dict.items():
                model_dict[name].copy_(param)

        model.eval()

        # create folder where to save the embeddings
        embeddings_path = config.embeddings_save_path
        if 'use_best_model' in config.keys():
            embeddings_path = config.embeddings_save_path+'_best_model'
        if not os.path.exists(embeddings_path):
            os.makedirs(embeddings_path)

        if config.split is None or config.split=='random' or config.split=='train_test' or config.split=='train_val_test_intra_test' :
            # calculate embeddings for training set and save them somewhere
            if 'train' in subsets or 'train_val' in subsets or 'full' in subsets:
                print("TRAIN SET")
                train_embeddings = model.compute_representations(
                    data_module.train_dataloader())

                # convert the embeddings to pandas df and save them
                train_embeddings_df = embeddings_to_pandas(train_embeddings)
                train_embeddings_df.to_csv(embeddings_path+"/train_embeddings.csv")

            # same thing for validation set
            if 'val' in subsets or 'train_val' in subsets or 'full' in subsets:
                print("VAL SET")
                val_embeddings = model.compute_representations(
                    data_module.val_dataloader())

                val_embeddings_df = embeddings_to_pandas(val_embeddings)
                val_embeddings_df.to_csv(embeddings_path+"/val_embeddings.csv")

            # same thing for test set
            if 'test' in subsets or 'full' in subsets:
                print("TEST SET")
                test_embeddings = model.compute_representations(
                    data_module.test_dataloader())

                test_embeddings_df = embeddings_to_pandas(test_embeddings)
                test_embeddings_df.to_csv(embeddings_path+"/test_embeddings.csv")

            # same thing for test_intra if it exists
            try:
                print("TEST INTRA SET")
                test_intra_embeddings = model.compute_representations(
                    data_module.test_intra_dataloader())

                test_intra_embeddings_df = embeddings_to_pandas(test_intra_embeddings)
                test_intra_embeddings_df.to_csv(
                    embeddings_path+"/test_intra_embeddings.csv")
            except:
                print("No test_intra set")

            # same thing on the train_val dataset
            if 'train_val' in subsets or 'full' in subsets:
                print("TRAIN_VAL SET")
                train_val_df = pd.concat([train_embeddings_df, val_embeddings_df],
                                        axis=0)
                train_val_df.to_csv(embeddings_path+"/train_val_embeddings.csv")

            # same thing on the entire dataset
            print("FULL SET")
            if 'full' in subsets:
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
                full_df.to_csv(embeddings_path+"/full_embeddings.csv")

        print("ALL EMBEDDINGS GENERATED: OK")

        if config.split=='custom':
            print('CUSTOM SPLITS FOR CROSS VAL')
            # calculate embeddings for training set and save them somewhere
            print("TEST SET (UNION OF CUSTOM SPLITS)")
            test_embeddings = model.compute_representations(
                data_module.test_dataloader())
            
            # convert the embeddings to pandas df and save them
            test_embeddings_df = embeddings_to_pandas(test_embeddings)
            test_embeddings_df.to_csv(embeddings_path+"/custom_cross_val_embeddings.csv")

        save_used_datasets(embeddings_path, config.dataset.keys())

    return(valid_path)

if __name__ == "__main__":
    compute_embeddings()
