import numpy as np
import pandas as pd
import os
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

#### ARGUMENTS
dims = [f'dim{k}' for k in range(1,33)]
#emb_dir = '/neurospin/dico/jlaval/Output/test_augmented_embeddings/09-33-31_0/troiani_augmented_embeddings'
emb_dir = '/neurospin/dico/jlaval/Output/test_augmented_embeddings_cing/11-22-27_0/ukb_augmented_embeddings'
n_cpus= 30
####

def process_element(sub_embs):

    dists = pairwise_distances(sub_embs, metric='cosine')
    dists = dists[~np.eye(dists.shape[0], dtype=bool)]
    mean = np.mean(dists)
    median = np.median(dists)
    quant = np.quantile(dists, 0.95)
    maxi = np.max(dists)
    std = np.std(dists)

    return([mean, median, quant, maxi, std])


# Function to process the list using multiprocessing with n CPUs and additional arguments
def process_list(list_sub_embs, n_cpus):
    # Create a partial function with y and z fixed
    partial_func = partial(process_element)
    
    # Create a pool of workers with n_cpus
    with mp.Pool(processes=n_cpus) as pool:
        # Map the partial function to the elements list
        results = list(tqdm(pool.imap(partial_func, list_sub_embs), total=len(list_sub_embs)))
    return results


embs_dir = os.listdir(emb_dir)
embs_dir = [direc for direc in embs_dir if 'embeddings' in direc]
embs_list = []
for file in embs_dir:
    embs = pd.read_csv(os.path.join(emb_dir, file))
    embs_list.append(embs)
full_embs = pd.concat(embs_list)
subjects = embs['ID'].tolist()
nb_subs = len(subjects)

# get an array of embs for each subject
# not parallel because too heavy
list_sub_embs = []
print('Get individual embs')
for k, subject in enumerate(tqdm((subjects))):
    sub_embs = full_embs[k::nb_subs][dims].to_numpy()
    list_sub_embs.append(sub_embs)

print('Compute stats in parallel')
stats = process_list(list_sub_embs, n_cpus=n_cpus)
df_augmentation_stats = pd.DataFrame(data = np.array(stats),
                                     columns=['Mean', 'Median', 'Q95', 'Max', 'Std'],
                                     index=subjects)

print('Saving DataFrame')
df_augmentation_stats.to_csv(os.path.join(emb_dir, 'summary_stats.csv'), index_label='Subject')