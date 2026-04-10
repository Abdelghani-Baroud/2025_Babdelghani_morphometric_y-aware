import hydra
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import json
import os

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import auc, roc_curve, roc_auc_score, balanced_accuracy_score, \
                            mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_predict, train_test_split, cross_validate, \
                                    LeaveOneGroupOut, cross_val_score, GridSearchCV
from scipy.stats import pearsonr

from pqdm.processes import pqdm
from joblib import cpu_count
from functools import partial

from sklearn.preprocessing import StandardScaler
# from contrastive.models.binary_classifier import BinaryClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor

from contrastive.data.utils import read_labels

from contrastive.utils.config import process_config
from contrastive.utils.logs import set_root_logger_level, set_file_logger
from contrastive.evaluation.utils_pipelines import save_used_label
from contrastive.evaluation.auc_score import regression_roc_auc_score

from sklearn.utils._testing import ignore_warnings
from sklearn.metrics import r2_score
from sklearn.exceptions import ConvergenceWarning


_parallel = False

log = set_file_logger(__file__)


def define_njobs():
    """Returns number of cpus used by main loop
    """
    nb_cpus = cpu_count()
    return max(nb_cpus - 2, 1)


def load_embeddings(dir_path, labels_path, config, subset='full'):
    """Load the embeddings and the labels.

    Arguments:
        - dir_path: path where the embeddings are stored. Either 
        the folder that contains them or directly the target file.
        - labels_path: the file where the labels are stored.
        - config: the omegaconf object related to the current ANN model.
        - subset: str. Target subset of the data the classifiers will be trained 
        on. Usually either 'train', 'val', 'train_val', 'test' or 'test_intra'.
    """
    # load embeddings
    if config.split is not None and config.split=='custom':
        embeddings = pd.read_csv(
                dir_path+f'/custom_cross_val_embeddings.csv', index_col=0)
    elif config.split is None or config.split=='random' or config.split=='train_test':
        # if targeting directly the target csv file
        if not os.path.isdir(dir_path):
            embeddings = pd.read_csv(dir_path, index_col=0)
        # if only giving the directory (implies constraints on the file name)
        # take only a specified subset
        elif subset != 'full':
            embeddings = pd.read_csv(
                    dir_path+f'/{subset}_embeddings.csv', index_col=0)
        # takes all the subjects
        else:
            if os.path.exists(dir_path+'/full_embeddings.csv'):
                embeddings = pd.read_csv(
                    dir_path+'/full_embeddings.csv', index_col=0)
            elif os.path.exists(dir_path+'/pca_embeddings.csv'):
                embeddings = pd.read_csv(
                    dir_path+'/pca_embeddings.csv', index_col=0)
            else:
                train_embeddings = pd.read_csv(
                    dir_path+'/train_embeddings.csv', index_col=0)
                val_embeddings = pd.read_csv(
                    dir_path+'/val_embeddings.csv', index_col=0)
                test_embeddings = pd.read_csv(
                    dir_path+'/test_embeddings.csv', index_col=0)
                embs_list = [train_embeddings, val_embeddings,
                            test_embeddings]
                try:
                    test_intra_embeddings = pd.read_csv(
                        dir_path+'/test_intra_embeddings.csv', index_col=0)
                    embs_list.append(test_intra_embeddings)
                except:
                    pass
                    
                # regroup them in one dataframe
                embeddings = pd.concat(embs_list, axis=0, ignore_index=False)
    else:
        raise ValueError("Wrong split config specified")

    embeddings.sort_index(inplace=True)
    log.debug(f"sorted embeddings: {embeddings.head()}")

    # get the labels (0 = no paracingulate, 1 = paracingulate)
    # and match them to the embeddings
    # /!\ use read_labels
    label_scaling = (None if 'label_scaling' not in config.keys()
                     else config.label_scaling)
    labels = read_labels(labels_path, config.data[0].subject_column_name,
                         config.label_names, label_scaling)
    labels.rename(columns={config.label_names[0]: 'label'}, inplace=True)
    labels = labels[labels.Subject.isin(embeddings.index)]
    # for unique IDs, add NaNs to labels when in embeddings only
    #missing_labels = set(embeddings.index.tolist()).difference(set(labels.index.tolist()))
    #missing_labels = 
    labels.sort_values(by='Subject', inplace=True, ignore_index=True)
    log.debug(f"sorted labels: {labels.head()}")

    embeddings = embeddings[embeddings.index.isin(labels.Subject)]
    embeddings.sort_index(inplace=True)
    #drop duplicates here in case
    embeddings = embeddings.reset_index().drop_duplicates(subset='ID', keep='first').set_index('ID')
    labels = labels.drop_duplicates(subset='Subject', keep='first')
    if not embeddings.reset_index().ID.equals(labels.Subject):
        raise ValueError("Embeddings and labels do not have the same list of subjects")
    log.debug(f"sorted embeddings: {embeddings.head()}")

    # /!\ multiple labels is not handled

    return embeddings, labels


def compute_binary_indicators(Y, proba_pred):
    """Compute ROC curve + AUC, and best balanced accuracy (by threshold search)."""

    # ---- get labels_true as numeric 0/1 ----
    if isinstance(Y, torch.Tensor):
        labels_true = Y.detach().cpu().numpy().astype("float64")
    else:
        # pandas Series / numpy array / list
        y = Y if hasattr(Y, "values") else pd.Series(Y)

        # normalize strings (critical)
        if pd.api.types.is_string_dtype(y) or y.dtype == object:
            y_str = y.astype(str).str.strip().str.lower()

            mapping = {
                "absent": 0.0,
                "present": 1.0,
                "no": 0.0,
                "yes": 1.0,
                "false": 0.0,
                "true": 1.0,
                "0": 0.0,
                "1": 1.0,
            }
            y = y_str.map(mapping)

        # force numeric
        y = pd.to_numeric(y, errors="coerce")

        if y.isna().any():
            bad = (Y[y.isna()].dropna().unique()[:10]
                   if hasattr(Y, "__getitem__") else [])
            raise ValueError(
                f"compute_binary_indicators: non-numeric labels after mapping. Examples: {bad}"
            )

        labels_true = y.values.astype("float64")

    # ensure binary
    uniq = np.unique(labels_true)
    if not np.all(np.isin(uniq, [0.0, 1.0])):
        raise ValueError(f"compute_binary_indicators: labels must be binary 0/1, got unique={uniq}")

    # ---- metrics ----
    curves = roc_curve(labels_true, proba_pred[:, 1])
    roc_auc = roc_auc_score(labels_true, proba_pred[:, 1])

    # ---- best balanced accuracy by threshold ----
    max_accuracy = -1.0
    best_labels_pred = None

    for threshold in np.linspace(0, 1, 101):
        labels_pred_0 = (proba_pred[:, 0] < threshold)
        labels_pred_1 = (proba_pred[:, 0] >= threshold)

        accuracy_0 = balanced_accuracy_score(labels_true, labels_pred_0)
        accuracy_1 = balanced_accuracy_score(labels_true, labels_pred_1)

        if accuracy_0 >= accuracy_1:
            accuracy = accuracy_0
            labels_pred = labels_pred_0
        else:
            accuracy = accuracy_1
            labels_pred = labels_pred_1

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_labels_pred = labels_pred

    return curves, roc_auc, max_accuracy, best_labels_pred.astype(int)

def to_numeric_binary(y):
    """
    Convert labels to numeric 0/1.
    Accepts pandas Series / array / list / torch tensor.
    Handles 'present'/'absent' with casing/spaces.
    """
    if isinstance(y, torch.Tensor):
        arr = y.detach().cpu().numpy()
        return arr.astype("float64")

    s = y if isinstance(y, pd.Series) else pd.Series(y)

    # normalize strings
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        s = s.astype(str).str.strip().str.lower()
        mapping = {
            "absent": 0.0, "present": 1.0,
            "no": 0.0, "yes": 1.0,
            "false": 0.0, "true": 1.0,
            "0": 0.0, "1": 1.0,
        }
        s = s.map(mapping)

    s = pd.to_numeric(s, errors="coerce")

    if s.isna().any():
        bad = s[s.isna()]
        raise ValueError(
            f"Non-numeric labels after mapping. Examples: "
            f"{pd.Series(y).astype(str)[bad.index].unique()[:20]}"
        )

    arr = s.values.astype("float64")
    uniq = np.unique(arr)
    if not np.all(np.isin(uniq, [0.0, 1.0])):
        raise ValueError(f"Labels must be binary 0/1, got unique={uniq}")
    return arr


def compute_multiclass_indicators(Y, proba_pred):
    """Compute ROC auc and accuracy for multiclass label"""
    if type(Y) == torch.tensor:
        labels_true = Y.detach_().numpy()
    else:
        labels_true = Y.values.astype('float64')

    # TODO: add metrics, return list of values, len = number of labels
    roc_aucs = roc_auc_score(Y, proba_pred, multi_class='ovr', average=None)
    max_accuracies = []
    #for k in range(proba_pred.shape[1]):
    #    max_accuracy = 0
    #    for threshold in np.linspace(0,1,101):
    #        labels_pred_0 = proba_pred[:, k] < threshold
    #        labels_pred_1 = proba_pred[:, k] >= threshold
    #        labels_true_binarized = labels_true==k
    #        accuracy_0 = balanced_accuracy_score(labels_true_binarized, labels_pred_0)
    #        accuracy_1 = balanced_accuracy_score(labels_true_binarized, labels_pred_1)
    #        accuracy = max(accuracy_0, accuracy_1)
    #        if accuracy > max_accuracy:
    #            max_accuracy = accuracy
    #    max_accuracies.append(max_accuracy)
    max_accuracies=[0,0,0,0]
    return roc_aucs, max_accuracies


def compute_auc(column, label_col=None):
    log.debug("COMPUTE AUC")
    log.debug(label_col.head())
    log.debug(column.head())
    return roc_auc_score(label_col, column)


def get_average_model(labels_df):
    """Get a model with performance that is representative of the group, 
    i.e. the one with the median auc."""
    aucs = labels_df.apply(compute_auc, args=[labels_df.label])
    aucs = aucs[aucs.index != 'label']
    aucs = aucs[aucs == aucs.quantile(interpolation='nearest')]
    return (aucs.index[0])


def post_processing_results(cv_scores, cv_stds, splits_scores, labels, embeddings, Curves, aucs, accuracies,
                            values, columns_names, mode, subset, results_save_path):
    """Get the mean and the median AUC and accuracy, plot the ROC curves and 
    the generated files."""

    # labels_true = labels.label.values.astype('float64')
    labels_true = to_numeric_binary(labels["label"])


    # compute agregated models
    predicted_labels = labels[columns_names]

    labels['median_pred'] = predicted_labels.median(axis=1)
    labels['mean_pred'] = predicted_labels.mean(axis=1)

    # plot ROC curves
    plt.figure()

    # ROC curves of all models
    for curves in Curves[mode]:
        plt.plot(curves[0], curves[1], color='grey', alpha=0.1)
    plt.plot([0, 1], [0, 1], color='r', linestyle='dashed')

    # get the average model (with AUC as a criteria)
    # /!\ This model is a classifier that exists in the pool
    # /!\ This model != 'mean_pred' or 'median_pred'
    average_model = get_average_model(
        labels[['label'] + columns_names].astype('float64'))
    labels['average_model'] = labels[average_model]
    roc_curve_average = roc_curve(labels_true, labels[average_model].values)
    # ROC curves of "special" models
    roc_curve_median = roc_curve(labels_true, labels.median_pred.values)
    roc_curve_mean = roc_curve(labels_true, labels.mean_pred.values)

    plt.plot(roc_curve_average[0], roc_curve_average[1],
             color='red', alpha=0.5, label='average model')
    plt.plot(roc_curve_median[0], roc_curve_median[1],
             color='blue', label='agregated model (median)')
    plt.plot(roc_curve_mean[0], roc_curve_mean[1],
             color='black', label='agregated model (mean)')
    plt.legend()
    plt.title(f"{subset} ROC curves")
    plt.savefig(results_save_path+f"/{subset}_ROC_curves.png")

    # compute accuracy and area under the curve
    print(f"{subset} cross_val accuracy",
          np.mean(accuracies[mode]),
          np.std(accuracies[mode]))
    print(f"{subset} cross_val AUC", np.mean(aucs[mode]), np.std(aucs[mode]))

    values[f'{subset}_total_balanced_accuracy'] = \
        [np.mean(accuracies[mode]), np.std(accuracies[mode])]
    values[f'{subset}_auc'] = [np.mean(aucs[mode]), np.std(aucs[mode])]
    values['cv_score'] = np.mean(cv_scores)
    values['cv_std'] = np.mean(cv_stds) # TODO : doesn't make sens to take mean here, to apply only one there is one iteration
    values['splits_score']=splits_scores[0]

    # save predicted labels
    labels.to_csv(results_save_path+f"/{subset}_predicted_probas.csv",
                  index=False)
    # DEBUG embeddings.to_csv(results_save_path+f"/effective_embeddings.csv",
    #                         index=True)


def train_one_classifier(config, inputs, subjects, i=0):
    """Trains one classifier, whose type is set in config_no_save.

    Args:
        - config: config file
        - inputs: dictionary containing the input data,
        with X key containing embeddings
        and Y key labels. If a test set is defined,
        it also contains X and Y for the test set.
        - i: seed for the SVM.
        Is automatically changed in each call of train_svm_classifiers.
    """

    X = inputs['X']
    Y = inputs['Y']
    outputs = {}

    # cv stratification
    if config.split is None or config.split=='train_test':
        pass
    elif config.split=='random':
        cv=config.cv
    elif config.split=='custom':
        if 'splits_basedir' not in config.keys():
            raise ValueError("A custom split should be specified for custom CV")
        else:
            root_dir = '/'.join(config.splits_basedir.split('/')[:-1])
            basedir = config.splits_basedir.split('/')[-1]
            splits_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir) if f.startswith(basedir) and '.csv' in f]
            splits_subs = [pd.read_csv(file, header=None) for file in splits_dirs]
            labels = np.concatenate([[i] * len(K) for i, K in enumerate(splits_subs)])
            splits_subs_and_labels = pd.concat(splits_subs)
            splits_subs_and_labels.columns=['ID']
            splits_subs_and_labels['labels'] = labels
            subs_embeddings = pd.DataFrame({'ID': subjects, 'X': list(X.values), 'Y': Y})
            df = subs_embeddings.merge(splits_subs_and_labels, on='ID')
            groups, X, Y = df['labels'], np.vstack(df['X'].values), df['Y']
            logo = LeaveOneGroupOut()
            cv = [*(logo.split(X, Y, groups=groups))]
    else:
        raise ValueError("Wrong split config specified")

    # train and predict
    # regression
    if 'label_type' in config.keys() and config['label_type']=='continuous':
        if config.classifier_name == 'logistic': # TODO: change the parameter name for Elastic
            #model = LinearRegression() # TODO: ElasticNet instead + add gridsearch  
            parameters={'l1_ratio': np.linspace(0,1,11), 'alpha': [10**k for k in range(-3,4)]}   
            model = ElasticNet(max_iter=100000) # TODO : change the number of iterations ?
        
        elif config.classifier_name == 'knn':
             parameters = {
                'n_neighbors': [5, 10, 20, 30, 40,50,60,70,100],
                'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                'weights': ['uniform', 'distance'],
                'leaf_size': [1,2,3,5,10, 20, 30],
                'metric': ['euclidean', 'manhattan','chebyshev', 'cosine']
              }
             model = KNeighborsRegressor() 
        else:
            model = SVR(kernel='linear',max_iter=config.class_max_epochs,
                        C=0.01)
            
        # Initialize GridSearch  
        clf = GridSearchCV(model, parameters, cv=cv, scoring='r2', refit=True, n_jobs=define_njobs())
        print(f'Parameters for GridSearch: {parameters}')

        if config.split is None:
            model.fit(X,Y)
            val_pred = model.predict(X)
        elif config.split=='train_test':
            train = pd.read_csv(os.path.join(config.embeddings_save_path, 'train_embeddings.csv'), usecols=['ID'])
            test = pd.read_csv(os.path.join(config.embeddings_save_path, 'test_embeddings.csv'), usecols=['ID'])
            subs_embeddings = pd.DataFrame({'ID': subjects, 'X': list(X.values), 'Y': Y})
            subs_embeddings_train = subs_embeddings.merge(train, on='ID')
            X_train, Y_train = np.vstack(subs_embeddings_train['X'].to_numpy()), subs_embeddings_train['Y']
            subs_embeddings_test = subs_embeddings.merge(test, on='ID')
            X_test, Y_test = np.vstack(subs_embeddings_test['X'].to_numpy()), subs_embeddings_test['Y']
            #model.fit(X_train, Y_train)
            #val_pred = model.predict(X_test)
            #X,Y = X_test, Y_test
            clf.fit(X_train, Y_train)
            print(f'best params : {clf.best_params_}')
            print(f'best score : {clf.best_score_}')
            md = clf.best_estimator_
            labels_proba = md.predict_proba(X_test)
            X,Y = X_test, Y_test
        else:
            clf.fit(X,Y)
            print(f'best params : {clf.best_params_}')
            print(f'best score : {clf.best_score_}')
            md = clf.best_estimator_
            val_pred = cross_val_predict(md, X, Y, cv=cv)
            #val_pred = cross_val_predict(model, X, Y, cv=cv)
        print(f'True label mean: {np.mean(Y):.3f}, std: {np.std(Y):.3f}')
        print(f'Predicted label mean: {np.mean(val_pred):.3f}, std: {np.std(val_pred):.3f}')
        r2 = r2_score(Y, val_pred)
        mse = mean_squared_error(Y, val_pred)
        mae = mean_absolute_error(Y, val_pred)
        reg_auc = regression_roc_auc_score(Y, val_pred, num_rounds=1000) # ignore this score
        pred_vs_true = np.vstack((Y,val_pred)).T
        outputs['pred_vs_true'] = pred_vs_true
        outputs['MSE'] = mse
        outputs['MAE'] = mae
        outputs['r2'] = r2
        outputs['reg_auc'] = reg_auc

    # classification
    else:
        # choose the classifier type
        # /!\ The chosen classifier must have a predict_proba method.
        if config.classifier_name == 'svm':
            parameters={'C': [10**k for k in range(-3,4)]}
            model = SVC(kernel='linear', probability=True,
                        max_iter=config.class_max_epochs, random_state=i,
                        class_weight='balanced', decision_function_shape='ovr')
        elif config.classifier_name == 'neural_network': # DEPRECATED ?
            model = MLPClassifier(hidden_layer_sizes=config.classifier_hidden_layers,
                                activation=config.classifier_activation,
                                batch_size=config.class_batch_size,
                                max_iter=config.class_max_epochs, random_state=i)
        elif config.classifier_name == 'logistic':
            #model = LogisticRegression(C=0.01, solver='liblinear', penalty='l2',
            #                           max_iter=config.class_max_epochs,
            #                           random_state=i)
            parameters={'l1_ratio': np.linspace(0,1,11), 'C': [10**k for k in range(-3,4)]}
            model = LogisticRegression(solver='saga', penalty='elasticnet',
                                       max_iter=1000000, random_state=i)
            #parameters={'n_neighbors': [30,40,50,60,70], 'weights': ['distance'], 'leaf_size': [1,2], 'metric': ['chebyshev', 'cosine']}
            #model = KNeighborsClassifier() # LogisticRegression is better
        elif config.classifier_name == 'knn':
             parameters = {
                'n_neighbors': [5, 10, 20, 30, 40,50,60,70,100],
                'weights': ['uniform', 'distance'],
                'leaf_size': [1,2,3,5,10, 20, 30],
                'metric': ['euclidean', 'manhattan','chebyshev', 'cosine']
              }
             model = KNeighborsClassifier()    
        else:
            raise ValueError(f"The chosen classifier ({config.classifier_name}) is not handled by the pipeline. \
                               Choose a classifier type that exists in configs/classifier.")
        
        # Initialize GridSearch
        clf = GridSearchCV(model, parameters, cv=cv, scoring='roc_auc_ovr_weighted', refit=True, n_jobs=define_njobs()) # TODO : GridSearch won't be good for preterms
        print(f'Parameters for GridSearch: {parameters}')

        # create function to avoid copy paste ?
        if config.split is not None and config.split=='train_test':
            train = pd.read_csv(os.path.join(config.embeddings_save_path, 'train_embeddings.csv'), usecols=['ID'])
            test = pd.read_csv(os.path.join(config.embeddings_save_path, 'test_embeddings.csv'), usecols=['ID'])
            subs_embeddings = pd.DataFrame({'ID': subjects, 'X': list(X.values), 'Y': Y})
            subs_embeddings_train = subs_embeddings.merge(train, on='ID')
            X_train, Y_train = np.vstack(subs_embeddings_train['X'].to_numpy()), subs_embeddings_train['Y']
            subs_embeddings_test = subs_embeddings.merge(test, on='ID')
            X_test, Y_test = np.vstack(subs_embeddings_test['X'].to_numpy()), subs_embeddings_test['Y']
            clf.fit(X_train, Y_train)
            print(f'best params : {clf.best_params_}')
            print(f'best score : {clf.best_score_}')
            md = clf.best_estimator_
            labels_proba = md.predict_proba(X_test)
            X,Y = X_test, Y_test
        elif config.split is not None and (config.split=='random' or config.split=='custom'):
            #labels_proba = cross_val_predict(model, X, Y, cv=cv, method='predict_proba')
            clf.fit(X,Y)
            print(f'best params : {clf.best_params_}')
            print(f'best score : {clf.best_score_}')
            md = clf.best_estimator_
            labels_proba = cross_val_predict(md, X, Y, cv=cv, method='predict_proba') # why does it give slightly diff results ??
        else: # no split, simply use all data to fit # DON'T USE IT
            clf.fit(X,Y)
            print(f'best params : {clf.best_params_}')
            print(f'best score : {clf.best_score_}')
            md = clf.best_estimator_
            labels_proba = md.predict_proba(X)
    
        if 'label_type' in config.keys() and config['label_type']=='multiclass':
            roc_aucs, accuracies = compute_multiclass_indicators(Y, labels_proba)
            outputs['proba_of_1'] = labels_proba
            outputs['roc_auc'] = roc_aucs
            outputs['balanced_accuracy'] = accuracies
        else:
            curves, roc_auc, accuracy, labels_pred = compute_binary_indicators(Y, labels_proba)
            outputs['proba_of_1'] = labels_proba[:, 1]
            outputs['label_pred'] = labels_pred
            outputs['roc_auc'] = roc_auc
            outputs['balanced_accuracy'] = accuracy
            outputs['curves'] = curves

    # Add the GridSearchCV results
    ## TODO : remove compute indicators, just use cv_results_
    # TODO : get the proba predictions etc ..., all from GridSearchCV
    cv_score = clf.cv_results_['mean_test_score'][clf.best_index_]
    cv_std = clf.cv_results_['std_test_score'][clf.best_index_]
    splits_score = [clf.cv_results_[f'split{k}_test_score'][clf.best_index_] for k in range(0,5)] ## TODO : NOT ASSUME 5 CV HERE

    outputs['cv_score'] = cv_score
    outputs['cv_std'] = cv_std
    outputs['splits_score']=splits_score

    return outputs
    

@ignore_warnings(category=ConvergenceWarning)
def train_n_repeat_classifiers(config, subset='full'):
    """Sets up the save paths, loads the embeddings and then loops 
    to train the n_repeat (=5) classifiers."""
    ## import the data

    # set up load and save paths
    train_embs_path = config.training_embeddings
    # /!\ in fact all_labels (=train_val and test labels)
    train_lab_paths = config.data[0].subject_labels_file

    # if not specified, the outputs of the classifier will be stored next
    # to the embeddings used to generate them
    results_save_path = (config.results_save_path if config.results_save_path
                         else train_embs_path)

    # remove the filename from the path if it is a file
    if not os.path.isdir(results_save_path):
        results_save_folder, _ = os.path.split(results_save_path)
    else:
        results_save_folder, _ = results_save_path, ''
    # add a subfolder with the evaluated label as name
    results_save_folder = results_save_folder + "/" + config.label_names[0]
    if not os.path.exists(results_save_folder):
        os.makedirs(results_save_folder)

    embeddings, labels = load_embeddings(
        train_embs_path, train_lab_paths, config, subset=subset)
    names_col = 'ID' if 'ID' in embeddings.columns else 'Subject' # issue here ?
    X = embeddings.loc[:, embeddings.columns != names_col]
    Y = labels.label
    # make sur X and labels have the same subjects
    #common_ids = X.index.intersection(labels['Subject'])
    # Filter both DataFrames
    #X = X.loc[common_ids].reset_index().drop_duplicates(subset='ID', keep='first').set_index('ID')
    #Y = labels[labels['Subject'].isin(common_ids)].label
    print(f'Total number of subjects considered : {X.shape[0]}')

    subjects = embeddings.index.tolist()
    # Builds objects where the results are saved
    # Depending on label type
    if 'label_type' in config.keys() and config['label_type']=='continuous':
        pred_values_list = []
    elif 'label_type' in config.keys() and config['label_type']=='multiclass':
        aucs = {'cross_val': []}
        accuracies = {'cross_val': []}
        proba_pred_list = []
        cv_scores = []
        cv_stds = []
        splits_scores = []
    else:
        Curves = {'cross_val': []}
        aucs = {'cross_val': []}
        accuracies = {'cross_val': []}
        cv_scores = []
        cv_stds = []
        splits_scores = []
        if config.split!='train_test':
            proba_matrix = np.zeros((labels.shape[0], config.n_repeat))
        else:
            test = pd.read_csv(os.path.join(config.embeddings_save_path, 'test_embeddings.csv'), usecols=['ID'])
            proba_matrix = np.zeros((test.shape[0], config.n_repeat)) # report only test samples
        labels_pred_matrix = np.zeros(proba_matrix.shape, dtype=int)

    inputs = {}
    # rescale embeddings
    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X)
    if 'label_type' in config.keys() and config['label_type']=='continuous':
        Y= Y.to_numpy().reshape(-1, 1)
        Y = scaler.fit_transform(Y)
        Y = Y.reshape(-1)
        Y = pd.Series(Y)
    inputs['X'] = X
    inputs['Y'] = Y

    if config.split=='train_test':
        # Save perfs and probas only for test subs
        # Reformat subset and labels
        subset = 'test'
        test = pd.read_csv(os.path.join(config.embeddings_save_path, 'test_embeddings.csv'), usecols=['ID'])
        # restrict labels to test
        labels = labels.loc[labels['Subject'].isin(test.ID)]

    # Configures loops

    repeats = range(config.n_repeat)

    ## Train classifiers
    ## TODO : simplified outputs
    if 'label_type' in config.keys() and config['label_type']=='continuous':
        outputs = train_one_classifier(config, inputs, subjects) # perform once since it's deterministic
        #labels_preds = outputs['labels_pred']
        # TODO: add a list of labels preds and save like proba matrix
        reg_auc = outputs['reg_auc']
        mse = outputs['MSE']
        mae = outputs['MAE']
        r2 = outputs['r2']
        pred_vs_true = outputs['pred_vs_true']
        cv_score = outputs['cv_score']
        cv_std = outputs['cv_std']
        splits_score = outputs['splits_score']
        
        values = {}
        values[f'{subset}_auc'] = reg_auc
        values[f'{subset}_mse'] = mse
        values[f'{subset}_mae'] = mae
        values[f'{subset}_r2'] = r2
        values[f'cv_score'] = cv_score
        values[f'cv_std'] = cv_std
        values[f'splits_score']=splits_score
        # save results
        print(f"results_save_path = {results_save_folder}")
        filename = f"{subset}_values.json"
        with open(os.path.join(results_save_folder, filename), 'w+') as file:
            json.dump(values, file)
        print(f'Regression AUC: {reg_auc}, MSE: {mse}, MAE: {mae}, r2: {r2}')

        #plot regression
        print(pred_vs_true.shape)
        plt.scatter(pred_vs_true[:, 0], pred_vs_true[:, 1])
        plt.xlabel('True label')
        plt.ylabel('prediction')
        filename = f"{subset}_prediction_plot.png"
        plt.savefig(os.path.join(results_save_folder, filename))
        plt.close()

    else:

        # Actual loop done config.n_repeat times
        if _parallel:
            print(f"Computation done IN PARALLEL: {config.n_repeat} times")
            print(f"Number of subjects used by the SVM: {len(inputs['X'])}")
            func = partial(train_one_classifier, config, inputs, subjects)
            outputs = pqdm(repeats, func, n_jobs=define_njobs())
        else:
            outputs = []
            print("Computation done SERIALLY")
            for i in repeats:
                print("model number", i)
                outputs.append(train_one_classifier(config, inputs, subjects, i))
        
        if 'label_type' in config.keys() and config['label_type']=='multiclass':
            # Put together the results
            for i, o in enumerate(outputs):
                roc_auc = o['roc_auc']
                accuracy = o['balanced_accuracy']
                probas_pred = o['proba_of_1']
                cv_score = o['cv_score']
                cv_std = o['cv_std']
                splits_score = o['splits_score']

                aucs['cross_val'].append(roc_auc)
                accuracies['cross_val'].append(accuracy)
                proba_pred_list.append(probas_pred)
                cv_scores.append(cv_score)
                cv_stds.append(cv_std)
                splits_scores.append(splits_score)
            
            #save proba matrix
            columns_names = [str(int(i)) for i in np.unique(labels.label)]
            for k, probas_pred in enumerate(proba_pred_list):
                filename = f"{subset}_probas_pred_{k}.csv"
                probas = pd.DataFrame(probas_pred, columns=columns_names, index=labels.Subject)
                probas.to_csv(os.path.join(results_save_folder, filename))

            #compute metrics and print
            values = {}
            mode = 'cross_val'
            aucs = np.array(aucs[mode])
            accuracies = np.array(accuracies[mode])
            # compute weighted auc using label proportion in inputs
            weighted_auc = 0
            auc_list = (np.mean(aucs, axis=0)).tolist()
            for idx, number in enumerate(np.unique(inputs['Y'], return_counts=True)[1]):
                auc = auc_list[idx]
                weight = number / len(inputs['Y'])
                weighted_auc += auc*weight

            target =  f'{subset}_ovr_balanced_accuracy'
            values[target] = [(np.mean(accuracies, axis=0)).tolist(), (np.std(accuracies, axis=0)).tolist()]
            print(f"{subset} cross_val accuracy:\n", f'Average: {np.mean(values[target][0])}'
                  , f'OVR: {values[target]}')
            values[f'{subset}_total_balanced_accuracy']=\
                [np.mean(accuracies), np.mean(values[target][1])]
            
            target = f'{subset}_ovr_auc'
            values[target] = [auc_list, (np.std(aucs, axis=0)).tolist()]
            print(f"{subset} cross_val AUC:\n", f'Average: {np.mean(values[target][0])}'
                  , f'Weighted: {weighted_auc}'
                  , f'OVR: {values[target]}')
            values[f'{subset}_auc']=\
                [np.mean(aucs), np.mean(values[target][1])]
            values[f'{subset}_weighted_auc']=weighted_auc
            values['cv_score']=np.mean(cv_scores) # TODO : no reason to take mean
            values['cv_std']=np.mean(cv_stds)
            values['splits_score']=splits_score
        else:
            # Put together the results
            for i, o in enumerate(outputs):
                probas_pred = o['proba_of_1']
                labels_pred = o['label_pred']
                curves = o['curves']
                roc_auc = o['roc_auc']
                accuracy = o['balanced_accuracy']
                cv_score = o['cv_score']
                cv_std = o['cv_std']
                splits_score = o['splits_score']

                proba_matrix[:, i] = probas_pred
                labels_pred_matrix[:, i] = labels_pred
                Curves['cross_val'].append(curves)
                aucs['cross_val'].append(roc_auc)
                accuracies['cross_val'].append(accuracy)
                cv_scores.append(cv_score)
                cv_stds.append(cv_std)
                splits_scores.append(splits_score)

            # add the predictions to the df where the true values are
            columns_names = ["proba_pred_"+str(i) for i in range(config.n_repeat)]
            probas = pd.DataFrame(
                proba_matrix, columns=columns_names, index=labels.index)
            columns_names_label_pred = ["label_pred_"+str(i) for i in range(config.n_repeat)]
            labels_preds = pd.DataFrame(
                labels_pred_matrix, columns=columns_names_label_pred, index=labels.index)
            labels = pd.concat([labels, probas, labels_preds], axis=1)

            # post processing (mainly plotting graphs)
            values = {}
            mode = 'cross_val'

            ### TODO : add cv_scores in post_processing !!
            post_processing_results(cv_scores, cv_stds, splits_scores, labels, embeddings, Curves, aucs,
                                    accuracies, values, columns_names,
                                    mode, subset, results_save_folder)
            
        # save results
        print(f"results_save_path = {results_save_folder}")
        filename = f"{subset}_values.json"
        with open(os.path.join(results_save_folder, filename), 'w+') as file:
            json.dump(values, file)

        # plt.show()
        plt.close('all')

        save_used_label(os.path.dirname(results_save_folder), config)


#@hydra.main(config_name='config_no_save', config_path="../configs")
def train_classifiers(config, subsets=None):
    """Train classifiers (either SVM or neural networks) to classify target embeddings
    with the given label.
    
    All the relevant information should be passed thanks to the input config.
    
    It saves txt files containg the acuracies, the aucs and figures of the ROC curves."""

    config = process_config(config)

    set_root_logger_level(config.verbose)

    if config.split is None or config.split=='random':
        for subset in subsets:
            print("\n")
            log.info(f"USING SUBSET {subset}")
            # the choice of the classifiers' type is now inside the function 
            train_n_repeat_classifiers(config, subset=subset)
    
    elif config.split=='custom':
        print("\n")
        log.info(f"USING SUBSET test")
        # make sure a train file is given in this mode
        train_n_repeat_classifiers(config, subset='test')
    
    elif config.split=='train_test':
        print("\n")
        log.info(f"USING SUBSET full")
        # make sure a train file is given in this mode
        train_n_repeat_classifiers(config, subset='full')


if __name__ == "__main__":
    train_classifiers()
