
# -------------------------------------------------------------------------------
# Script:        Feature and Model Parameter Optimization using NSGA-II
# Author:        Allan Espindola
# Created Date:  14/08/2023
# Description:   This script performs joint optimization of feature combinations 
#                and model parameters using the Non-dominated Sorting Genetic 
#                Algorithm II (NSGA-II). It systematically explores various 
#                feature subsets and their corresponding optimal model parameters 
#                to achieve the best trade-off between model performance and 
#                complexity.
# -------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from EspPipeML import esp_utilities
import argparse

import warnings
#warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.")
# Suprimir apenas o aviso específico de XGBoost sobre dispositivos incompatíveis
warnings.filterwarnings('ignore', category=UserWarning, message=".*Falling back to prediction using DMatrix due to mismatched devices.*")

from my_operations import SamplingNoReposition, FullIdentityPreservingCrossover, AddDeleteReplaceFeatureMutation, FeatureSelectionProblem, MyCallback, PerformanceTracker

def continue_from_checkpoint(checkpoint_file, remaining_gens, problem, performance_tracker):
    algorithm = esp_utilities.load_from_pickle(checkpoint_file)

    # Continue optimization
    res = minimize(
        problem,
        algorithm,
        ('n_gen', remaining_gens),   # This is how many generations are left
        seed=1,
        verbose=True,
        callback=MyCallback(checkpoint_file)  # Reuse the callback for further checkpointing
    )

    return performance_tracker, res


def run_nsga2(configs, checkpoint_file=None):
    # If a checkpoint is provided, load the previous state and continue
    if checkpoint_file:
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            algorithm = checkpoint_data['algorithm']
            last_gen = checkpoint_data['current_gen']
            remaining_gens = configs['n_gen'] - last_gen
            print(f"Resuming from generation {last_gen}. Remaining generations: {remaining_gens}.")
    else:
        # Define the NSGA-II algorithm
        algorithm = NSGA2(
            pop_size=configs['pop_size'],
            n_offsprings=int(100 * configs['rate_offsprings']),
            sampling=SamplingNoReposition(),
            mutation=AddDeleteReplaceFeatureMutation(prob=configs['mutation_prob']),
            crossover=FullIdentityPreservingCrossover(),
        )
        remaining_gens = configs['n_gen']

    performance_tracker = PerformanceTracker(
        execution_name=configs['execution_name'],
        clf_name=configs['clf_name'],
        y_true=configs['y_test'],
        n_hp=configs['n_hp'],
        y_multi_class=configs['y_test_multi_class'],
        fitness_metric=configs['fitness_metric']
    )

    problem = FeatureSelectionProblem(
        configs['X_train'],
        configs['y_train'],
        configs['y_train_multiclass'],
        configs['X_test'],
        configs['y_test'],
        configs['n_features'],
        configs['f_min'],
        configs['f_max'],
        configs['n_hp'],
        performance_tracker,
        configs['clf_name'],
        configs['feature_names'],
        configs['neuron_min'],
        configs['neuron_max'],
        configs['hidden_min'],
        configs['hidden_max'],
        configs['cores'],
        configs['fitness_metric'],
        configs['device']
    )

    # Set up the callback with a checkpoint file name
    checkpoint_file_name = configs['check_point_name'] + '_checkpoint.pkl'
    callback = MyCallback(checkpoint_file_name)
    
    res = minimize(
            problem,
            algorithm,
            ('n_gen', remaining_gens),
            seed=1,
            verbose=True,
            callback=callback
        )
    return performance_tracker, res

def run_everything(dataset_name, clfs, checkpoint_file=None, checkpoint_clf=None, cores=1, fitness_metric='auc', device='cuda'):
    dataloader = esp_utilities.DatasetLoader(dataset_name, scale_data=True, scale_on_full_dataset=False)
    # creates a stratified train-validation split and converts the data to pandas dataframes
    X_train, X_val, y_train, y_val, y_train_multiclass, y_val_multiclass = [
        pd.DataFrame(data) for data in train_test_split(
                                                        dataloader.X_train, 
                                                        dataloader.y_train, 
                                                        dataloader.y_train_multiclass, 
                                                        test_size=0.3, 
                                                        stratify=dataloader.y_train_multiclass, 
                                                        random_state=42
                                                        )
        ]


    # dataloader = esp_utilities.DatasetLoader('unsw-nb15', scale_data=True, scale_on_full_dataset=False, toy=[True, 100])
    execution_name = '../results/nsga2/feature_selection/'+ dataset_name +'_'
    check_point_name = '../results/nsga2/checkpoint/feature_selection/'+ dataset_name +'_'
    
    configs = {
        'execution_name': '',
        'check_point_name': '',
        'clf_name': '',
        'y_train': y_train,
        'X_train': X_train,
        'y_train_multiclass': y_train_multiclass,
        'y_test': y_val,
        'X_test': X_val,
        'y_test_multi_class': y_val_multiclass,
        'n_features': X_train.shape[1],
        'feature_names': X_train.columns.tolist(),
        'f_min': 5,
        'f_max': X_train.shape[1],
        'pop_size': 100, #X.shape[1]*2,
        'n_gen': 100,
        'mutation_prob': 0.1,
        'rate_offsprings': 0.9,
        'cores': cores,
        'fitness_metric': fitness_metric,
        'device': device
    }

    if checkpoint_clf:
        clfs = [checkpoint_clf]
    # else:
        # clfs = ['nb', 'rf', 'ada', 'bag', 'mlp', 'svm', 'td', 'xgb']


    for clf_name in clfs:
        print('Running {} clasifier...' .format(clf_name))
        configs['clf_name'] = clf_name
        configs['execution_name'] = execution_name + clf_name + '_' + fitness_metric
        configs['check_point_name'] = check_point_name + clf_name + '_' + fitness_metric

        if configs['clf_name'] in ['rf', 'svm', 'mlp', 'td', 'xgb']:
            configs['n_hp'] = 2
        elif configs['clf_name'] in ['dt', 'ada', 'bag']:
            configs['n_hp'] = 1
        elif configs['clf_name'] == 'nb':
            configs['n_hp'] = 0

        if configs['clf_name'] not in ['mlp', 'td']:
            configs['neuron_min'] = 0
            configs['neuron_max'] = 0
            configs['hidden_min'] = 0
            configs['hidden_max'] = 0
        elif configs['clf_name'] == 'mlp':
            configs['neuron_min'] = 100
            configs['neuron_max'] = 200
            configs['hidden_min'] = 2
            configs['hidden_max'] = 4
        elif configs['clf_name'] == 'td':
            configs['neuron_min'] = 256
            configs['neuron_max'] = 512
            configs['hidden_min'] = 5
            configs['hidden_max'] = 20

        if checkpoint_file:
            performance_tracker, res = run_nsga2(configs, checkpoint_file)
        else:
            performance_tracker, res = run_nsga2(configs)

        esp_utilities.save_to_pickle(performance_tracker, configs['execution_name'] + '_performance_tracker.pkl')
        esp_utilities.save_to_pickle(res, configs['execution_name'] + '_res.pkl')

        print('___________________________________________\n\n')





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run machine learning models.')
    parser.add_argument('--clfs', nargs='+', default=['mlp'], help='List of classifiers')
    parser.add_argument('--cores', type=int, default=20, help='Number of cores to use')
    parser.add_argument('--metric', type=str, default='auc', help='Name of the fitness metric to be used for optimization')
    parser.add_argument('--device', type=str, default='cuda', help='Device to be used for training')
    

    args = parser.parse_args()
    print(args.clfs)
    print(type(args.clfs))

    # Assuming 'unsw-nb15' is a fixed dataset
    run_everything('unsw-nb15', clfs=args.clfs, cores=args.cores, fitness_metric=args.metric, device=args.device)
