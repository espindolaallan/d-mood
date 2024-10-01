
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

from my_operations_ensemble import EnsembleSamplingNoReposition, FullIdentityPreservingCrossover, AddDeleteReplaceFeatureMutation, EnsembleSelectionProblem, MyCallback
esp_utilities.fix_seeds(42)

def continue_from_checkpoint(checkpoint_file, remaining_gens, problem):
    algorithm = esp_utilities.load_from_pickle(checkpoint_file)

    # Continue optimization
    res = minimize(
        problem,
        algorithm,
        ('n_gen', remaining_gens),   # This is how many generations are left
        seed=1,
        verbose=True,
        save_history=True,
        callback=MyCallback(checkpoint_file)  # Reuse the callback for further checkpointing
    )

    return res


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
            sampling=EnsembleSamplingNoReposition(),
            mutation=AddDeleteReplaceFeatureMutation(prob=configs['mutation_prob']),
            crossover=FullIdentityPreservingCrossover(),
        )
        remaining_gens = configs['n_gen']

    problem = EnsembleSelectionProblem(
        configs['X_train'],
        configs['y_train'],
        configs['y_train_multiclass'],
        configs['X_test'],
        configs['y_test'],
        configs['n_features'],
        configs['f_min'],
        configs['f_max'],
        configs['n_hp'],
        configs['feature_names'],
        configs['cores'],
        configs['fitness_metric'],
        configs['device']
    )

    # Set up the callback with a checkpoint file name
    checkpoint_file_name = configs['check_point_name']
    callback = MyCallback(checkpoint_file_name)
    
    res = minimize(
            problem,
            algorithm,
            ('n_gen', remaining_gens),
            seed=1,
            verbose=True,
            save_history=True,
            callback=callback
        )
    return res

def run_everything(dataset_name, checkpoint_file=None, cores=1, fitness_metric='auc', device='cuda'):
    dataloader = esp_utilities.DatasetLoader(dataset_name, scale_data=True, scale_on_full_dataset=False)
    # dataloader = esp_utilities.DatasetLoader(dataset_name, scale_data=True, scale_on_full_dataset=False, toy=True)
    
    # creates a stratified train-validation split and converts the data to pandas dataframes
    X_train, X_val, y_train, y_val, y_train_multiclass, y_val_multiclass = [
        pd.DataFrame(data) for data in train_test_split(
                                                        dataloader.X_train, 
                                                        dataloader.y_train, 
                                                        dataloader.y_train_multiclass, 
                                                        test_size=0.3, 
                                                        # stratify=dataloader.y_train_multiclass, 
                                                        random_state=42
                                                        )
        ]


    # dataloader = esp_utilities.DatasetLoader('unsw-nb15', scale_data=True, scale_on_full_dataset=False, toy=[True, 100])
    execution_name = '../results/nsga2/feature_selection/'+ dataset_name
    check_point_name = '../results/nsga2/checkpoint/feature_selection/'+ dataset_name
    
    configs = {
        'execution_name': '',
        'check_point_name': '',
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
        'device': device,
        'n_hp': 2,
        'execution_name': execution_name + '_proposal_' + fitness_metric,
        'check_point_name': check_point_name + '_proposal_' + fitness_metric
    }


    if checkpoint_file:
        res = run_nsga2(configs, checkpoint_file)
    else:
        res = run_nsga2(configs)

    esp_utilities.save_to_pickle(res, configs['execution_name'] + '_res.pkl')

    print('___________________________________________\n\n')





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run machine learning models.')
    parser.add_argument('--cores', type=int, default=20, help='Number of cores to use')
    parser.add_argument('--metric', type=str, default='auc', help='Name of the fitness metric to be used for optimization')
    parser.add_argument('--device', type=str, default='cuda', help='Device to be used for training')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file to resume optimization')
    

    args = parser.parse_args()

    # Assuming 'unsw-nb15' is a fixed dataset
    run_everything('unsw-nb15', cores=args.cores, fitness_metric=args.metric, device=args.device, checkpoint_file=args.checkpoint)
