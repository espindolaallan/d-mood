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


import numpy as np
import random
from pymoo.core.crossover import Crossover
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from multiprocessing import Pool
from pymoo.core.callback import Callback
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import pickle
from EspPipeML import esp_utilities
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from fastai.tabular.all import *
from sklearn.tree import DecisionTreeClassifier
from icecream import ic
import time


class EnsembleSamplingNoReposition(Sampling):
     def _do(self, problem, n_samples, **kwargs):
        
        # Define the function to set the chromosome for each classifier
        def setting_chromsome_for_each_clf(clf_name, problem):
            total_features = int((problem.n_var / 5 ) - problem.n_hp)
            X = np.empty((0, total_features), dtype=int)

            if clf_name in ['dt']:
                hp1 = np.random.randint(low=3, high=25, size=n_samples).reshape(-1, 1).astype(int) #max_depth
                hp2 = np.zeros((n_samples, 1), dtype=int) #n_estimator
            if clf_name in ['rf', 'xgb']:
                hp1 = np.random.randint(low=3, high=25, size=n_samples).reshape(-1, 1).astype(int) #max_depth
                hp2 = np.random.randint(low=50, high=100, size=n_samples).reshape(-1, 1).astype(int) #n_estimator
            elif clf_name in ['mlp']:
                hp1 = np.random.randint(low=problem.mlp_neuron_min, high=problem.mlp_neuron_max, size=n_samples).reshape(-1, 1).astype(int) #number of neurons
                hp2 = np.random.randint(low=problem.mlp_hidden_min, high=problem.mlp_hidden_max, size=n_samples).reshape(-1, 1).astype(int) #hidden layers
            elif clf_name in ['td']:
                hp1 = np.random.randint(low=problem.td_neuron_min, high=problem.td_neuron_max, size=n_samples).reshape(-1, 1).astype(int) #number of neurons
                hp2 = np.random.randint(low=problem.td_hidden_min, high=problem.td_hidden_max, size=n_samples).reshape(-1, 1).astype(int) #hidden layers

            vec_n_features = np.random.randint(low=problem.f_min, high=problem.f_max, size=n_samples)

            for n_features in vec_n_features:
                selected_features = np.random.permutation(total_features)[:n_features] #features no repo
                chromosome = np.full(total_features, -1, dtype=int) # create chromosome with -1
                chromosome[:n_features] += selected_features #add selected features to the chromosome

                X = np.vstack((X, chromosome)) #add chromsome to the matrix
            
            X = np.concatenate((hp1, hp2, X), axis=1) #add hp 1 and 2 to the matrix at the first pos
            return X
        
        X_list = []
        # Create the chromosome for each classifier
        for clf_name in ['dt', 'rf', 'xgb', 'mlp', 'td']:
            X_list.append(setting_chromsome_for_each_clf(clf_name, problem))
        # Concatenate the chromosomes into a single matrix
        X = np.hstack(X_list)
        return X
     
class FullIdentityPreservingCrossover(Crossover):
    def __init__(self, prob=0.9, **kwargs):
        super().__init__(2, 2, prob, **kwargs)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape
        Q = np.empty((self.n_offsprings, n_matings, n_var), dtype=int)

        # Each classifier has n_var_per_classifier variables
        n_var_per_classifier = n_var // 5  # Number of variables per classifier

        for mating in range(n_matings):
            offsprings = np.empty((self.n_offsprings, n_var), dtype=int)
            for classifier_index in range(5):  # Iterate over each classifier
                # Determine the start and end positions of the variables for the current classifier
                start = classifier_index * n_var_per_classifier
                end = start + n_var_per_classifier

                # Get the parents for the current classifier
                p1, p2 = X[0, mating, start:end], X[1, mating, start:end]
                p1, p2 = p1[p1 != -1], p2[p2 != -1]  # Remove empty features (-1)
                common_elements, diff_elements = self._preprocess_for_crossover(p1, p2, problem.n_hp)
                
                # Crossover for the current classifier
                offsprings_part = self._crossover([p1, p2], common_elements, diff_elements, problem.n_hp)

                # Insert the offspring into the correct position in the offsprings matrix
                offsprings[:, start:end] = self._padding_offspring(offsprings_part, problem)

            # Add the offsprings to the final matrix
            Q[:, mating, :] = offsprings

        return Q

    

    def _get_diff_and_shuffle(self, parent_features, common_elements):
        #diff elements
        p1 = np.setdiff1d(parent_features[0], common_elements)
        p2 = np.setdiff1d(parent_features[1], common_elements)
        #shuffle
        np.random.shuffle(p1)
        np.random.shuffle(p2)
        return [p1, p2]

    def _preprocess_for_crossover(self, p1, p2, n_hp):
        parent_features = [p1[n_hp:], p2[n_hp:]] # only features
        common_elements = np.intersect1d(parent_features[0], parent_features[1])
        # Get diff elements and shuffle them 
        diff_elements = self._get_diff_and_shuffle(parent_features, common_elements)
        return common_elements, diff_elements

    def _padding_offspring(self, offsprings, problem):
        pad_len_1 = int((problem.n_var / 5) - offsprings[0].shape[0])
        pad_len_2 = int((problem.n_var / 5) - offsprings[1].shape[0])

        offspring_1 = np.concatenate((offsprings[0], np.full(pad_len_1, -1)))
        offspring_2 = np.concatenate((offsprings[1], np.full(pad_len_2, -1)))

        return [offspring_1, offspring_2]


    def _crossover(self, parents, common_elements, diff_elements, n_hp):
        len_diff_1, len_diff_2 = len(diff_elements[0]), len(diff_elements[1])
        #Calculate crossover using half of max diff elements
        #crossover_point = int(max(len_diff_1, len_diff_2) * 0.5)

        #Calculate crossover point as a random percentage (30%-100%)
        crossover_point = int(max(len_diff_1, len_diff_2) * random.uniform(0.3, 1.0) + 0.5)


        to_cross_1, to_keep_1 = diff_elements[0][:crossover_point], diff_elements[0][crossover_point:]
        to_cross_2, to_keep_2 = diff_elements[1][:crossover_point], diff_elements[1][crossover_point:]
        preserv_offspring_1 = np.concatenate((parents[0][:n_hp], common_elements))
        preserv_offspring_2 = np.concatenate((parents[1][:n_hp], common_elements))

        offspring_1 = np.concatenate((preserv_offspring_1, to_keep_1, to_cross_2))
        offspring_2 = np.concatenate((preserv_offspring_2, to_keep_2, to_cross_1))
        return [offspring_1, offspring_2]



class AddDeleteReplaceFeatureMutation(Mutation):
    def __init__(self, prob=0.3, prob_var=None, **kwargs) -> None:
        super().__init__(prob, prob_var, **kwargs)

        self.prob = prob

    def _do(self, problem, X, **kwargs):
        X_by_clf = np.split(X, 5, axis=1)
        X_list = []

        for clf_name, X_clf in zip(['dt', 'rf', 'xgb', 'mlp', 'td'], X_by_clf):
            #delete
            _X = self._delete(X_clf, problem)
            #rep
            _X = self._replace(_X, problem)
            #add
            _X = self._add(_X, problem)
            
            # mutate neurons and hidden layers
            if clf_name in ['mlp', 'td']:
                _X = self._mutate_network_structure(_X, problem, clf_name)

            # mutate tree hyperparameters
            elif clf_name in ['dt', 'rf', 'xgb']:
                _X = self._mutate_tree_hyperparameters(_X, problem, clf_name)

            X_list.append(_X)

        return np.concatenate(X_list, axis=1)
    
    
    def prob_decision(self):
        return np.random.rand() < self.prob
    

    def _mutate_tree_hyperparameters(self, X, problem, clf_name):
        _X = X.copy()
        for i in range(_X.shape[0]):
            hp = _X[i, :problem.n_hp]
            if clf_name in ['rf', 'xgb']:
                hp[0] = np.random.randint(3, 25)
                hp[1] = np.random.randint(50, 100)
                decision = np.random.choice(['add_depth', 'remove_depth', 'add_tree', 'remove_tree', 'none'], p=[0.2, 0.2, 0.2, 0.2, 0.2])
            elif clf_name in ['dt']:
                hp[0] = np.random.randint(3, 25)
                decision = np.random.choice(['add_depth', 'remove_depth', 'none'], p=[0.3, 0.3, 0.4])

            if decision == 'add_depth' and hp[0] < 25:
                hp[0] += 1
            elif decision == 'remove_depth' and hp[0] > 3:
                hp[0] -= 1
            elif decision == 'add_tree' and hp[1] < 100:
                hp[1] += 1
            elif decision == 'remove_tree' and hp[1] > 50:
                hp[1] -= 1

            _X[i, :problem.n_hp] = hp
        return _X

    def _mutate_network_structure(self, X, problem, clf_name):
        """
        Mutates the network structure of each solution in X for a specified classifier (clf_name)
        by either adding or removing a layer or neuron, or potentially making no change.
        The probability of each decision is 0.2.
        """
        _X = X.copy()

        for i in range(_X.shape[0]):
            hp = _X[i, :problem.n_hp]  # hiperparâmetros

            # determine the min and max number of neurons and hidden layers for the specified classifier
            if clf_name == 'mlp':
                min_neuron, max_neuron = problem.mlp_neuron_min, problem.mlp_neuron_max
                min_hidden, max_hidden = problem.mlp_hidden_min, problem.mlp_hidden_max
            elif clf_name == 'td':
                min_neuron, max_neuron = problem.td_neuron_min, problem.td_neuron_max
                min_hidden, max_hidden = problem.td_hidden_min, problem.td_hidden_max
            else:
                raise ValueError("Unknown classifier name")

            decision = np.random.choice(['add_neuron', 'remove_neuron', 'add_layer', 'remove_layer', 'none'], p=[0.2, 0.2, 0.2, 0.2, 0.2])

            if decision == 'add_neuron' and hp[0] < max_neuron:
                hp[0] += 1
            elif decision == 'remove_neuron' and hp[0] > min_neuron:
                hp[0] -= 1
            elif decision == 'add_layer' and hp[1] < max_hidden:
                hp[1] += 1
            elif decision == 'remove_layer' and hp[1] > min_hidden:
                hp[1] -= 1

            _X[i, :problem.n_hp] = hp

        return _X

    def _add(self, X, problem):
        max_size = problem.f_max
        _X = X.copy()
        all_features = set(range(problem.n_features))

        for k in range(_X.shape[0]):
            features = _X[k][problem.n_hp:]
            features = features[features != -1] #only valid features
            hp = _X[k][:problem.n_hp] #hyperparameters
            len_features = features.shape[0]
            available_features = list(all_features - set(features))
            
            if len_features < max_size and self.prob_decision(): #conditions: size and prob true
                selected_feature = random.choice(available_features)
                features = np.append(features, selected_feature)

            chromosome = np.concatenate((hp, features))
            _X[k] = self._pedding_chromosome(chromosome, problem)
        return _X
    
    def _delete(self, X, problem):
        min_size = problem.f_min
        _X = X.copy()

        for k in range(_X.shape[0]):
            features = _X[k][problem.n_hp:]
            features = features[features != -1] #only valid features
            hp = _X[k][:problem.n_hp] #hyperparameters
            len_features = features.shape[0]
            _features = []

            for i in features:
                if len_features > min_size and self.prob_decision(): #conditions: size and prob true
                    len_features -= 1
                else:
                    _features.append(i)
            chromosome = np.concatenate((hp, _features))
            _X[k] = self._pedding_chromosome(chromosome, problem)
        return _X
            
    def _replace(self, X, problem):
        _X = X.copy()
        all_features = set(range(problem.n_features))

        for k in range(_X.shape[0]):
            features = _X[k][problem.n_hp:]
            features = features[features != -1]  # only valid features
            _features = []
            hp = _X[k][:problem.n_hp]  # hyperparameters
            available_features = list(all_features - set(features))

            for i in features:
                if self.prob_decision() and available_features:  # Checa a cada iteração
                    selected_feature = random.choice(available_features)
                    available_features.remove(selected_feature)  # update available list of features
                    _features.append(selected_feature)
                else:
                    _features.append(i)

            chromosome = np.concatenate((hp, _features))
            _X[k] = self._pedding_chromosome(chromosome, problem)
        return _X

    def _pedding_chromosome(self, chromosome, problem):
        pad_len = problem.n_var_by_clf - chromosome.shape[0]
        _chromosome = np.concatenate((chromosome, np.full(pad_len, -1)))
        return _chromosome
    


class EnsembleSelectionProblem(Problem):
    def __init__(self, X_train, y_train, y_train_multiclass, X_test, y_test, n_features, f_min, f_max, n_hp,
                 feature_names, cores=1, fitness_metric='auc', device='cuda', **kwargs):           
        super().__init__(
            n_var=(n_features + n_hp)*5, 
            n_obj=3, 
            n_constr=0,
            )
        
        self.X_train = X_train
        self.y_train = y_train
        self.y_train_multiclass = y_train_multiclass
        self.X_test = X_test
        self.y_test = y_test
        self.f_min = f_min
        self.f_max = f_max
        self.n_hp = n_hp
        self.n_features = n_features
        self.feature_names = feature_names
        self.mlp_neuron_min = 100
        self.mlp_neuron_max = 200
        self.mlp_hidden_min = 2
        self.mlp_hidden_max = 4
        self.td_neuron_min = 256
        self.td_neuron_max = 512
        self.td_hidden_min = 5
        self.td_hidden_max = 20
        self.cores = cores
        self.fitness_metric = fitness_metric
        self.device = device
        self.n_var_by_clf = self.n_var // 5

    def _evaluate(self, X, out, *args, **kwargs):
        with Pool(processes=self.cores) as pool:
            results = pool.starmap(evaluate_a_chromosome, [(x, self.X_train, self.y_train, self.y_train_multiclass, self.X_test, self.y_test, 
                                                            self.feature_names, self.fitness_metric, self.device) for x in X])
        
        performance_score, disagreement_score, effort_score, y_preds = zip(*results)
        out["F"] = np.column_stack([performance_score, disagreement_score, effort_score])

        

def evaluate_a_chromosome(x, X_train, y_train, y_train_multiclass, X_test, y_test, feature_names, fitness_metric, device = 'cuda'):
    
    def train_and_evaluate_sklearn_models(clf, X_train, y_train, X_test):       
        clf.fit(X_train, y_train.values.ravel())
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        return y_prob, y_pred
    
    threshold = 0.3 # threshold for binary classification (mlp and td)

    # Split the chromosome by classifier
    x_by_classifier = np.split(x, 5)
    x_decoded = {}
    X_train_selected = {}
    X_test_selected = {}

    # Decode the chromosome for each classifier
    for xi, clf_name in zip(x_by_classifier, ['dt', 'rf', 'xgb', 'mlp', 'td']):
        x_decoded[clf_name] = XDecode(x=xi, clf_name=clf_name, feature_names=feature_names)
    
        # Get the features for each classifier
        X_train_selected[clf_name] = X_train.iloc[:, x_decoded[clf_name].features]
        X_test_selected[clf_name] = X_test.iloc[:, x_decoded[clf_name].features]

        # in the case where no features are selected
        if not np.any(x_decoded[clf_name].features):
            return 1

    # Define the classifiers
    clf_dt = DecisionTreeClassifier(
        max_depth=x_decoded['dt'].max_depth,
        random_state=42,
        class_weight='balanced'
    )
    y_prob_dt, y_pred_dt = train_and_evaluate_sklearn_models(clf_dt, X_train_selected['dt'], y_train, X_test_selected['dt'])

    clf_rf = RandomForestClassifier(
        n_estimators=x_decoded['rf'].n_estimators,
        max_depth=x_decoded['rf'].max_depth,
        random_state=42,
        class_weight='balanced'
    )
    y_prob_rf, y_pred_rf = train_and_evaluate_sklearn_models(clf_rf, X_train_selected['rf'], y_train, X_test_selected['rf'])
    
    count_class_0, count_class_1 = y_train.value_counts()
    scale_pos_weight = count_class_0 / count_class_1

    clf_xgb = XGBClassifier(
        n_estimators=x_decoded['xgb'].n_estimators,
        max_depth=x_decoded['xgb'].max_depth,
        random_state=42, 
        tree_method='hist', 
        # device=device, 
        scale_pos_weight=scale_pos_weight
    )
    y_prob_xgb, y_pred_xgb = train_and_evaluate_sklearn_models(clf_xgb, X_train_selected['xgb'], y_train, X_test_selected['xgb'])

    torch.cuda.set_device(device)
    layers = [x_decoded['mlp'].n_neurons] * x_decoded['mlp'].n_layers
    # Get the input dimension
    input_dim = X_train_selected['mlp'].shape[1]
    # Create and move the model to the specified device
    clf_mlp = esp_utilities.MLP(
        input_dim=input_dim, 
        num_neurons=x_decoded['mlp'].n_neurons, 
        num_layers=x_decoded['mlp'].n_layers
        )
    clf_mlp.to(device)
    esp_utilities.train_mlp(X_train_selected['mlp'].to_numpy(), y_train.to_numpy(), model=clf_mlp, device=device, y_train_multiclass=y_train_multiclass) # train the model
    y_prob_mlp = esp_utilities.get_probabilities(clf_mlp, X_test_selected['mlp'], device=device) # get the probabilities
    y_pred_mlp = (y_prob_mlp > threshold).astype(int) # get the predictions


    # Train the classifiers
    td_object = x_decoded['td']
    layers = [td_object.n_neurons] * td_object.n_layers
    y_name = 'label'


    train_df = pd.concat([X_train_selected['td'].reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_selected['td'].reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    cont_names = x_decoded['td'].selected_feature_names   # Since all features are numerical, I will only have continuous features
    cat_names = []  # No categorical data in your case
    procs = []  # No additional preprocessing needed as the data is already normalized

    splits = stratified_splitter(train_df, y_train_multiclass, valid_pct=0.2)
    # TabularPandas for training data
    to = TabularPandas(train_df, 
                procs=procs, 
                cat_names=cat_names, 
                cont_names=cont_names, 
                y_names=y_name, 
                splits=splits, 
                y_block=CategoryBlock())  # Use CategoryBlock for classification tasks
    
    # TabularPandas for test data
    to_test = TabularPandas(test_df,
                procs=procs, 
                cat_names=cat_names, 
                cont_names=cont_names, 
                y_names=y_name, 
                y_block=CategoryBlock())
    
    # Create DataLoaders for training and validation
    dls = to.dataloaders(bs=256, device=torch.device(device))

    # Create DataLoaders for test data
    dls_test = to_test.dataloaders(bs=256, device=torch.device(device))

    # Create and move the model to the specified device
    clf_td = tabular_learner(dls, layers=layers)
    clf_td.model.to(torch.device(device))


    # Train the model
    clf_td.fit_one_cycle(100, 0.001, cbs=[EarlyStoppingCallback(monitor='valid_loss', min_delta=0, patience=5)])


    # Get the predictions
    probs_td, targs = clf_td.get_preds(dl=dls_test[0])

    y_prob_td = probs_td[:, 1].numpy()  # Probabilities of class 1, positive class
    y_pred_td = (probs_td[:, 1] > threshold).int().numpy()

    # Get the mean probabilities
    mean_probs = np.mean([y_prob_dt, y_prob_rf, y_prob_xgb, y_prob_mlp, y_prob_td], axis=0)
    # Get the predictions
    all_preds = np.array([y_pred_dt, y_pred_rf, y_pred_xgb, y_pred_mlp, y_pred_td])
    # Get the majority voting
    y_vote = np.sum(all_preds, axis=0) > (all_preds.shape[0] / 2)

    if fitness_metric == 'auc':
        performance_score = 1 - roc_auc_score(y_test, mean_probs)
    elif fitness_metric == 'f1':
        performance_score = 1 - f1_score(y_test, y_vote, average='weighted')

    
    predictions = {
        'dt': y_pred_dt,
        'rf': y_pred_rf,
        'xgb': y_pred_xgb,
        'mlp': y_pred_mlp,
        'td': y_pred_td
    }

    mean_disagreement = esp_utilities.ensemble_disagreement(predictions)
    mean_disagreement = 1 - mean_disagreement

    
    effort_score,  ensemble_effort = esp_utilities.get_ensemble_effort(clf_dt, clf_rf, clf_xgb, clf_mlp, clf_td)

    del clf_mlp
    torch.cuda.empty_cache()

    # ic(performance_score, mean_disagreement, effort_score, ensemble_effort)

    return (performance_score, mean_disagreement, effort_score, y_vote)


class XDecode():
    def __init__(self, x, clf_name, feature_names, eval=False):
        self.features = self.get_features(x, initial_position=2, eval=eval)
        self.selected_feature_names = self.get_selected_features(feature_names)

        if clf_name in ['dt']:
            self.max_depth = self.get_first_hp(x)
        elif clf_name in ['rf', 'xgb']:
            self.max_depth = self.get_first_hp(x)
            self.n_estimators = self.get_second_hp(x)
        
        elif clf_name in ['mlp', 'td']:
            self.n_neurons = self.get_first_hp(x)
            self.n_layers = self.get_second_hp(x)

    def get_features(self, x, initial_position=2, eval=False):
        features = x[initial_position:]
        if not eval:
            return features[features != -1]
        else:
            return [int(num) for num in features if num >= 0][:-2]
    
    def get_first_hp(self, x):
        return x[0]
    
    def get_second_hp(self, x):
        return x[1]
    
    def get_selected_features(self, feature_names):
        selected_feature_names = [feature_names[i] for i in self.features]
        return selected_feature_names


class MyCallback(Callback):

    def __init__(self, config_name):
        super().__init__()
        self.config_name = config_name
        self.history = []

    def __call__(self, algorithm):
        self.save_history(algorithm)

        self.save_checkpoint(algorithm)

    def save_history(self, algorithm):
        # Add actual population to history
        self.history.append(algorithm.pop)

        # Save history to file
        history_file = "{}_history.pkl".format(self.config_name)
        with open(history_file, 'wb') as f:
            pickle.dump(self.history, f)

    def save_checkpoint(self, algorithm):
        current_gen = algorithm.n_gen
        create_checkpoint = not (current_gen + 1) % 10  #checkpoint every 10th gen
        if create_checkpoint:
            checkpoint_file = "{}_checkpoint_gen_{}.pkl".format(self.config_name, current_gen)
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({'algorithm': algorithm, 'current_gen': current_gen}, f)
            self.last_checkpoint_gen = current_gen
            print(f"Checkpoint saved for generation {current_gen}")

def stratified_splitter(df, stratify_col, valid_pct=0.2):
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=valid_pct, random_state=42)
    for train_index, valid_index in stratified_split.split(df, stratify_col):
        return list(train_index), list(valid_index)
