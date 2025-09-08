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

class SamplingNoReposition(Sampling):
     def _do(self, problem, n_samples, **kwargs):
        total_features = problem.n_var - problem.n_hp
        X = np.empty((0, total_features), dtype=int)

        if problem.clf_name in ['rf', 'xgb']:
            hp1 = np.random.randint(low=3, high=25, size=n_samples).reshape(-1, 1) #max_depth
            hp2 = np.random.randint(low=50, high=100, size=n_samples).reshape(-1, 1) #n_estimator
        if problem.clf_name in ['dt']:
            hp1 = np.random.randint(low=3, high=25, size=n_samples).reshape(-1, 1) #max_depth
        elif problem.clf_name in ['ada', 'bag']:
            hp1 = np.random.randint(low=50, high=100, size=n_samples).reshape(-1, 1) #n_estimator
        elif problem.clf_name == 'svm':
            hp1 = np.random.randint(low=0, high=65535, size=n_samples).reshape(-1, 1) #C
            hp2 = np.random.randint(low=0, high=65535, size=n_samples).reshape(-1, 1) #gamma
        elif problem.clf_name in ['mlp', 'td']:
            hp1 = np.random.randint(low=problem.neuron_min, high=problem.neuron_max, size=n_samples).reshape(-1, 1) #number of neurons
            hp2 = np.random.randint(low=problem.hidden_min, high=problem.hidden_max, size=n_samples).reshape(-1, 1) #hidden layers

        vec_n_features = np.random.randint(low=problem.f_min, high=problem.f_max, size=n_samples)

        for n_features in vec_n_features:
            selected_features = np.random.permutation(total_features)[:n_features] #features no repo
            chromosome = np.full(total_features, -1, dtype=int) # create chromosome with -1
            chromosome[:n_features] += selected_features #add selected features to the chromosome

            X = np.vstack((X, chromosome)) #add chromsome to the matrix
        
        if problem.clf_name in ['rf', 'xgb', 'svm', 'mlp', 'td']:
            X = np.concatenate((hp1, hp2, X), axis=1) #add hp 1 and 2 to the matrix at the first pos
        elif problem.clf_name in ['dt', 'ada', 'bag']:
            X = np.concatenate((hp1, X), axis=1) #add hp 1 to the matrix at the first pos

        return X
     

class FullIdentityPreservingCrossover(Crossover):
    def __init__(self, prob=0.9, **kwargs):
        super().__init__(2, 2, prob, **kwargs)

    def _do(self, problem, X, **kwargs):
        #print(X.shape)
        #print(X)

        n_parents, n_matings, n_var = X.shape
        Q = np.empty((self.n_offsprings, n_matings, n_var), dtype=int)
        n_hp = problem.n_hp

        for mating in range(n_matings):
            p1, p2 = X[0, mating], X[1, mating]
            p1, p2 = p1[p1 != -1], p2[p2 != -1] # removing empty features (-1)
            common_elements, diff_elements = self._preprocess_for_crossover(p1, p2, problem.n_hp)
            #crossover
            offsprings = self._crossover([p1, p2], common_elements, diff_elements, n_hp)
            #padding with -1
            offsprings = self._pedding_offspring(offsprings, problem)
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

    def _pedding_offspring(self, offsprings, problem):
        pad_len_1 = problem.n_var - offsprings[0].shape[0]
        pad_len_2 = problem.n_var - offsprings[1].shape[0]

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
        #delete
        _X = self._delete(X, problem)
        #rep
        _X = self._replace(_X, problem)
        #add
        _X = self._add(_X, problem)
        
        # mutate neurons and hidden layers
        if problem.clf_name in ['mlp', 'td']:
            _X = self._mutate_network_structure(_X, problem)

        # mutate tree hyperparameters
        elif problem.clf_name in ['dt', 'rf', 'xgb']:
            _X = self._mutate_tree_hyperparameters(_X, problem)

        return _X
    
    
    def prob_decision(self):
        return np.random.rand() < self.prob
    

    def _mutate_tree_hyperparameters(self, X, problem):
        _X = X.copy()
        for i in range(_X.shape[0]):
            hp = _X[i, :problem.n_hp]
            if problem.clf_name in ['rf', 'xgb']:
                hp[0] = np.random.randint(3, 25)
                hp[1] = np.random.randint(50, 100)
                decision = np.random.choice(['add_depth', 'remove_depth', 'add_tree', 'remove_tree', 'none'], p=[0.2, 0.2, 0.2, 0.2, 0.2])
            elif problem.clf_name in ['dt']:
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


    def _mutate_network_structure(self, X, problem):
        """
        Mutates the network structure of each solution in X by either adding or removing a layer or neuron,
        or potentially making no change. The probability of each decision is 0.2.
        """
        _X = X.copy()

        for i in range(_X.shape[0]):
            hp = _X[i, :problem.n_hp]  # hyperparameters
            decision = np.random.choice(['add_neuron', 'remove_neuron', 'add_layer', 'remove_layer', 'none'], p=[0.2, 0.2, 0.2, 0.2, 0.2])

            if decision == 'add_neuron' and hp[0] < problem.neuron_max:
                hp[0] += 1
            elif decision == 'remove_neuron' and hp[0] > problem.neuron_min:
                hp[0] -= 1
            elif decision == 'add_layer' and hp[1] < problem.hidden_max:
                hp[1] += 1
            elif decision == 'remove_layer' and hp[1] > problem.hidden_min:
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
        pad_len = problem.n_var - chromosome.shape[0]
        _chromosome = np.concatenate((chromosome, np.full(pad_len, -1)))
        return _chromosome
    


class FeatureSelectionProblem(Problem):
    def __init__(self, X_train, y_train, y_train_multiclass, X_test, y_test, n_features, f_min, f_max, n_hp, performance_tracker, clf_name, feature_names, neuron_min, neuron_max, hidden_min, hidden_max, cores=1, fitness_metric='auc', device='cuda', **kwargs):                
        super().__init__(
            n_var=n_features + n_hp, 
            n_obj=1, 
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
        self.performance_tracker = performance_tracker
        self.clf_name = clf_name
        self.feature_names = feature_names
        self.neuron_min = neuron_min
        self.neuron_max = neuron_max
        self.hidden_min = hidden_min
        self.hidden_max = hidden_max
        self.cores = cores
        self.fitness_metric = fitness_metric
        self.device = device

    def _evaluate(self, X, out, *args, **kwargs):
        with Pool(processes=self.cores) as pool:
            results = pool.starmap(evaluate_a_chromosome, [(x, self.X_train, self.y_train, self.y_train_multiclass, self.X_test, self.y_test, self.n_hp, self.clf_name, self.feature_names, self.fitness_metric, self.device) for x in X])
        
        objectives, y_preds = zip(*results)
        res = np.array(objectives, dtype=float)#.reshape(-1, 1)
        self.performance_tracker.save_best_per_generation(objectives=res, solutions=X, y_preds=y_preds)
        #self.performance_tracker.save_to_file()
        self.performance_tracker.plot_best_per_gen_with_n_features()
        out["F"] = res.reshape(-1, 1)
        #precision, recall = zip(*objectives)
        #print(res)
        #out["F"] = np.column_stack([precision, recall])
        

def evaluate_a_chromosome(x, X_train, y_train, y_train_multiclass, X_test, y_test, n_hp, clf_name, feature_names, fitness_metric, device = 'cuda'):
    x_decoded = XDecode(x=x, n_hp=n_hp, clf_name=clf_name, feature_names=feature_names)
    threshold = 0.3 # threshold for binary classification (mlp and td)
    
    X_train_selected = X_train.iloc[:, x_decoded.features]
    X_test_selected = X_test.iloc[:, x_decoded.features]

    # in the case where no features are selected
    if not np.any(x_decoded.features):
        return 1

    if clf_name == 'rf':
        clf = RandomForestClassifier(
            n_estimators=x_decoded.n_estimators,
            max_depth=x_decoded.max_depth,
            random_state=42,
            # n_jobs=-1,
            class_weight='balanced'
        )
    elif clf_name == 'dt':
        clf = DecisionTreeClassifier(
            max_depth=x_decoded.max_depth,
            random_state=42,
            class_weight='balanced'
        )
    elif clf_name == 'xgb':
        count_class_0, count_class_1 = y_train.value_counts()
        scale_pos_weight = count_class_0 / count_class_1

        clf = XGBClassifier(
            n_estimators=x_decoded.n_estimators,
            max_depth=x_decoded.max_depth,
            random_state=42, 
            tree_method='hist', 
            device=device, 
            scale_pos_weight=scale_pos_weight
            )
    elif clf_name == 'ada':
        clf = AdaBoostClassifier(
            n_estimators=x_decoded.n_estimators,
            random_state=42
        )
    elif clf_name == 'bag':
        clf = BaggingClassifier(
            n_estimators=x_decoded.n_estimators,
            random_state=42
        )
    elif clf_name == 'svm':
        clf = SVC(
            kernel='rbf',
            C=x_decoded.C,
            gamma=x_decoded.gamma,
        )
    elif clf_name == 'mlp':
        torch.cuda.set_device(device)
        layers = [x_decoded.n_neurons] * x_decoded.n_layers
        # Get the input dimension
        input_dim = X_train_selected.shape[1]
        # Create and move the model to the specified device
        clf = esp_utilities.MLP(
            input_dim=input_dim, 
            num_neurons=x_decoded.n_neurons, 
            num_layers=x_decoded.n_layers
            )
        clf.to(device)
        esp_utilities.train_mlp(X_train_selected.to_numpy(), y_train.to_numpy(), model=clf, device=device, y_train_multiclass=y_train_multiclass) # train the model
        probs = esp_utilities.get_probabilities(clf, X_test_selected) # get the probabilities

        if fitness_metric == 'auc':
            y_pred = probs
            fitness_score = roc_auc_score(y_test, y_pred)
        else:
            y_pred = (probs > threshold).astype(int) # get the predictions
            fitness_score = f1_score(y_test, y_pred, average='weighted')

        del clf
        torch.cuda.empty_cache()


    elif clf_name == 'td':
        layers = [x_decoded.n_neurons] * x_decoded.n_layers
        # X_train_numpy = np.column_stack((X_train_selected.to_numpy(), y_train))
        # X_test_numpy = np.column_stack((X_test_selected.to_numpy(), y_test))
        y_name = 'label'

        # train_df = pd.DataFrame(X_train_numpy, columns=x_decoded.selected_feature_names + [y_name])
        # test_df = pd.DataFrame(X_test_numpy, columns=x_decoded.selected_feature_names + [y_name])

        train_df = pd.concat([X_train_selected, y_train], axis=1)
        test_df = pd.concat([X_test_selected, y_test], axis=1)

        cont_names = x_decoded.selected_feature_names   # Since all features are numerical, I will only have continuous features
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
        # clf = tabular_learner(dls, layers=layers, metrics=f1_score_binary)
        clf = tabular_learner(dls, layers=layers)
        clf.model.to(torch.device(device))

        # Train the model
        clf.fit_one_cycle(100, 0.001, cbs=[EarlyStoppingCallback(monitor='valid_loss', min_delta=0, patience=5)])

        # Get the predictions
        probs, targs = clf.get_preds(dl=dls_test[0])

        if fitness_metric == 'auc':
            y_pred = probs[:, 1].numpy()  # Probabilities of class 1, positive class
            fitness_score = roc_auc_score(y_test, y_pred)
        else:
            y_pred = (probs[:, 1] > threshold).int().numpy()
            fitness_score = f1_score(y_test, y_pred, average='weighted')

    elif clf_name == 'nb':
        clf = GaussianNB()
    
    if clf_name in ['dt', 'rf', 'xgb', 'ada', 'bag']:
        clf.fit(X_train_selected, y_train)

        if fitness_metric == 'auc':
            y_pred = clf.predict_proba(X_test_selected)[:, 1]
            fitness_score = roc_auc_score(y_test, y_pred)
        else:
            y_pred = clf.predict(X_test_selected)
            fitness_score = f1_score(y_test, y_pred, average='weighted')

    #recall = recall_score(y, y_pred)
    #precision = precision_score(y, y_pred)

    return (-fitness_score, y_pred) #-recall, -precision


class XDecode():
    def __init__(self, x, n_hp, clf_name, feature_names, eval=False):
        self.features = self.get_features(x, initial_position=n_hp, eval=eval)
        self.selected_feature_names = self.get_selected_features(feature_names)
        if clf_name in ['rf', 'xgb']:
            self.max_depth = self.get_first_hp(x)
            self.n_estimators = self.get_second_hp(x)
        elif clf_name in ['dt']:
            self.max_depth = self.get_first_hp(x)
        elif clf_name in ['ada', 'bag']:
            self.n_estimators = self.get_first_hp(x)
        elif clf_name == 'svm':
            self.C = self.get_c(x)
            self.gamma = self.get_gamma(x)
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

    def get_c(self, x):
        val = x[0]
        min_int = 0
        max_int = 65535
        min_real = 0.0001
        max_real = 10000.0
        return min_real + ((val - min_int) / (max_int - min_int)) * (max_real - min_real)
    
    def get_gamma(self, x):
        val = x[1]
        min_int = 0
        max_int = 65535
        min_real = 0.00002
        max_real = 0.2
        return min_real + ((val - min_int) / (max_int - min_int)) * (max_real - min_real)
    
    def get_selected_features(self, feature_names):
        selected_feature_names = [feature_names[i] for i in self.features]
        return selected_feature_names


#class MyCallback(Callback):
#    def __init__(self) -> None:
#        super().__init__()
#        self.best = []
#        self.best_f_per_gen = []

#    def notify(self, algorithm):
#        self.best.append(algorithm.opt[0].F)
#        self.best_f_per_gen.append(algorithm.pop.get('F').min())


class MyCallback(Callback):

    def __init__(self, config_name):
        super().__init__()
        self.config_name = config_name

    def __call__(self, algorithm):
        current_gen = algorithm.n_gen
        create_checkpoint = not (current_gen + 1) % 10 #checkpoint every 10th gen
        if create_checkpoint:
            checkpoint_file = "{}_checkpoint_gen_{}.pkl".format(self.config_name, current_gen)
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({'algorithm': algorithm, 'current_gen': current_gen}, f)
            self.last_checkpoint_gen = current_gen
            print(f"Checkpoint saved for generation {current_gen}")


class PerformanceTracker():
    def __init__(self, execution_name, clf_name, n_hp, fitness_metric='auc', y_true=None, y_multi_class=None) -> None:
        self.n_hp = n_hp
        self.clf_name = clf_name
        self.y_true = y_true
        self.y_multi_class = y_multi_class
        self.execution_name = execution_name
        self.best_per_generation = []
        self.solution_outcomes = []
        self.best_result_per_generation = []
        self.best_y_pred_per_generation = []
        self.fitness_metric = fitness_metric

    def save_best_per_generation(self, objectives, solutions, y_preds):
        best_obj = objectives.min()
        best_index = objectives.argmin()
        best_solution = solutions[best_index]
        self.best_y_pred_per_generation.append(y_preds[best_index])
        best_solution = np.append(best_solution, [best_index, best_obj])
        self.best_per_generation.append(best_solution)
        self.best_result_per_generation.append(best_obj)

    def save_solution_outcomes(self, objectives, solutions):
        _objectives = objectives.reshape(-1, 1)
        _solutions = np.hstack((solutions, _objectives))
        self.solution_outcomes.append(_solutions)


    def plot_best_per_gen(self):
        generations = range(1, len(self.best_result_per_generation) + 1)
        plt.plot(generations, self.best_result_per_generation, marker='o')
        plt.xlabel('Generation')
        plt.ylabel('{} Score' .format(self.fitness_metric.upper()))
        plt.title('Best objective value Per Generation')
        plt.grid(True)

        plt.savefig(self.execution_name+ '_plot', format='png')  # saves the plot to file
        #plt.show()

    def plot_best_per_gen_with_n_features(self, show=False):
        plt.clf()
        plt.figure(figsize=(15,6))
        matrix = np.vstack(self.best_per_generation)
        features = matrix[:, self.n_hp:-2]
        n_features = np.sum(features != -1, axis=1)
        fitness_score = 1 - np.abs(matrix[:, -1])

        plt.plot(fitness_score, marker='o', linestyle='-', color='green', label=self.fitness_metric.upper() + ' Score')
        for i, txt in enumerate(n_features):
            plt.annotate(txt, (i, fitness_score[i]), xytext=(0,-15), textcoords='offset points', fontsize=6)

        plt.xticks(np.arange(0, len(fitness_score), step=1), fontsize=8, rotation=90)

        plt.grid(which='both', linestyle='-', linewidth=0.1, color='gray')

        plt.xlabel('Generation')
        plt.ylabel('Loss - Relative to {} Score' .format(self.fitness_metric.upper()))
        plt.title('{} - Fitness Convergence' .format(self.clf_name.upper()))

        if show:
            plt.show()
        else:
            plt.savefig(self.execution_name + '_plot.png', format='png')  #saves the plot to file
        plt.close()

    def save_to_file(self):
        best_per_generation = np.vstack(self.best_per_generation)
        best_per_generation_y_pred = np.vstack(self.best_y_pred_per_generation)
        np.save(self.execution_name + '_best_per_generation', best_per_generation)
        np.save(self.execution_name + '_best_per_generation_y_pred', best_per_generation_y_pred)

    def classification_stats(self, n_gen):
        # Initialize
        y_pred = self.best_y_pred_per_generation[n_gen]
        # Create a dataframe from the data
        data = pd.DataFrame({
            'y_true': self.y_true,
            'y_pred': y_pred,
            'Traffic Class': self.y_multi_class,
        })

        # Create a function to apply on each group
        def group_classification(group):
            right_classification = sum(group['y_true'] == group['y_pred'])
            missclassification = len(group) - right_classification
            percentage_of_right = right_classification / len(group) #* 100

            return pd.Series({
                'Missclassification': missclassification,
                'Right Classification': right_classification,
                'Accuracy': percentage_of_right
            })

        # Apply function on each group
        counts = data.groupby('Traffic Class').apply(group_classification)
        counts.reset_index(inplace=True)
            
        return counts.sort_values('Traffic Class')
    

def f1_score_binary(inputs, targets):
    # Move tensors to CPU and convert to numpy arrays
    inputs = inputs.argmax(dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    return f1_score(targets, inputs, average='binary', zero_division=0)

def auc_score_binary(inputs, targets):
    # Move tensors to CPU and convert to numpy arrays
    # inputs = inputs[:, 1].cpu().numpy()
    print('******* type *******')
    print(type(inputs))
    inputs = inputs[:, 1].numpy()  
    targets = targets.cpu().numpy()

    return roc_auc_score(targets, inputs)


def stratified_splitter(df, stratify_col, valid_pct=0.2):
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=valid_pct, random_state=42)
    for train_index, valid_index in stratified_split.split(df, stratify_col):
        return list(train_index), list(valid_index)
