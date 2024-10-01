import pickle
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from IPython.display import display, HTML
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from itertools import combinations
from icecream import ic
from xgboost import XGBClassifier

def fix_seeds(seed=42):
    """
    Fix random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model_effort(model):
    try:
        if hasattr(model, 'estimators_'):  # Para modelos de ensemble
            return sum(tree.tree_.node_count for tree in model.estimators_)
        elif hasattr(model, 'tree_'):  # Para uma única árvore
            return model.tree_.node_count
        elif hasattr(model, 'get_booster'):  # Para modelos XGBoost
            return sum(tree.count('\n') for tree in model.get_booster().get_dump())
        elif hasattr(model, 'model'):  # Para modelos FastAI
            return sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        elif hasattr(model, 'parameters'):  # Para MLP ou modelos PyTorch genéricos
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            raise ValueError(f"Model of type {type(model).__name__} is not supported by get_model_effort.")
    except Exception as e:
        print(f"Error calculating model effort: {e}")
        raise 

def get_ensemble_effort(clf_dt, clf_rf, clf_xgb, clf_mlp, clf_td, baseline_effort=[5167, 406984, 7264, 25301, 1079908]):
    """
    Get the normalized effort of an ensemble model based on the sum of the efforts of its base models.
    parameters:
    clf_dt (obj): Decision Tree model object.
    clf_rf (obj): Random Forest model object.
    clf_xgb (obj): XGBoost model object.
    clf_mlp (obj): MLP model object.
    clf_td (obj): TabNet model object.
    baseline_effort (list): List of efforts of the base models.
    """
    ensemble_effort = [get_model_effort(clf_dt), get_model_effort(clf_rf), get_model_effort(clf_xgb), get_model_effort(clf_mlp), get_model_effort(clf_td)]
    normalized_ensemble_effort = sum(ensemble_effort)/sum(baseline_effort)
    return normalized_ensemble_effort, ensemble_effort

def disagreement(predictions_1, predictions_2):

    assert len(predictions_1) == len(predictions_2), "The number of predictions must be the same for both classifiers."
    
    N = len(predictions_1)
    disagree_count = sum(p1 != p2 for p1, p2 in zip(predictions_1, predictions_2))
    
    return disagree_count / N

def ensemble_disagreement(predictions):
    classifier_pairs = combinations(predictions.keys(), 2)
    total_disagreement = 0
    pair_count = 0

    for classifier1, classifier2 in classifier_pairs:
        pred1, pred2 = predictions[classifier1], predictions[classifier2]
        
        disagreement_value = disagreement(pred1, pred2)

        total_disagreement += disagreement_value
        pair_count += 1

    return total_disagreement / pair_count if pair_count > 0 else 0

class MLP(nn.Module):
    def __init__(self, input_dim, num_neurons=100, num_layers=3):
        super(MLP, self).__init__()

        # Assegurar que há pelo menos uma camada
        if num_layers < 1:
            raise ValueError("Número de camadas deve ser pelo menos 1")

        self.layers = nn.ModuleList()

        # Primeira camada oculta
        self.layers.append(nn.Linear(input_dim, num_neurons))

        # Camadas ocultas adicionais
        for _ in range(1, num_layers):
            self.layers.append(nn.Linear(num_neurons, num_neurons))

        # Camada de saída
        self.output_layer = nn.Linear(num_neurons, 1)

    def forward(self, x):
        # Passagem pelas camadas ocultas
        for layer in self.layers:
            x = torch.relu(layer(x))

        # Passagem pela camada de saída
        x = self.output_layer(x)
        return x
    
def train_mlp(X_train, Y_train, model, num_epochs=100, learning_rate=0.001, batch_size=32, patience=5, val_ratio=0.2, verbose=False, device='cuda', y_train_multiclass=None):
    """
    Train a multi-layer perceptron (MLP) model with early stopping.

    Parameters:
    X_train (numpy.ndarray): Training features.
    Y_train (numpy.ndarray): Training labels.
    model (nn.Module): MLP model to be trained.
    num_epochs (int): Number of epochs for training.
    learning_rate (float): Learning rate for the optimizer.
    batch_size (int): Batch size for data loading.
    patience (int): Number of epochs to wait for improvement before stopping.
    val_ratio (float): Ratio of training data to be used as validation data.

    Returns:
    None
    """
    # Splitting training data into training and validation sets
    if y_train_multiclass is not None:
        X_train, X_val, Y_train, Y_val, _, _ = train_test_split(X_train, Y_train, y_train_multiclass, test_size=val_ratio, stratify=y_train_multiclass, random_state=42)
    else:
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_ratio, stratify=Y_train, random_state=42)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val = torch.tensor(Y_val, dtype=torch.float32).to(device)

    # Reshape Y if necessary
    Y_train = Y_train.view(Y_train.shape[0], 1)
    Y_val = Y_val.view(Y_val.shape[0], 1)

    # Create datasets and data loaders
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping variables
    best_loss = float('inf')
    epochs_no_improve = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        val_loss /= len(val_loader)

        if verbose:
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')

        # Check for early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                if verbose:
                    print("Early stopping triggered")
                break
    if verbose:
        print("Training complete!")

import os
def load_trained_mlp_model(input_dim, num_layers=3, num_neuron=100, device='cuda', ga=False):
    """
    Carrega o modelo MLP pré-treinado.
    
    Parâmetros:
    input_dim (int): Número de recursos de entrada.
    num_layers (int): Número de camadas no MLP.
    num_neuron (int): Número de neurônios por camada.
    device (str ou torch.device): Dispositivo para carregar o modelo ('cpu' ou 'cuda').
    ga (bool): Se deve carregar um modelo otimizado por GA.
    
    Retorna:
    nn.Module: Modelo MLP pré-treinado.
    """
    if isinstance(device, str):
        device = torch.device(device)

    # Diretório base do arquivo atual (esp_utilities.py)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Diretório do projeto
    project_root = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir, 'source'))

    # Caminho do diretório de modelos relativo ao diretório base
    model_dir = os.path.join(project_root, 'models')

    # Construção do caminho do arquivo do modelo
    if ga:
        model_path = os.path.join(model_dir, 'mlp_pytorch_ga_usnw-nb15.pth')
    else:
        model_path = os.path.join(model_dir, 'mlp_pytorch_default_usnw-nb15.pth')

    print(f"Carregando modelo de: {model_path}")

    # Instância do modelo MLP
    model = MLP(input_dim=input_dim, num_neurons=num_neuron, num_layers=num_layers)
    
    # Carregar os pesos do modelo pré-treinado
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Definir o modo de avaliação e mover para o dispositivo
    model.eval()
    model.to(device)

    return model

def get_probabilities(model, X, device='cuda', return_gpu_time=False):
    """
    Get the probabilities of the positive class for each sample in the dataset.
    Optionally, measure and return the GPU execution time.
    
    Parameters:
    - model (torch.nn.Module): Pre-trained model.
    - X (pd.DataFrame): Input features.
    - device (str): The device to use ('cuda' for GPU or 'cpu').
    - return_gpu_time (bool): Whether to return the GPU execution time.

    Returns:
    - probabilities (pd.Series): Probabilities of the positive class for each sample.
    - gpu_time_ms (float, optional): GPU execution time in milliseconds, if requested.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
        
        if return_gpu_time:
            # Creating CUDA events for timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start_event.record()

        outputs = model(X_tensor)
        probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
        
        if return_gpu_time:
            end_event.record()
            torch.cuda.synchronize()
            gpu_time_ms = start_event.elapsed_time(end_event)
            return pd.Series(probabilities), gpu_time_ms
        
        return pd.Series(probabilities)

class DatasetLoader:
    """
    A utility class for loading, preprocessing, and managing specific datasets for binary or multiclass classification.

    This class automatically loads the specified dataset upon instantiation, separates it into features and labels,
    and provides the option to scale the feature data. It allows the selection of either all features in the dataset
    or a specific set of mixed features defined by the class. The scaling of data can be done either using only the
    training data or the combined training and test data.

    Class Attributes:
        time_based_features (list): List of time-based features.
        volume_based_features (list): List of volume-based features.
        mixed_features (list): Combined list of time and volume-based features.

    Instance Attributes:
        X_train (pandas.DataFrame): Training feature data.
        y_train (pandas.Series): Training labels for binary classification.
        y_train_multiclass (pandas.Series): Training labels for multiclass classification.
        X_test (pandas.DataFrame): Testing feature data.
        y_test (pandas.Series): Testing labels for binary classification.
        y_test_multiclass (pandas.Series): Testing labels for multiclass classification.
        scaler (MinMaxScaler): An instance of MinMaxScaler, used if scaling is applied.
    """


    time_based_features = [
        'activeTimeMaxMilliseconds', 'activeTimeMeanMilliseconds', 'interPacketTimeMillisecondsStdev',
        'interPacketTimeSecondsStdev', 'idleMeanMilliseconds', 'interPacketTimeSecondsSumFwd',
        'interPacketTimeMillisecondsMax', 'idleMaxMilliseconds', 'interPacketTimeSecondsMax',
        'activeTimeStdMilliseconds', 'interPacketTimeMillisecondsSum', 'interPacketTimeSecondsMean',
        'interPacketTimeSecondsStdevFwd', 'flowDurationMilliseconds', 'idleStdMilliseconds',
        'interPacketTimeMillisecondsSumFwd', 'interPacketTimeSecondsMaxFwd', 'interPacketTimeSecondsMeanFwd',
        'interPacketTimeSecondsMaxBwd', 'interPacketTimeSecondsStdevBwd', 'interPacketTimeSecondsMeanBwd',
        'interPacketTimeMilliseconds', 'bwdJitterMilliseconds', 'interPacketTimeSecondsSumBwd'
        ]
    
    volume_based_features = [
        'minimumIpTotalLength', 'minimumIpTotalLengthFwd', 'minimumIpTotalLengthBwd','bwdBytesAvg',
        'fwdBytesAvg', 'ipTotalLengthMeanBwd', 'ipTotalLengthMeanFwd', 'packetLenAvg', 'ipTotalLengthMean',
        'maximumIpTotalLengthFwd', 'maximumIpTotalLength', 'maximumIpTotalLengthBwd', 'ipTotalLengthStdevBwd',
        'octetTotalCountFwd', 'ipTotalLengthFwd', 'ipTotalLengthBwd', 'octetTotalCountBwd', 'packetLen',
        'ipTotalLength', 'octetTotalCount', 'ipTotalLengthStdevFwd', 'bwdBytesPerMicroseconds', 'ipTotalLengthVar',
        'ipTotalLengthStdev', 'flowBytesByMicroseconds'
        ]
    
    mixed_features = list(set(time_based_features+volume_based_features))

    def __init__(self, dataset_name, selected_features=True, scale_data=False, scale_on_full_dataset=False, toy=False):
        """
        Initializes the DatasetLoader with the given dataset.

        Parameters:
            dataset_name (str): Name of the dataset to be loaded.
            selected_features (bool): Whether to use selected (mixed) features or all features.
            scale_data (bool): Whether to scale the feature data or not.
            scale_on_full_dataset (bool): If True, scale the data based on the combined training and test datasets.
                                         If False, scale only based on the training dataset.
            toy (bool): If True, load a toy dataset with balanced attack and normal instances.
        """
        self.dataset_paths = {
            'unsw-nb15': ('../datasets/unsw-nb15_train_day_22.csv.gz', 
                          '../datasets/unsw-nb15_test_day_17.csv.gz'),
            'bot-iot': ('path/to/bot-iot/train.csv', 'path/to/bot-iot/test.csv'),
            'cicids-2017': ('path/to/cicids-2017/train.csv', 'path/to/cicids-2017/test.csv')
        }

        if dataset_name not in self.dataset_paths:
            raise ValueError(f"Dataset {dataset_name} not recognized.")

        train_path, test_path = self.dataset_paths[dataset_name]

        # Load datasets with selected features or all features
        if selected_features:
            cols_to_use = self.mixed_features + ['attackCategory', 'label']
        else:
            cols_to_use = None  # Load all columns

        # Load the dataset with balanced true and false instances if it is a toy dataset
        if toy:
            full_train_df = pd.read_csv(train_path, usecols=cols_to_use)
            full_test_df = pd.read_csv(test_path, usecols=cols_to_use)

            # Selecione todas as instâncias de ataque
            attacks_train = full_train_df[full_train_df['label'] == 1]
            attacks_test = full_test_df[full_test_df['label'] == 1]

            # Selecione um número igual de instâncias normais
            num_attacks_train = len(attacks_train)
            num_attacks_test = len(attacks_test)

            normals_train = full_train_df[full_train_df['label'] == 0].sample(n=num_attacks_train)
            normals_test = full_test_df[full_test_df['label'] == 0].sample(n=num_attacks_test)

            # Combine e embaralhe os dados
            train_df = pd.concat([attacks_train, normals_train]).sample(frac=1).reset_index(drop=True)
            test_df = pd.concat([attacks_test, normals_test]).sample(frac=1).reset_index(drop=True)

            # Estratificação de 10% do conjunto de treino
            train_df, _ = train_test_split(
                train_df, 
                test_size=0.97, 
                stratify=train_df['attackCategory'], 
                random_state=42  # Use um valor constante para reprodutibilidade
            )

            # Estratificação de 10% do conjunto de teste
            test_df, _ = train_test_split(
                test_df, 
                test_size=0.99, 
                stratify=test_df['attackCategory'], 
                random_state=42  # Use um valor constante para reprodutibilidade
            )

        else:
            train_df = pd.read_csv(train_path, usecols=cols_to_use)
            test_df = pd.read_csv(test_path, usecols=cols_to_use)

        # Fixing a typo in the dataset
        train_df['attackCategory'] = train_df['attackCategory'].apply(lambda x: x.replace("Backdoorss", "Backdoors"))
        test_df['attackCategory'] = test_df['attackCategory'].apply(lambda x: x.replace("Backdoorss", "Backdoors"))
        
        # Separating features from labels
        self.X_train = train_df.drop(['label', 'attackCategory'], axis=1)
        self.y_train = train_df['label']
        self.y_train_multiclass = train_df['attackCategory']
        self.X_test = test_df.drop(['label', 'attackCategory'], axis=1)
        self.y_test = test_df['label']
        self.y_test_multiclass = test_df['attackCategory']

 
        if scale_data:
            self.scaler = MinMaxScaler()
            if scale_on_full_dataset:
                self._scale_on_full_dataset(train_df, test_df)
            else:
                self._scale_on_train_dataset()
        else:
            self.scaler = None

    def _scale_on_train_dataset(self):
        """Scales the training data and applies the same transformation to the test data."""
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=self.X_train.columns)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=self.X_test.columns)

    def _scale_on_full_dataset(self, train_df, test_df):
        """Scales the combined training and test data."""
        full_dataset = pd.concat([train_df.drop(['label', 'attackCategory'], axis=1), 
                                  test_df.drop(['label', 'attackCategory'], axis=1)])
        full_scaled = pd.DataFrame(self.scaler.fit_transform(full_dataset), columns=full_dataset.columns)

        # Splitting the scaled data back into training and test sets
        self.X_train = full_scaled.iloc[:len(self.X_train)]
        self.X_test = full_scaled.iloc[len(self.X_train):]



    def get_full_train_set(self):
        """
        Returns the concatenated training dataset including features and both binary and multiclass labels.

        Returns:
            pandas.DataFrame: The full training dataset.
        """
        return pd.concat([self.X_train, self.y_train, self.y_train_multiclass], axis=1)

    def get_full_test_set(self):
        """
        Returns the concatenated testing dataset including features and both binary and multiclass labels.

        Returns:
            pandas.DataFrame: The full testing dataset.
        """
        return pd.concat([self.X_test, self.y_test, self.y_test_multiclass], axis=1)
    
    def get_stratified_sample(self, dataset='train', sample_size=0.2):
        """
        Returns a stratified sample of the specified dataset.

        Parameters:
            dataset (str): Specifies which dataset to sample from ('train' or 'test').
            sample_size (float): The proportion of the dataset to include in the sample (between 0 and 1).

        Returns:
            (DataFrame, Series, Series): Sampled feature data (X), binary labels (y), and multiclass labels (y_multiclass).
        """
        if dataset not in ['train', 'test']:
            raise ValueError("Dataset must be 'train' or 'test'.")

        if dataset == 'train':
            X = self.X_train
            y = self.y_train
            y_multiclass = self.y_train_multiclass
        else:
            X = self.X_test
            y = self.y_test
            y_multiclass = self.y_test_multiclass

        # Ensuring sample size is a float and within a valid range
        sample_size = float(sample_size)
        if not 0 < sample_size <= 1:
            raise ValueError("Sample size must be between 0 and 1.")

        # Stratified sampling based on multiclass labels
        _, X_sample, _, y_sample = train_test_split(
            X, y, 
            test_size=sample_size, 
            stratify=y_multiclass, 
            random_state=42
        )

        y_multiclass_sample = y_multiclass.loc[X_sample.index]

        return X_sample, y_sample, y_multiclass_sample

def stratified_splitter(df, stratify_col, valid_pct=0.2):
    """
    Perform a stratified split of the dataset into training and validation sets.
    Parameters:
    df (pd.DataFrame): The dataset to be split.
    stratify_col (str): The column to use for stratification.
    valid_pct (float): The proportion of the dataset to include in the validation set.
    Returns:
    Tuple[list, list]: The indices of the training and validation sets.
    """
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=valid_pct, random_state=42)
    for train_index, valid_index in stratified_split.split(df, stratify_col):
        return list(train_index), list(valid_index)

def feature_importance_ranking(estimator, X, y, y_multiclass, perm_max_samples=1.0):
    """
    Function to rank features based on permutation importance using sklearn with optional stratified sampling.

    :param estimator: The trained estimator (model).
    :param X: Pandas DataFrame containing the features.
    :param y: Pandas Series containing the target variable.
    :param perm_max_samples: Proportion of samples to use for permutation importance. 
                             If less than 1, uses stratified sampling based on this proportion.
    :return: DataFrame with features ranked by their importance.
    """
    if perm_max_samples < 1:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=perm_max_samples, random_state=42)

        # Get the indices for the single split
        _, sample_index = next(splitter.split(X, y_multiclass))

        # Use the indices to create the sample datasets
        X_sample = X.iloc[sample_index]
        y_sample = y.iloc[sample_index]

    else:
        X_sample = X
        y_sample = y

    # Calculate permutation importance
    result = permutation_importance(estimator, X_sample, y_sample, n_repeats=2, random_state=42, scoring='f1')

    # Create a DataFrame to hold feature importances
    importance_df = pd.DataFrame({'feature': X.columns, 'importance': result.importances_mean})

    # Sort the DataFrame by importance in descending order
    importance_df.sort_values(by='importance', ascending=False, inplace=True)
    importance_df.reset_index(drop=True, inplace=True)

    return importance_df

def save_to_pickle(data, filename):
    """
    Save a Python object to a pickle file.
    
    Parameters:
    - data: The Python object to be saved.
    - filename: The path and name of the file where the object will be saved.
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def load_from_pickle(filename):
    """
    Load a Python object from a pickle file.
    
    Parameters:
    - filename: The path and name of the file from which the object will be loaded.
    
    Returns:
    - The loaded Python object.
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

# def load_carray_dataset(path, cols, df=None):
#     """
#     Load dataset and extract features and labels.

#     Args:
#     - path (str): Path to the dataset.
#     - cols (list): Columns to be used.
#     - df (pd.DataFrame, optional): A DataFrame to use instead of loading from a file.

#     Returns:
#     - X (CArray): Feature matrix.
#     - y (CArray): Label vector.
#     """
#     if df is None:
#         df = pd.read_csv(path, usecols=cols)
#     X = CArray(df.drop(['attackCategory', 'label'], axis=1).values)
#     y = CArray(df['label'].values)
#     y_multi_class = df['attackCategory']
#     return X, y, y_multi_class

def load_realistic_attacks(data_loader, factors=[f for f in range(-90,101, 10)], minmax_scale=True):
    """
    Load the realistic attacks datasets.
    parameters:
    factors (list): List of factors to load.
    data_loader (DatasetLoader): DatasetLoader object from esp_utilities. Only for unsw-nb15 dataset.

    """
    path = '../datasets/realistic_attacks/'
    features = data_loader.X_train.columns.to_list()
    cols = features + ['label', 'attackCategory']
    scaler = data_loader.scaler

    datasets = {}
    for attack in ['time', 'volume', 'time_volume']:
        datasets_by_factor = {}
        for factor in factors:
            if factor == 0:
                df = data_loader.get_full_test_set()
            else:
                if attack == 'time':
                    file = path + 'unsw-nb15_30min_'+ str(factor) +'x_slow_day_17.csv.gz'
                elif attack == 'volume':
                    file = path + 'unsw-nb15_30min_'+ str(factor) +'percent_payload_day_17.csv.gz'
                elif attack == 'time_volume':
                    file = path + 'unsw-nb15_30min_'+ str(factor) +'_time_volume_day_17.csv.gz'
            
                df = pd.read_csv(file, compression='gzip', usecols=cols)

                if minmax_scale:
                    df[features] = scaler.transform(df[features])
            datasets_by_factor[factor] = df

        datasets[attack] = datasets_by_factor
    return datasets

# def train_sec_svm_classifier(dataset_path, features):
#     """
#     Train an SVM classifier.

#     Args:
#     - dataset_path (str): Path to the training dataset.
#     - features (list): Features to be used.

#     Returns:
#     - classifier: Trained SVM classifier.
#     """
#     X, y, _ = load_carray_dataset(dataset_path, features)
#     print('Training SVM...')
#     classifier = CClassifierSVM(kernel='rbf', C=1.0, n_jobs=-1)
#     classifier.fit(X, y)
#     return classifier

def compare_attackwise_accuracies(model, df_baseline, df_manipulated):
    """
    Evaluate attackwise accuracy of the given model on baseline and manipulated datasets.

    Args:
    - model (someModelType): Trained model to evaluate.
    - df_baseline (pd.DataFrame): Baseline dataset.
    - df_manipulated (pd.DataFrame): Manipulated dataset.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: Attackwise accuracy for baseline and manipulated datasets.
    """
    
    # Load the baseline dataset
    X_baseline, y_baseline, y_baseline_multi = load_carray_dataset(None, None, df=df_baseline)
    
    # Load the manipulated dataset
    Xm, ym, ym_multi = load_carray_dataset(None, None, df=df_manipulated)

    # Predict using baseline data
    y_pred_baseline = model.predict(X_baseline)
    df_baseline_accuracy = compute_attackwise_accuracy(y_baseline, y_pred_baseline, y_baseline_multi)

    # Predict using manipulated data
    y_pred_manipulated = model.predict(Xm)
    df_manipulated_accuracy = compute_attackwise_accuracy(ym, y_pred_manipulated, ym_multi)

    return df_baseline_accuracy, df_manipulated_accuracy


def compute_attackwise_accuracy(y_test, y_pred, y_multi_class):
    """
    Compute and return the accuracy of predictions, attack-wise.

    Given the true labels, predicted labels, and corresponding attack categories,
    this function calculates the accuracy of predictions for each unique attack category.
    Additionally, it provides counts of total instances, correctly classified instances,
    and misclassified instances per category. The results are returned as a DataFrame
    sorted by total instances per category in descending order.

    Parameters:
    - y_test (array-like): True labels of the test set.
    - y_pred (array-like): Predicted labels of the test set.
    - y_multi_class (array-like): Corresponding attack categories of each instance in the test set.

    Returns:
    - pd.DataFrame: A DataFrame containing the following columns:
        - 'category': Unique attack categories.
        - 'accuracy': Accuracy of predictions within each category.
        - 'total_instances': Total number of instances per category.
        - 'correctly_classified': Number of instances correctly classified per category.
        - 'misclassified': Number of instances misclassified per category.
      The DataFrame is sorted by 'total_instances' in descending order.
    """
    df = pd.DataFrame({
        'attackCategory': y_multi_class.tolist(),
        'label': y_test.tolist(),
        'pred': y_pred.tolist()
    })

    unique_categories = df['attackCategory'].unique()
    
    results = []
    for category in unique_categories:
        # Filter rows for current category
        df_filtered = df[df['attackCategory'] == category]
        
        # Calculate accuracy
        accuracy = (df_filtered['label'] == df_filtered['pred']).mean()
        
        # Count total instances
        total_instances = len(df_filtered)

        # Count correctly classified instances
        correctly_classified = sum(df_filtered['label'] == df_filtered['pred'])

        # Count misclassified instances
        misclassified = total_instances - correctly_classified

        # Append to results
        results.append({
            'category': category,
            'accuracy': accuracy,
            'total_instances': total_instances,
            'correctly_classified': correctly_classified,
            'misclassified': misclassified,
        })

    # Convert to DataFrame for easier visualization
    df_attackwise_accuracy = pd.DataFrame(results)

    # Sort by category in descending order
    df_attackwise_accuracy.sort_values('category', ascending=False, inplace=True)
    df_attackwise_accuracy.reset_index(drop=True, inplace=True)
    return df_attackwise_accuracy

def attack_evaluation(attack_datasets, nw_attack, models, feature_set, threshold=(0.5, 0.3), feature_set_dict = None, ensemble =False, device='cuda'):
    """
    Evaluate the evasion rate of the given models on the specified attack datasets.
    Parameters:
    - attack_datasets (dict): Dictionary containing the attack datasets.
    - nw_attack (str): The attack category to evaluate.
    - models (dict): Dictionary containing the trained models.
    - feature_set (list): List of features to use.
    - threshold (tuple): Tuple containing the thresholds for each model.
    - feature_set_dict (dict, optional): Dictionary containing the feature sets for each model.
    - ensemble (bool): Whether to evaluate the ensemble model.
    - device (str): The device to use ('cuda' for GPU or 'cpu').
    Returns:
    - dict: Dictionary containing the evasion rates for each model.
    - pd.DataFrame: DataFrame containing the evasion rates for each model.
    """
    # dict_keys(['time', 'volume', 'time_volume'])
    # dict_keys([-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    results = {}
    results_list = []
    _feature_set = feature_set.copy()
    for model_name, model in models.items():
        if feature_set_dict is not None:
            _feature_set = feature_set_dict[model_name].copy()
        results_by_attack = {}

        model_threshold = threshold[0] if model_name in ['dt', 'rf', 'xgb'] else threshold[1]

        for attack, datasets_by_factor in attack_datasets.items():
            results_by_factor = {}

            for factor, dataset in datasets_by_factor.items():
                _dataset = dataset.copy()
                _dataset = _dataset.loc[_dataset['attackCategory'] == nw_attack]

                if model_name in ['dt', 'rf', 'xgb']:
                    y_pred = model.predict(_dataset[_feature_set])
                    y_prob = model.predict_proba(_dataset[_feature_set])[:, 1]
                elif model_name == 'mlp':
                    y_pred, y_prob = predict_with_custom_threshold(model, _dataset[_feature_set], threshold=model_threshold, device=device, probs=True)
                elif model_name in ['tabular', 'td']:
                    y_pred = np.array(predict_with_custom_threshold_deep(model.model, _dataset, threshold=model_threshold, feature_set=_feature_set, device=device))
                    y_prob = np.array(predict_with_custom_threshold_deep(model.model, _dataset, threshold=model_threshold, feature_set=_feature_set, device=device, prob=True))    
                # elif 'resnet' in model_name:
                #     # Function fails without this series of ones; exact reason unknown, further analysis needed
                #     series_with_ones = pd.Series(np.ones(_dataset.shape[0]))
                #     print(type(_dataset[_feature_set]))
                #     y_prob = get_probabilities_resnet(model=model, X=_dataset[_feature_set], y=series_with_ones)
                #     y_pred = (y_prob > model_threshold).astype(float)
                # # Error
                else:
                    print('Classifier not found!')
                    break
                
                evasion_rate = compute_evasion_rate(np.ones(len(y_pred)), y_pred)
                
                results_by_factor[factor] = evasion_rate
                results_list.append([model_name, attack, factor, evasion_rate, y_pred, y_prob])

            results_by_attack[attack] = results_by_factor
        df_results = pd.DataFrame(results_list, columns=['model', 'attack', 'factor', 'asr', 'y_hat', 'y_prob'])
        results[model_name] = results_by_attack
    return results, df_results

def train_meta_classifier(X_train, y_train, clf_name='xgb'):
    if clf_name == 'dt':
        clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    elif clf_name == 'rf':
        clf = RandomForestClassifier(random_state=42,class_weight='balanced', n_jobs=-1)
    elif clf_name == 'xgb':
        count_class_0, count_class_1 = y_train.value_counts()
        scale_pos_weight = count_class_0 / count_class_1
        clf = XGBClassifier(random_state=42, tree_method='hist', scale_pos_weight=scale_pos_weight, n_jobs=-1)
    elif clf_name == 'lr':
        clf = LogisticRegression(random_state=42, class_weight='balanced', n_jobs=-1)
    ic(clf_name, clf, X_train.shape, y_train.shape)
    clf.fit(X_train, y_train.values.ravel())
    return clf


def evaluate_meta_classifiers(clf_name, X_train, y_train, X_test, return_clf=False):
    clf = train_meta_classifier(X_train, y_train, clf_name)
    y_pred = clf.predict(X_test)
    if return_clf:
        return y_pred, clf
    return y_pred

def rescale_probabilities(p):
    """
    Rescale probabilities from range [0.0 - 0.3] to [0.0 - 0.5] and [0.3 - 1.0] to [0.5 - 1.0].
    """
    if p >= 0.3:
        return 0.5 + (p - 0.3) / 1.4
    else:
        return p / 0.6

def max_rule(y_probs, resc_neural=False):
    """
    Apply the max rule to the predicted probabilities.
    y_probs: Dataframe with the predicted probabilities of each classifier (dt, rf, xgb, mlp dt).
    resc_neural: Whether to rescale the probabilities of the neural network.
    """
    _y_probs = y_probs.copy()
    if resc_neural:
        _y_probs['mlp'] = _y_probs['mlp'].apply(rescale_probabilities)
        _y_probs['td'] = _y_probs['td'].apply(rescale_probabilities)
    # Use the DataFrame max method to find the maximum value for each row
    return _y_probs.max(axis=1).to_numpy()


    return np.max(y_probs, axis=0)
def voting_strategy(all_preds, strategy, clf_name=None, train_probs=None, y_train=None, test_probs=None, return_clf=False):
    """
    Apply a specified voting strategy to the predictions of multiple classifiers.

    Parameters:
    - all_preds: A 2D numpy array where each row represents the predictions from a classifier.
    - strategy: The voting strategy to apply. Can be 'majority', 'unanimous', or 'any'.

    Returns:
    - np.ndarray: The final ensemble prediction.
    """
    features = ['dt', 'rf', 'xgb', 'mlp', 'td']
    if strategy == 'majority':
        y_vote = np.sum(all_preds, axis=0) > (len(all_preds) / 2)
    elif strategy == 'unanimous':
        y_vote = np.sum(all_preds, axis=0) == all_preds.shape[0]
    elif strategy == 'any':
        y_vote = np.any(all_preds, axis=0)
    elif strategy == 'stacking':
        
        if return_clf:
            y_vote, clf = evaluate_meta_classifiers(clf_name, train_probs[features], y_train, test_probs[features], return_clf=True)
            return y_vote.astype(int), clf
        y_vote = evaluate_meta_classifiers(clf_name, train_probs[features], y_train, test_probs[features])
    elif strategy == 'w_average' and test_probs is not None:
        weights = pd.Series({'dt': 0.5, 'rf': 0.5, 'xgb': 0.5, 'mlp': 0.3, 'td': 0.3})
        y_vote = np.average(test_probs[features], axis=1, weights=weights)
    elif strategy == 'max_rule' and test_probs is not None:
        y_vote = max_rule(test_probs, resc_neural=True)
    else:
        raise ValueError(f"Invalid strategy '{strategy}' is test_probs None: {test_probs is None}. Choose from 'majority', 'unanimous', 'any', 'stacking', or 'w_average'.")
    
    return y_vote.astype(int)
    
def compute_ensemble_attackwise_accuracy(y, y_multiclass, y_pred_dict, strategy='majority', clf_name=None, y_prob_dict=None, y_prob_train_dict=None, y_train=None):
    """
    Compute attackwise accuracy for each classifier and the ensemble.

    Parameters:
    - y (array-like): True labels of the test set.
    - y_multiclass (array-like): Corresponding attack categories of each instance in the test set.
    - y_pred_dict (dict): Dictionary containing predicted labels for each classifier.
    - y_prob_dict (dict, optional): Dictionary containing predicted probabilities for each classifier.

    Returns:
    - pd.DataFrame: Attackwise accuracy for each classifier and the ensemble.
    - If y_prob_dict is provided, the DataFrame will also include AUC scores.
    """
    results = {}
    all_preds = []
    for key, value in y_pred_dict.items():
        results[key] = compute_attackwise_accuracy(value, y, y_multiclass)
        all_preds.append(value)

    if strategy == 'stacking':
        y_vote, clf = voting_strategy(all_preds=np.array(all_preds), strategy=strategy, clf_name=clf_name, train_probs=pd.DataFrame(y_prob_train_dict), y_train=y_train, test_probs=pd.DataFrame(y_prob_dict), return_clf=True)
    else:
        y_vote = voting_strategy(all_preds=np.array(all_preds), strategy=strategy, test_probs=pd.DataFrame(y_prob_dict))
    results['ensemble'] = compute_attackwise_accuracy(y_vote, y, y_multiclass)

    first_cicle = True
    for key, value in results.items():
        if first_cicle:
            first_cicle = False
            df = value[['category', 'total_instances', 'accuracy']].copy()
            df.rename(columns={'accuracy': key + '_accuracy'}, inplace=True)
        else:
            df[key + '_accuracy'] = value['accuracy']

    auc_scores = []
    if y_prob_dict is not None:
        y_prob_dict['ensemble'] = np.mean(list(y_prob_dict.values()), axis=0)
        for key, value in y_prob_dict.items():
            fpr, tpr, _ = roc_curve(y, value)
            auc_score = auc(fpr, tpr)
            auc_scores.append(auc_score)

        # Creating the row to append
        new_row = pd.DataFrame([{
            'category': 'AUC',
            'total_instances': '-',
            'dt_accuracy': auc_scores[0],
            'rf_accuracy': auc_scores[1],
            'xgb_accuracy': auc_scores[2],
            'mlp_accuracy': auc_scores[3],
            'td_accuracy': auc_scores[4],
            'ensemble_accuracy': auc_scores[5]
        }])

        # Append the new row to the DataFrame
        df = pd.concat([df, new_row], ignore_index=True)

    if strategy == 'stacking':
        return df, clf
    return df

def compute_evasion_rate(y_true, y_pred):
    """
    Computes the evasion rate for a given set of predictions.
    
    Parameters:
        y_true (Series): The true labels.
        y_pred (Series): The predicted labels.
    
    Returns:
        float: The evasion rate.
    """
    # Compute the number of attacks that were successfully evaded
    num_attacks_evaded = ((y_true == 1) & (y_pred == 0)).sum()
    
    # Compute the number of attacks
    num_attacks = (y_true == 1).sum()
    
    # Compute the evasion rate
    evasion_rate = num_attacks_evaded / num_attacks
    
    return evasion_rate

def predict_with_custom_threshold(model, X, threshold=0.4, device='cuda', probs=False):
    """
    Predict labels for input data using a custom classification threshold.

    Parameters:
    model (nn.Module): Trained model for prediction.
    X (pd.DataFrame): Input features.
    threshold (float): Custom threshold for classification.

    Returns:
    pd.Series or torch.Tensor or eagerpy.tensor.pytorch.PyTorchTensor or original type: Predicted labels.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
        outputs = model(X_tensor)
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > threshold).float()  # Apply custom threshold
        predictions = pd.Series(predictions.cpu().numpy().flatten())
        if probs:
            # Verificar a forma de probabilities
            if probabilities.dim() == 2 and probabilities.size(1) == 2:
                # Probabilidades binárias com duas saídas (para cada classe)
                probabilities = probabilities[:, 1].cpu().numpy().flatten()
            else:
                # Probabilidades já para a classe positiva
                probabilities = probabilities.cpu().numpy().flatten()
            return predictions, probabilities
        return predictions
    
def predict_with_custom_threshold_deep(model, X, threshold=0.5, device='cpu', prob=False, feature_set=None):
    """
    Convert a Pandas DataFrame to two PyTorch tensors for inference, one for categorical data (empty in this case)
    """
    # Sem colunas categóricas - criar tensor vazio
    cat_tensor = torch.tensor([], dtype=torch.int64).reshape(0, 0).to(device)

    # Converter todas as colunas para tensor contínuo
    if feature_set is None:
        cont_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    else:
        cont_tensor = torch.tensor(X[feature_set].values, dtype=torch.float32).to(device)

    with torch.no_grad():
        model.eval()
        preds = model(cat_tensor, cont_tensor)[:, 1]

    probs = torch.sigmoid(preds)
    probs = pd.Series(probs.cpu().numpy().flatten())
    if prob:
        return probs
    
    predictions = (probs > threshold).astype(float)
    # predictions = (probs > threshold).float()
    # predictions = pd.Series(predictions.cpu().numpy().flatten())

    return predictions

def plot_boxplots(df, rank=None):
    """
    This function receives a DataFrame df and plots a boxplot for each numeric feature on a single axis.
    If a rank is provided, it appends the rank next to the feature name on the x-axis.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        rank (list, optional): List of ranks for each numeric feature in df. Default is None.
        
    Returns:
        None
    """
    # Filter numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    
    # If rank is provided, modify column labels
    if rank:
        if len(rank) != len(numeric_cols):
            raise ValueError("Length of rank list must match number of numeric columns in df.")
        labels = [f"{col} - {r:.3f}" for col, r in zip(numeric_cols, rank)]
    else:
        labels = numeric_cols
    
    # Plotting
    plt.figure(figsize=(36, 8))
    
    # Plotting all boxplots on a single axis
    plt.boxplot([df[col] for col in numeric_cols], labels=labels, vert=True)
    
    # Rotate feature names
    plt.xticks(rotation=90)
    
    plt.title("Boxplot of Each Feature")
    plt.ylabel("Normalized Value")
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_roc_and_auc(model, df, features):
    """
    Plot the ROC curve and compute the AUC for a given model and dataset.

    Parameters:
    - model : sklearn-like model with a decision_function method
        The classifier for which the ROC curve and AUC will be computed.
    - df : pandas.DataFrame
        The dataset containing the data to be tested.
    - features : list
        The columns from the dataframe to be used as features.

    Returns:
    - roc_auc : float
        The Area Under the Curve (AUC) of the plotted ROC curve.

    Example:
    >>> from sklearn.linear_model import LogisticRegression
    >>> model = LogisticRegression()
    >>> df_sample = pd.DataFrame({"feature1": [1,2,3,4,5], "feature2": [5,4,3,2,1], "target": [0,1,0,1,0]})
    >>> features = ["feature1", "feature2"]
    >>> plot_roc_and_auc(model, df_sample, features)
    """

    # Load dataset and get decision function output
    X, y, _ = load_carray_dataset(path=None, cols=None, df=df[features])
    decision_function_output = model.decision_function(X) 
    decision_function_output = decision_function_output[:, 1].tondarray().ravel()

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y.tondarray(), decision_function_output)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    return roc_auc

def custom_x_format(x, pos):
    if x == 0:
        return 'Baseline'
    elif x >= 0 and x < 10:
        return f'{x}'
    elif x > 9:
        return f'+{x}%'
    else:
        return f'{x}%'

def custom_y_format(y, pos):
    return f'{y:.1f}'  # Format as a decimal with one decimal place

def line_plot_by_attack(attack_data, line_color_scheme, marker_scheme, output_filename=None, dpi=300, fig_size=(6, 3.6),
                         global_font_size=18, xlabel='Perturbation Factor', plot_fake=False, linestyle='--', 
                         fillstyle='none', markersize=10, linewidth=1.5, ignore_first_xstick=False, background_color='#FFFFFF'):
    plt.rcParams['font.family'] = 'Times New Roman' #'serif'
    plt.rcParams['font.size'] = global_font_size
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    # Muda a cor do fundo da figura
    fig.patch.set_facecolor(background_color)

    # Muda a cor do fundo dos eixos
    ax.set_facecolor(background_color)

    for classifier, data in attack_data.items():
        factors, rates = zip(*sorted(data.items()))
        line_color = line_color_scheme[classifier]
        marker = marker_scheme[classifier]

        if plot_fake:
            rates = generate_fake_data(rates)
            fig.text(0.5, 0.5, 'Fake Data', fontsize=40, color='gray', ha='center', va='center', alpha=0.1)

        ax.plot(factors, rates, marker=marker, markersize=markersize, color=line_color, linestyle=linestyle, 
                linewidth=linewidth, label=classifier, fillstyle=fillstyle)

    plt.xlabel(xlabel)
    plt.ylabel('Attack Success Rate')
    # plt.legend()  # Adiciona a legenda

    # Customize y-tick locations to include specific values
    y_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    # ax = plt.gca()
    ax.set_yticks(y_ticks)
    
    # Customize x-tick locations to show every 3rd data point
    x_ticks = list(data.keys())
    x_ticks_every_third = x_ticks[::3]
    if ignore_first_xstick:
        x_ticks_every_third = x_ticks[1::3]

    x_labels = [custom_x_format(x, i) for i, x in enumerate(x_ticks) if x in x_ticks_every_third]
    ax.set_xticks(x_ticks_every_third)
    ax.set_xticklabels(x_labels, rotation=45)
    

    plt.ylim(0, 1.1)
    # Remover as bordas superior e direita
    # ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if output_filename is None:
        plt.show()
    else:
        path = '../results/realistic/figures/'
        plt.savefig(path + output_filename, format='pdf', bbox_inches='tight')
        plt.close()

def generate_fake_data(data, margin=0.5):
    """
    Generate fake data by adding random noise to the original data.
    Parameters:
    - data (list): List of original data values.
    - margin (float): Maximum percentage of noise to add to each value.
    """
    fake_data = []
    for d in data:
        # Generate a new value by adding a random value between -margin and margin
        new_value = d * (1 + np.random.uniform(-margin, margin))
        # Ensure the new value is between 0 and 1
        new_value = max(min(new_value, 1), 0)
        fake_data.append(new_value)
    return fake_data

def plot_ensemble_asr(ensemble_results, gray=False, output_filename=None, ensemble_only=False):
    """
    Plot the attack success rate for each classifier and the ensemble.
    Parameters:
    - ensemble_results (dict): Dictionary containing attack success rates for each classifier.
    - gray (bool): If True, use a grayscale color scheme for the classifier but ensemble. If False, use a colored scheme.
    """
    if gray:
        color_scheme = ['#808080', '#808080', '#808080', '#808080', '#808080', '#d62728']
    else:
        color_scheme = ['#73C698', '#DBB84D', '#1A6FDF', '#F14040', '#756BB1', '#d62728'] # colors based on a paper
    maker_scheme = ['^', 'o', 's', 'D', 'P', '*']
    if ensemble_only:
        classifier_marker = {'bca_ensemble': maker_scheme[2], 'bea_ensemble': maker_scheme[3], 'ensemble': maker_scheme[5]}    
        classifier_color = {'bca_ensemble': color_scheme[2], 'bea_ensemble': color_scheme[3], 'ensemble': color_scheme[5]}
    else:
        classifier_marker = {'dt': maker_scheme[4], 'rf': maker_scheme[0], 'xgb': maker_scheme[1], 'mlp': maker_scheme[2], 'td': maker_scheme[3], 'ensemble': maker_scheme[5]}    
        classifier_color = {'dt': color_scheme[4], 'rf': color_scheme[0], 'xgb': color_scheme[1], 'mlp': color_scheme[2], 'td': color_scheme[3], 'ensemble': color_scheme[5]}
    all_attacks = ['time', 'volume', 'time_volume']

    if output_filename is not None:
        legend_filename = f'legend_{output_filename}.pdf'
    else:
        legend_filename = None
    plot_legend_by_attack(classifier_color, classifier_marker, line_width=3, symbol_size=12, fig_width=17, fig_height=0.15,  output_filename=legend_filename)

    for attack in all_attacks:
        attack_data = {}  # Dicionário para armazenar dados de todos os classificadores para este ataque
        for classifier, attacks in ensemble_results.items():
            if attack in attacks:
                attack_data[classifier] = attacks[attack]
    
        if attack_data:  # Verifica se há dados para este tipo de ataque
            if output_filename is not None:
                output_plot_name = f'{output_filename}_{attack}.pdf'
            else:
                output_plot_name = None
            line_plot_by_attack(
                attack_data,
                classifier_color,
                classifier_marker,
                #output_filename=f'feasible_full_features_{attack}.pdf',
                output_filename=output_plot_name,
                dpi=120,
                fig_size=(6, 3.6),
                global_font_size=20,
                xlabel='Perturbation Factor',
                plot_fake=False
            )

def plot_legend_by_attack(classifier_color, classifier_marker, line_width=2, symbol_size=10, fig_width=8, fig_height=0.1, output_filename=None, background_color='#FFFFFF', ours=True):
    """
    Plot a legend for the classifiers used in the attack success rate plots.
    Parameters:
    - classifier_color (dict): Dictionary mapping classifier names to colors.
    - classifier_marker (dict): Dictionary mapping classifier names to markers.
    - line_width (int): Width of the lines in the legend.
    - symbol_size (int): Size of the markers in the legend.
    - fig_width (float): Width of the figure.
    - fig_height (float): Height of the figure.
    - output_filename (str): Name of the output file. If None, the plot is displayed instead of saved.
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_width, fig_height)  # Ajuste manual do tamanho da figura

    # Muda a cor do fundo da figura
    fig.patch.set_facecolor(background_color)

    # Muda a cor do fundo dos eixos
    ax.set_facecolor(background_color)

    # Dummy plots for creating the legend
    lines = []  # To keep track of the line objects
    for classifier, color in classifier_color.items():
        marker = classifier_marker[classifier]
        if classifier == 'ensemble' and ours:
            classifier = 'Ours'
        elif classifier == 'bca_ensemble':
            classifier = 'Baseline (All Feat.)'
        elif classifier == 'bea_ensemble':
            classifier = 'Baseline (Feat. Sel.)'
        else:
            classifier = classifier.upper()
        
        line, = ax.plot([], [], color=color, marker=marker, linestyle='--', linewidth=line_width, markersize=symbol_size, label=classifier, fillstyle='none')
        lines.append(line)

    # Create the legend without a title
    legend = ax.legend(handles=lines, loc='center', ncol=len(classifier_color), frameon=False)
    # plt.gca().add_artist(legend)
    ax.add_artist(legend)

    # Adjust the axes to fit the legend
    ax.axis('off')

    if output_filename is not None:
        path = '../results/realistic/figures/'
        plt.savefig(path + output_filename, format='pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def compute_ks_statistic_per_column(df1, df2):
    """
    Calculate the Kolmogorov-Smirnov statistic for each pair of columns between two dataframes.
    
    This function compares the distributions of corresponding columns between two dataframes using 
    the two-sample Kolmogorov-Smirnov test. It returns a Series containing the K-S statistic for each column.

    Parameters:
    - df1, df2 (pd.DataFrame): Dataframes to compare. They should have the same column names.

    Returns:
    - pd.Series: Kolmogorov-Smirnov statistic for each column.
    """
    ks_statistic = {}
    for col in df1.columns:
        ks_stat, _ = ks_2samp(df1[col], df2[col])
        ks_statistic[col] = ks_stat
    return pd.Series(ks_statistic)

def rank_columns_by_ks_statistic(df1, df2, numeric_only=True):
    """
    Sort columns based on the Kolmogorov-Smirnov statistic computed between two dataframes.

    This function uses the compute_ks_statistic_per_column function to compute the K-S statistic for each pair of columns 
    between two dataframes and returns a dataframe with feature names and their KS_Stat sorted by the biggest to the lowest changes.

    Parameters:
    - df1, df2 (pd.DataFrame): Dataframes to compare. They should have the same column names.
    - numeric_only (bool): If True (default), only compute and return K-S statistic for numeric columns. If False, compute and return for all columns.

    Returns:
    - DataFrame: Feature names and KS_Stat sorted by the biggest to the lowest changes.
    """
    
    if numeric_only:
        df1 = df1.select_dtypes(include='number')
        df2 = df2.select_dtypes(include='number')
    
    ks_statistic = compute_ks_statistic_per_column(df1, df2)
    sorted_stats = ks_statistic.sort_values(ascending=False)
    
    # Convert sorted series to dataframe
    df_result = sorted_stats.reset_index()
    df_result.columns = ["Feature", "KS_Stat"]
    
    return df_result

def display_dataframes_horizontally(*dataframes, columns_to_display=None):
    """
    Display multiple DataFrames side by side horizontally with optional specific columns.

    Parameters:
    *dataframes : unpacked tuple of pandas.DataFrame
        One or more DataFrames to be displayed side by side.
    columns_to_display : list of lists, optional
        A list where each inner list specifies the columns to display for the corresponding DataFrame.
        By default, all columns of each DataFrame are displayed.

    Returns:
    None

    Example:
    >>> df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    >>> df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8], "C": [9, 10]})
    >>> display_dataframes_horizontally(df1, df2, columns_to_display=[["A", "B"], ["B", "C"]])
    """
    
    html_str = '<table border="0"><tr style="border:0;">'
    
    for idx, df in enumerate(dataframes):
        # Check if columns_to_display is a list of strings
        if columns_to_display and isinstance(columns_to_display[0], str):
            selected_columns = columns_to_display
        # Else, treat it as a list of lists
        elif columns_to_display and len(columns_to_display) > idx:
            selected_columns = columns_to_display[idx]
        else:
            selected_columns = df.columns
        
        df_to_display = df[selected_columns]
        
        html_str += '<td style="border:0;">' + df_to_display.to_html() + '</td>'
        
    html_str += '</tr></table>'
    display(HTML(html_str))




def plot_bar_attackwise(dfs, df_labels, normal=True, title_complement='', fig_size=(20, 5), 
                        bar_width=0.5, star_label=None, legend_top_text=None):
    """
    Plots a grouped bar chart comparing attack accuracies across multiple dataframes, 
    with an option to mark a specific label's bar with a star and display a custom text 
    at the bottom left of the figure.

    Parameters:
        dfs (list of pd.DataFrame): Dataframes with attack accuracy data.
        df_labels (list of str): Labels for the dataframes for the plot legend.
        star_label (str, optional): Label for which a star is to be plotted on the bar.
        bottom_text (str, optional): Custom text to be displayed at the bottom left of the figure.

    Raises:
        ValueError: If the lengths of dfs and df_labels are not equal.
    """
    if len(dfs) != len(df_labels):
        raise ValueError("The number of dataframes and labels must be the same.")
    
    categories = dfs[0]['category'].unique()
    if not normal:
        categories = categories[categories != 'normal']

    num_categories = len(categories)
    x = np.arange(num_categories)
    num_dfs = len(dfs)
    width = bar_width / num_dfs

    fig, ax = plt.subplots(figsize=fig_size)
    
    for i, (df, label) in enumerate(zip(dfs, df_labels)):
        accuracies = df.set_index('category').reindex(categories)['accuracy']
        bars = ax.bar(x + i * width, accuracies, width, label=label)
        
        for bar in bars:
            yval = bar.get_height()
            offset = 0.01
            text_y = yval + offset
            ax.text(bar.get_x() + bar.get_width()/2, text_y, round(yval, 2), 
                    ha='center', va='bottom', rotation=90, fontsize=5)
            if label == star_label:
                ax.text(bar.get_x() + bar.get_width()/2, text_y + offset, '*', 
                        ha='center', va='bottom', color='red', fontsize=10)

    ax.set_xlabel('Attack')
    ax.set_ylabel('Accuracy')
    ax.set_title('Attack-Wise Accuracy: {}' .format(title_complement))
    ax.set_xticks(x + width * (num_dfs - 1) / 2)
    ax.set_xticklabels(categories)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)

    # Add custom text at the bottom left of the figure if provided
    if legend_top_text:
        ax.text(1.01, 1.01, legend_top_text, transform=ax.transAxes, fontsize=10)

    plt.xticks(rotation=90)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.show()



def rank_features_by_permutation(estimator, df, attack_name=None, perm_max_samples=1):
    """
    Ranks features by permutation importance for a given attack category in a dataset.
    
    This function computes the permutation importance of features in the dataset `df` for the specified `attack_name`.
    The dataset is filtered to include only the instances of the given attack and an equal number of normal instances.
    Permutation importance is calculated using the provided `estimator` and the F1 score as the evaluation metric.
    
    Parameters:
    - estimator: A fitted classifier that follows the scikit-learn estimator interface.
    - df: pandas DataFrame, The dataset containing both features and labels.
    - attack_name: str, The name of the attack category for which to compute permutation importance.
    
    Returns:
    - perm_df: pandas DataFrame, A DataFrame containing features ranked by their permutation importance scores in descending order.
      The DataFrame has two columns: 'feature' and 'score'.
      
    Note:
    - The function assumes that `esp_utilities.load_carray_dataset` automatically excludes label columns and returns data in the form of CArray.
    - The labels 'attackCategory' and 'label' are expected to be present in `df`, and they will not be included in the feature importance ranking.
    """
    # Custom scoring function to handle CArray inputs
    def custom_f1_score(y_true, y_pred, **kwargs):
        if isinstance(y_true, secml.array.CArray):
            y_true = y_true.tondarray()
        if isinstance(y_pred, secml.array.CArray):
            y_pred = y_pred.tondarray()
        return f1_score(y_true, y_pred, **kwargs)
    
    # Filter dataset based on attack category and sample normal instances
    if attack_name:
        df_attack = df[df['attackCategory'] == attack_name]
        df_normal = df[df['attackCategory'] == 'normal']
        sampled_normal_df = df_normal.sample(n=len(df_attack), random_state=42)
        df = pd.concat([df_attack, sampled_normal_df], ignore_index=True)

    # Load dataset and convert to NumPy arrays
    X, y, _ = load_carray_dataset(path=None, cols=None, df=df)
    X = X.tondarray()
    y = y.tondarray()

    # Get feature names after dropping label columns
    feature_names = df.drop(columns=['attackCategory', 'label']).columns.tolist()

    # Compute permutation importance
    scorer = make_scorer(custom_f1_score)
    result = permutation_importance(estimator, X, y, n_repeats=20, n_jobs=-1, scoring=scorer, random_state=42, max_samples=perm_max_samples)
    
    # Create DataFrame with permutation importance scores
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'score': result.importances_mean
    })
    
    # Rank features and reset index
    perm_df = perm_df.sort_values(by='score', ascending=False)
    perm_df.reset_index(drop=True, inplace=True)
    
    return perm_df

def plot_dataframe_boxplots_side_by_side(dfs, feature_set, attack_name, df_labels=None, fig_size=(15, 6), label_x_rotation=45):
    if not all(isinstance(df, pd.DataFrame) for df in dfs):
        raise TypeError("All elements in dfs should be pandas dataframes.")
    if not isinstance(feature_set, list):
        raise TypeError("feature_set should be of type list.")
    if not isinstance(attack_name, str):
        raise TypeError("attack_name should be a string.")
    
    if df_labels is None:
        df_labels = ['Baseline', 'Delay 50x', 'Payload 50%']
    if not isinstance(df_labels, list) or len(df_labels) != len(dfs):
        raise ValueError("df_labels should be a list of same length as dfs.")
    
    plt.figure(figsize=fig_size)
    plt.title(f'{attack_name}', fontsize=15)
    
    # Combining data from all dataframes for selected features
    combined_data = pd.concat([df[feature_set] for df in dfs if any(feat in df.columns for feat in feature_set)], keys=df_labels)
    combined_data = combined_data.reset_index(level=0).rename(columns={'level_0': 'DataFrame'})
    
    # Melting the data for seaborn to plot
    melted_data = pd.melt(combined_data, id_vars='DataFrame', var_name='Feature', value_name='Value')

    # Creating a boxplot for all features
    sns.boxplot(x='Feature', y='Value', hue='DataFrame', data=melted_data)
    plt.ylabel('Value')
    plt.xticks(rotation=label_x_rotation)
    
    # Adjusting the legend's position
    plt.legend(title='Dataset', loc='upper left', bbox_to_anchor=(1, -0.15))
    
    plt.tight_layout()
    plt.show()

def plot_dataframe_violin_plots_side_by_side(dfs, feature_set, attack_name, df_labels=None, fig_size=(15, 6), label_x_rotation=45):
    if not all(isinstance(df, pd.DataFrame) for df in dfs):
        raise TypeError("All elements in dfs should be pandas dataframes.")
    if not isinstance(feature_set, list):
        raise TypeError("feature_set should be of type list.")
    if not isinstance(attack_name, str):
        raise TypeError("attack_name should be a string.")
    
    if df_labels is None:
        df_labels = ['Baseline', 'Delay 50x', 'Payload 50%']
    if not isinstance(df_labels, list) or len(df_labels) != len(dfs):
        raise ValueError("df_labels should be a list of same length as dfs.")
    
    plt.figure(figsize=fig_size)
    plt.title(f'{attack_name}', fontsize=15)
    
    # Combining data from all dataframes for selected features
    combined_data = pd.concat([df[feature_set] for df in dfs if any(feat in df.columns for feat in feature_set)], keys=df_labels)
    combined_data = combined_data.reset_index(level=0).rename(columns={'level_0': 'DataFrame'})
    
    # Melting the data for seaborn to plot
    melted_data = pd.melt(combined_data, id_vars='DataFrame', var_name='Feature', value_name='Value')

    # Creating a violin plot for all features
    sns.violinplot(x='Feature', y='Value', hue='DataFrame', data=melted_data, split=True)
    plt.ylabel('Value')
    plt.xticks(rotation=label_x_rotation)
    
    # Adjusting the legend's position
    plt.legend(title='Dataset', loc='upper left', bbox_to_anchor=(1, -0.15))
    
    plt.tight_layout()
    plt.show()


def get_feature_imp(model, features):
    """
    Generate a DataFrame ranking features based on their importance as determined by the provided model.

    Args:
    model: A machine learning model with a feature_importances_ attribute.
    features: A list of feature names corresponding to the features in the model.

    Returns:
    DataFrame: A DataFrame with features and their corresponding importance scores, sorted in descending order.
    """
    rank = model.feature_importances_
    df_feature_imp = pd.DataFrame(rank, index=features, columns=['Importance']).sort_values(by='Importance', ascending=False)
    return df_feature_imp


def batch_normalize_dataframes(dataframes, exclude_columns=[]):
    """
    Normalize a list of pandas DataFrames and then split them back into their original structure.

    This function first merges the list of DataFrames into a single DataFrame, 
    adding an extra column ('DataFrameID') to track the original DataFrame of each row.
    It then normalizes the merged DataFrame, excluding specified columns.
    Finally, it splits the normalized DataFrame back into the original list of DataFrames structure.

    Parameters:
    - dataframes (list of pd.DataFrame): List of DataFrames to be normalized and split.
    - exclude_columns (list of str, optional): List of column names to exclude from normalization.

    Returns:
    - list of pd.DataFrame: List of normalized DataFrames, split according to the original structure.
    """
    
    # Step 1: Merge DataFrames with an identifier
    for i, df in enumerate(dataframes):
        df['DataFrameID'] = i
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Step 2: Normalize Data, excluding specified columns
    columns_to_normalize = [col for col in merged_df.columns if col not in exclude_columns + ['DataFrameID']]
    scaler = MinMaxScaler()
    merged_df[columns_to_normalize] = scaler.fit_transform(merged_df[columns_to_normalize])

    # Step 3: Split the DataFrame back into the original list structure
    split_dataframes = [merged_df[merged_df['DataFrameID'] == i].drop(columns=['DataFrameID']) for i in range(len(dataframes))]
    
    return split_dataframes

def random_undersample_dataset(dataset, label_column='label', random_state=42):
    """
    Perform undersampling on a dataset based on a binary label. The aim is to reduce the size of 
    the 'normal' class (label=0) to be equal or close to the size of the 'attack' class (label=1).

    Parameters:
    - dataset (pd.DataFrame): The dataset to be undersampled.
    - label_column (str): The name of the binary label column.

    Returns:
    - pd.DataFrame: The undersampled dataset.
    """
    # Separate the dataset into two groups based on the label
    normal_data = dataset[dataset[label_column] == 0]
    attack_data = dataset[dataset[label_column] == 1]

    # Determine the number of samples to match in the normal data
    n_samples = len(attack_data)

    # Resample the normal data to match the number of attack samples
    normal_sampled = resample(normal_data, 
                              replace=False, 
                              n_samples=n_samples, 
                              random_state=random_state)

    # Merge the undersampled normal data with the attack data
    undersampled_data = pd.concat([normal_sampled, attack_data], ignore_index=True)

    return undersampled_data


def get_features_with_low_correlation(df, threshold=0.9, method='spearman', labels=['attackCategory', 'label']):
    """
    Identify columns in the DataFrame that are not highly correlated.

    Args:
    df (pd.DataFrame): The DataFrame from which to analyze features.
    threshold (float): The correlation threshold to identify high correlation. Defaults to 0.9.
    method (str): The method for correlation (pearson, spearman, kendall). Defaults to 'spearman'.
    labels (list): List of label columns to include in the final column list. Defaults to ['attackCategory', 'label'].

    Returns:
    list: A list of column names with low correlation, including specified labels.
    """
    # Isolating the feature columns (excluding labels)
    features_df = df.drop(labels, axis=1, errors='ignore')

    # Calculate the correlation matrix
    corr_matrix = features_df.corr(method=method).abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find index of feature columns with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Columns not highly correlated
    non_correlated_columns = [col for col in features_df.columns if col not in to_drop]

    # Add the label columns back to the list
    final_columns = non_correlated_columns + labels

    return final_columns

# Example usage:
# df = pd.DataFrame(...)  # Assuming df is a DataFrame with some data
# columns_with_low_corr = get_features_with_low_correlation(df, threshold=0.9, method='spearman', labels=['attackCategory', 'label'])

import pandas as pd

def get_min_max(df):
    """
    This function takes a DataFrame as input and returns a new DataFrame
    containing the minimum and maximum values for each column.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    DataFrame: A new DataFrame with 'Column', 'Min', and 'Max' columns.
    """

    # Creating a list to hold min-max values for each column
    min_max_list = []

    # Loop through each column in the original DataFrame
    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        min_max_list.append({'Column': col, 'Min': min_val, 'Max': max_val})

    # Create a new DataFrame from the list
    min_max_df = pd.DataFrame(min_max_list)

    return min_max_df

# Example usage
# df = ... (your DataFrame)
# min_max_df = get_min_max(df)
# print(min_max_df)
