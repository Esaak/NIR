import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error  
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import optuna
from tqdm import tqdm
from scipy import stats
import joblib

import os
import re

# from perform_visualization import perform_eda, perform_eda_short, performance_visualizations

random_seed = 42
early_stopping_round = 100

# funcs

def get_dataset(folder_paths: list, filename_features: str, filename_target: str):
    X = pd.DataFrame()
    y = pd.DataFrame()
    for folder in folder_paths:
        X_tmp = pd.read_csv(folder + "/" + filename_features)
        y_tmp = pd.read_csv(folder + "/" + filename_target)
        
        is_unnamed = pd.isna(X_tmp.columns[0]) or str(X_tmp.columns[0]).startswith('Unnamed:')
        is_unnamed_y = pd.isna(y_tmp.columns[0]) or str(y_tmp.columns[0]).startswith('Unnamed:')
        if is_unnamed:
            X_tmp = X_tmp.drop(X_tmp.columns[0], axis=1)
        if is_unnamed_y:
            y_tmp = y_tmp.drop(y_tmp.columns[0], axis=1)
        
        # if X_tmp.columns[0] != "y":
        #     col_y = np.ones(X_tmp.shape[0]) * 1000
        #     X_tmp.insert(0, "y", col_y)
        
        print(X_tmp.shape, y_tmp.shape)
        X = pd.concat([X, X_tmp], axis = 0)
        y = pd.concat([y, y_tmp], axis = 0)
    X.reset_index(inplace=True, drop=True)
    y.reset_index(inplace=True, drop=True)
    print(X.shape, y.shape)
    return X, y

def add_exp_num(X):
    tracers_num = len(np.unique(X["Tracer"]))
    distances_num = len(np.unique(X["distances"]))
    experiment_num = []
    for exp in range(X.shape[0]//(tracers_num*distances_num)):
        experiment_num.extend(list(np.ones(tracers_num*distances_num) * (exp + 1)))
    X["experiment_num"] = experiment_num
    return X

def data_split(X, y, test_size = 0.2, valid_size = None, eval_size = None, random_seed = random_seed):
    rng = np.random.default_rng(seed=random_seed)
    
    experiment_nums = X["experiment_num"].unique()
    total_experiments = len(experiment_nums)
    
    n_test = int(test_size * total_experiments)
    
    splits = {}
    remaining_exps = set(experiment_nums)
    
    test_experiments = set(rng.choice(list(remaining_exps), n_test, replace=False))
    remaining_exps -= test_experiments
    splits['test'] = {
        'X': X[X["experiment_num"].isin(test_experiments)].copy(),
        'y': None
    }
    
    if valid_size:
        n_valid = int(valid_size * (total_experiments - n_test))
        valid_experiments = set(rng.choice(list(remaining_exps), n_valid, replace=False))
        remaining_exps -= valid_experiments
        splits['valid'] = {
            'X': X[X["experiment_num"].isin(valid_experiments)].copy(),
            'y': None
        }
    
    if eval_size:
        n_eval = int(eval_size * (total_experiments - n_test - n_valid))
        eval_experiments = set(rng.choice(list(remaining_exps), n_eval, replace=False))
        remaining_exps -= eval_experiments
        
        splits['eval'] = {
            'X': X[X["experiment_num"].isin(eval_experiments)].copy(),
            'y': None
        }
    
    splits['train'] = {
        'X': X[X["experiment_num"].isin(remaining_exps)].copy(),
        'y': None
    }
    
    for split_name in splits:
        splits[split_name]['y'] = y.loc[splits[split_name]['X'].index].copy()
    
    split_shapes = [f"{name}: X{splits[name]['X'].shape}, y{splits[name]['y'].shape}" 
                   for name in ['train', 'test', 'valid', 'eval'] 
                   if name in splits]
    print(f"Split shapes: {', '.join(split_shapes)}")
    
    if valid_size and eval_size:
        return (splits['train']['X'], splits['test']['X'], splits['valid']['X'], splits['eval']['X'],
                splits['train']['y'], splits['test']['y'], splits['valid']['y'], splits['eval']['y'])
    elif valid_size:
        return (splits['train']['X'], splits['test']['X'], splits['valid']['X'],
                splits['train']['y'], splits['test']['y'], splits['valid']['y'])
    else:
        return splits['train']['X'], splits['test']['X'], splits['train']['y'], splits['test']['y']

def del_outs(X, y):
    #Delete zeros
    mask = (y["c_std_y"] != 0) & (y["c_std_z"] != 0)
    X = X[mask]
    y = y[mask]
    points = np.linspace(0, np.max(y["c_std_y"]), 200)
    quantiles = np.histogram(y["c_std_y"], points)
    hist_mode = quantiles[1][np.argmax(quantiles[0])]
    cut_mask = y["c_std_y"] >= hist_mode
    X = X[cut_mask]
    y = y[cut_mask]
    print(X.shape, y.shape)
    return X, y

def make_means_target(X, y):
    y["c_mean_y"] = X["y"] - y["c_mean_y"]
    y["c_mean_z"] = X["z"] - y["c_mean_z"]
    return y

def del_columns(X, columns):
    X.drop(columns=columns, inplace=True)
    # return X

# stat_path = os.path.join(os.getcwd())
# pattern = re.compile(r'output_*\d')
# folder_paths =[]
# for folder_name in os.listdir(stat_path):
#     if pattern.match(folder_name):
#         folder_paths.append(folder_name)

filename_features = "features_full.csv"
filename_target = "target_full.csv"

folder_paths = ["/app/nse/ml/output_19_01_2025_2"]

X, y = get_dataset(folder_paths, filename_features, filename_target)
y = make_means_target(X, y)
y.rename(columns={"c_mean_y": "c_delta_y", "c_mean_z": "c_delta_z"}, inplace=True)
X = add_exp_num(X)
X, y = del_outs(X, y)
# X_train, X_test, X_valid, X_eval, y_train, y_test, y_valid, y_eval = data_split(X, y, test_size=0.2, valid_size=0.2, eval_size=0.1)
print("Split data...")
X_train, X_test, X_eval, y_train, y_test, y_eval = data_split(X, y, test_size=0.2, valid_size=0.1)

del_columns(X_train, ["experiment_num", "Tracer"])
del_columns(X_test, ["experiment_num", "Tracer"])
# del_columns(X_valid, ["experiment_num", "Tracer"])
del_columns(X_eval, ["experiment_num", "Tracer"])
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import quantile_transform, power_transform
class TargetPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, near_zero = 1e-3):
        self.near_zero = near_zero

    def fit(self, data: pd.DataFrame, *args):
        self.columns = data.columns
        self.std_y_bias = np.min(data["c_std_y"])
        self.std_z_bias = np.min(data["c_std_z"])
        
        self.std_y_std = np.std(data["c_std_y"])
        self.std_z_std = np.std(data["c_std_z"])

        self.mean_y_mean = np.mean(data["c_delta_y"])
        self.mean_z_mean = np.mean(data["c_delta_z"])

        self.mean_y_std = np.std(data["c_delta_y"])
        self.mean_z_std = np.std(data["c_delta_z"])

        self.mean_y_minuses = data.index[data["c_delta_y"] < 0].to_list()
        self.mean_z_minuses = data.index[data["c_delta_z"] < 0].to_list()

        return self

    def transform(self, data: pd.DataFrame):
        data["c_std_y"] -= self.std_y_bias - self.near_zero
        data["c_std_z"] -= self.std_z_bias - self.near_zero
        data["c_std_y"] = np.log(data["c_std_y"])
        data["c_std_z"] = np.log1p(data["c_std_z"])
        
        mean_y_sign = np.sign(data["c_delta_y"])
        data["c_delta_y"] = np.abs(data["c_delta_y"])       
        data["c_delta_y"] = np.log1p(data["c_delta_y"])
        # data["c_delta_y"] = np.log(data["c_delta_y"] + 1e-6)
        data["c_delta_y"]*=mean_y_sign
        
        data["c_delta_z"] = (data["c_delta_z"] - self.mean_z_mean)/self.mean_z_std
        mean_z_sign = np.sign(data["c_delta_z"])
        data["c_delta_z"]= mean_z_sign * np.log1p(np.abs(data["c_delta_z"]))
        # data["c_delta_z"]= mean_z_sign * np.log(np.abs(data["c_delta_z"]) + 1e-6)
        
        return data
    
    def inverse_transform(self, data: pd.DataFrame):
        if "c_std_y" in data.columns:
            data["c_std_y"] = np.exp(data["c_std_y"])
            data["c_std_y"] += self.std_y_bias - self.near_zero

        if "c_std_z" in data.columns:
            data["c_std_z"] = np.expm1(data["c_std_z"])
            data["c_std_z"] += self.std_z_bias - self.near_zero
        
        if "c_delta_y" in data.columns:
            mean_y_sign = np.sign(data["c_delta_y"])
            data["c_delta_y"] = np.abs(data["c_delta_y"])
            data["c_delta_y"] = np.expm1(data["c_delta_y"])
            data["c_delta_y"] *=mean_y_sign
        if "c_delta_z" in data.columns:
            mean_z_sign = np.sign(data["c_delta_z"])
            data["c_delta_z"] = np.expm1(np.abs(data["c_delta_z"]))
            data["c_delta_z"] *=mean_z_sign
            data["c_delta_z"] = data["c_delta_z"] * self.mean_z_std + self.mean_z_mean
        return data
from sklearn.preprocessing import StandardScaler
print("Preprocess data...")
scaler = StandardScaler()
X_train_n = scaler.fit_transform(X_train)
X_test_n = scaler.transform(X_test)
X_eval_n = scaler.transform(X_eval)

y_1 = y_train.copy()
t_preproc = TargetPreprocessor()
# t_preproc.fit(y)
t_preproc.fit(y)
y_1 = t_preproc.transform(y_1)
# perform_eda_short(X_train, y_train)
# t_preproc.fit(y)
y_eval = t_preproc.transform(y_eval)
print("x")
import os
import re
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional
import shap
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(y_pred, y_true))


class RegressionNet(nn.Module):
    
    def __init__(self, input_dim: int, hidden1: int = 1024, hidden2: int = 512, hidden3: int = 256, hidden4: int = 64, output_dim: int = 4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.Mish(),
            nn.Linear(hidden1, hidden2),
            nn.Mish(),
            nn.Linear(hidden2, hidden3),
            nn.Mish(),
            nn.Linear(hidden3, hidden4),
            nn.Mish(),
            nn.Linear(hidden4, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = None,
        learning_rate: float = 1e-3,
        device: str = None,
    ):
        assert(torch.cuda.is_available())
        self.device = torch.device('cuda')
        self.model = model.to(self.device)
        self.criterion = criterion if criterion else RMSELoss()
        
        # Adaptive learning rate with ReduceLROnPlateau
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,  # Reduce learning rate by half
            patience=5,  # Wait for 5 epochs without improvement
            min_lr=1e-6,  # Minimum learning rate
            verbose=True
        )
        
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 256,
        validation_data: Tuple[np.ndarray, pd.DataFrame] = None,
        early_stopping_patience: int = None
    ) -> Dict[str, List[float]]:
        # Convert data to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Prepare validation data if provided
        if validation_data:
            X_val, y_val = validation_data
            X_val_tensor = torch.FloatTensor(X_val.to_numpy()).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val.to_numpy()).to(self.device)
        
        # Training metrics
        metrics = {
            'train_loss': [],
            'val_loss': [] if validation_data else None,
            'learning_rates': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = os.path.join('best_model.pth')
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Calculate average loss for the epoch
            avg_train_loss = epoch_loss / len(dataloader)
            metrics['train_loss'].append(avg_train_loss)
            
            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            metrics['learning_rates'].append(current_lr)
            
            # Validation phase
            if validation_data:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()
                    metrics['val_loss'].append(val_loss)
                
                # Adaptive learning rate adjustment
                self.scheduler.step(val_loss)
                
                # Model checkpointing
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save the best model
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'epoch': epoch
                    }, best_model_path)
                else:
                    patience_counter += 1
                
                # Early stopping
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    break
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                log_msg = f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}'
                if validation_data:
                    log_msg += f', Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}'
                print(log_msg)
        
        return metrics
    
    def evaluate(self, X_test: np.ndarray, y_test: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        # Load the best model if it exists
        best_model_path = os.path.join('best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test.to_numpy()).to(self.device)
            y_test_tensor = torch.FloatTensor(y_test.to_numpy()).to(self.device)
            
            test_outputs = self.model(X_test_tensor)
            
            rmse_loss = self.criterion(test_outputs, y_test_tensor).item()
            
            # Convert predictions to DataFrame
            y_pred = pd.DataFrame(test_outputs.cpu().numpy(), columns=y_test.columns)
            
        return y_pred, rmse_loss
    
config = {
        'random_seed': 42,
        'data_path': '/app/nse/ml',
        'features_filename': 'features_full.csv',
        'targets_filename': 'target_full.csv',
        'model_params': {
            'input_dim': 9,
            "hidden1": 1024,
            'hidden2': 512,
            'hidden3': 128,
            'hidden4': 64,
            'output_dim': 1
        },
        'training_params': {
            'learning_rate': 1e-4,  # Increased from 1e-4
            'epochs': 50,  # Increased from 50
            'batch_size': 6144,
            'early_stopping_patience': 20
        },
        'output_dir': './outputs_16_03_2025',
        'model_checkpoints_dir': './model_checkpoints_16_03_2025'
    }
torch.manual_seed(config['random_seed'])
np.random.seed(config['random_seed'])
class UncertaintyMLPEstimator:
    def __init__(self, config, target_transformer, target_name, n_bootstrap=100, alpha=0.05, random_state=42):
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.random_state = random_state
        self.models = []
        self.base_model = None
        self.target_transformer = target_transformer
        self.target_name = target_name
        self.config = config
        self.base_model = RegressionNet(**self.config['model_params'])

    def fit(self, X, y, X_eval, y_eval):
        y = np.array(y)
        X = np.array(X)

        self.base_trainer = ModelTrainer(self.base_model)

        metrics = self.base_trainer.train(
            X, y,
            epochs=self.config['training_params']['epochs'],
            batch_size=self.config['training_params']['batch_size'],
            # validation_data=(data['X_val'], data['y_val']),
            validation_data=(X_eval, y_eval),
            early_stopping_patience=self.config['training_params']['early_stopping_patience']
        )
        
        # Bootstrap sampling for uncertainty estimation
        n_samples = X.shape[0]
        self.models = []
        
        for i in tqdm(range(self.n_bootstrap), desc="Fitting bootstrap models"):
            # Sample with replacement
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Create and fit a new model
            model = RegressionNet(**self.config['model_params'])
            trainer = ModelTrainer(model)

            metrics = trainer.train(
                X_bootstrap, y_bootstrap,
                epochs=self.config['training_params']['epochs'],
                batch_size=self.config['training_params']['batch_size'],
                # validation_data=(data['X_val'], data['y_val']),
                validation_data=(X_eval, y_eval),
                early_stopping_patience=config['training_params']['early_stopping_patience']
            )
            self.models.append(trainer)
        
        return self
    
   
    def predict_with_uncertainty(self, X, y):
        result = {}
        
        # Base model prediction
        y_pred_tmp, rmse_loss = self.base_trainer.evaluate(X, y)
        # y_pred_tmp = pd.DataFrame(data=y_pred_tmp, columns=[self.target_name])
        self.target_transformer.inverse_transform(y_pred_tmp)
        result['prediction'] = np.array(y_pred_tmp[self.target_name])

        # Bootstrap-based uncertainty
        bootstrap_predictions = []
        for trainers in tqdm(self.models, desc="Generating bootstrap predictions"):
            
            y_pred_tmp, rmse_loss = trainers.evaluate(X, y)
            # y_pred_tmp = pd.DataFrame(data=y_pred_tmp, columns=[self.target_name])
            self.target_transformer.inverse_transform(y_pred_tmp)
            result['prediction'] = np.array(y_pred_tmp[self.target_name])
            bootstrap_predictions.append(np.array(y_pred_tmp[self.target_name]))

        bootstrap_predictions = np.array(bootstrap_predictions)
        
        
        result['mean'] = np.mean(bootstrap_predictions, axis=0)
        result['std'] = np.std(bootstrap_predictions, axis=0)
        lower_percentile = 100 * self.alpha / 2
        upper_percentile = 100 * (1 - self.alpha / 2)
        result['lower_bound'] = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
        result['upper_bound'] = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
        
        # tree_predictions = []
        # for tree in self.base_model.estimators_:
        #     tree_predictions.append(tree.predict(X))
        # tree_predictions = np.array(tree_predictions)
        # result['tree_variance'] = np.var(tree_predictions, axis=0)
        
        
        return result
    
    def evaluate_uncertainty(self, X_test, y_test):
        predictions = self.predict_with_uncertainty(X_test, y_test)
        
        eval_results = {}
        y_test = np.array(np.squeeze(y_test.to_numpy()))
        # Evaluate prediction intervals
        y_test_bigger = y_test >= predictions['lower_bound']
        y_test_smaller = y_test <= predictions['upper_bound'] 
        in_interval = y_test_bigger * y_test_smaller 
        eval_results['interval_coverage'] = np.mean(in_interval)
        eval_results['expected_coverage'] = 1 - self.alpha
        # Calculate RMSE
        eval_results['rmse'] = np.sqrt(mean_squared_error(y_test, predictions['prediction']))
        eval_results['std_mean'] = np.mean(predictions["std"])
        
        # Calculate uncertainty calibration for regression
        # Sort predictions by uncertainty (std)
        # sorted_indices = np.argsort(predictions['std'])
        # sorted_errors = np.abs(y_test - predictions['prediction'])[sorted_indices]
        
        # # Divide into bins and calculate average error in each bin
        # n_bins = 10
        # bin_size = len(sorted_indices) // n_bins
        # bin_errors = []
        # bin_stds = []
        
        # for i in range(n_bins):
        #     start_idx = i * bin_size
        #     end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_indices)
        #     bin_errors.append(np.mean(sorted_errors[start_idx:end_idx]))
        #     bin_stds.append(np.mean(predictions['std'][sorted_indices[start_idx:end_idx]]))
        
        # eval_results['bin_errors'] = bin_errors
        # eval_results['bin_stds'] = bin_stds
        
        return eval_results
    
    def visualize_uncertainty(self, X_test, y_test=None, m_eval=None, n_samples=None):
        y_test.reset_index(inplace=True, drop=True)
        X_test.reset_index(inplace=True, drop=True)
        if n_samples is not None and n_samples < X_test.shape[0]:
            indices = np.random.choice(X_test.shape[0], n_samples, replace=False)
            X_test_subset = X_test.loc[indices]
            y_test_subset = y_test.loc[indices] if y_test is not None else None
        else:
            X_test_subset = X_test
            y_test_subset = y_test
        
        predictions = self.predict_with_uncertainty(X_test_subset, y_test_subset)
        
        
        # Sort by prediction for clearer visualization
        sort_idx = np.argsort(predictions['prediction'])
        
        plt.figure(figsize=(10, 4))
        plt.errorbar(
            np.arange(len(sort_idx)),
            predictions['prediction'][sort_idx],
            yerr=[
                np.abs(predictions['prediction'][sort_idx] - predictions['lower_bound'][sort_idx]),
                np.abs(predictions['upper_bound'][sort_idx] - predictions['prediction'][sort_idx])
            ],
            fmt='o', alpha=0.6, ecolor='lightgray', capsize=3,
            label = f'Uncertainty: {m_eval["std_mean"]:.8f},\nInterval coverage: {m_eval["interval_coverage"]:.4f} (expected: {m_eval["expected_coverage"]:.4f})'
        )
        y_test_subset.reset_index(inplace=True, drop=True)
        if y_test_subset is not None:
            plt.scatter(np.arange(len(sort_idx)), y_test_subset.loc[sort_idx], 
                        c='red', marker='x', label='Actual')
        
        plt.xlabel('Sample Index (sorted by prediction)')
        plt.ylabel('Prediction')
        plt.title(f'Predictions with Uncertainty, {self.target_name}')

        if y_test_subset is not None:
            plt.legend()
        plt.tight_layout()
        plt.savefig(f"uncertainty_{self.target_name}.png",  dpi=300, bbox_inches='tight')
            

    
    def save(self, filename):
        """Save the model and related objects"""
        joblib.dump(self, filename)
    
    @classmethod
    def load(cls, filename):
        """Load a saved model"""
        return joblib.load(filename)

print("Running ANN uncertainty estimation...")
X_train_n_pd = pd.DataFrame(X_train_n, columns=X_train.columns)
X_eval_n_pd = pd.DataFrame(X_eval_n, columns=X_train.columns)
X_test_n_pd = pd.DataFrame(X_test_n, columns=X_train.columns)

for target in y_1.columns:
    print(f"Start calculate uncertainty for {target}")
    nn_uncertainty = UncertaintyMLPEstimator(config, t_preproc,target, n_bootstrap=10)
    nn_uncertainty.fit(X_train_n_pd, pd.DataFrame(y_1[target], columns=[target]), X_eval_n_pd, pd.DataFrame(y_eval[target], columns=[target]))
    # cb_predictions = rf_uncertainty.predict_with_uncertainty(X_test_n)
    nn_eval = nn_uncertainty.evaluate_uncertainty(X_test_n_pd, pd.DataFrame(np.array(y_test[target]),columns=[target]))

    print(f"MLP RMSE: {nn_eval['rmse']:.4f}")
    print(f"Interval coverage: {nn_eval['interval_coverage']:.4f} (expected: {nn_eval['expected_coverage']:.4f})")

    # Visualize with a subset of samples for clarity
    nn_uncertainty.visualize_uncertainty(X_test_n_pd, pd.DataFrame(np.array(y_test[target]), columns=[target]), nn_eval, n_samples=100)