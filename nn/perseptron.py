import os
import re
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional

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

from perform_visualization import performance_visualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TargetPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, near_zero: float = 1e-3):
        self.near_zero = near_zero
        self.stats = {}
    
    def fit(self, data: pd.DataFrame, y=None) -> 'TargetPreprocessor':
        self.columns = data.columns
        
        # Store statistics for each target variable
        for col in ['c_std_y', 'c_std_z']:
            self.stats[f'{col}_min'] = np.min(data[col])
        
        for col in ['c_delta_y', 'c_delta_z']:
            self.stats[f'{col}_mean'] = np.mean(data[col])
            self.stats[f'{col}_std'] = np.std(data[col])
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        
        # Apply log transformation to standard deviations after bias correction
        for col in ['c_std_y', 'c_std_z']:
            data[col] -= self.stats[f'{col}_min'] - self.near_zero
            data[col] = np.log1p(data[col]) if col == 'c_std_z' else np.log(data[col])
        
        mean_y_sign = np.sign(data["c_delta_y"])
        data["c_delta_y"] = np.abs(data["c_delta_y"])       
        data["c_delta_y"] = np.log1p(data["c_delta_y"])
        data["c_delta_y"]*=mean_y_sign
        
        data["c_delta_z"] = (data["c_delta_z"] - self.stats["c_delta_z_mean"])/self.stats["c_delta_z_std"]
        mean_z_sign = np.sign(data["c_delta_z"])
        data["c_delta_z"]= mean_z_sign * np.log1p(np.abs(data["c_delta_z"]))

        return data
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        
        # Reverse log transformation and restore standard deviations
        data['c_std_y'] = np.exp(data['c_std_y'])
        data['c_std_z'] = np.expm1(data['c_std_z'])
        
        for col in ['c_std_y', 'c_std_z']:
            data[col] += self.stats[f'{col}_min'] - self.near_zero
        
        mean_y_sign = np.sign(data["c_delta_y"])
        data["c_delta_y"] = np.abs(data["c_delta_y"])
        data["c_delta_y"] = np.expm1(data["c_delta_y"])
        data["c_delta_y"] *=mean_y_sign
        
        mean_z_sign = np.sign(data["c_delta_z"])
        data["c_delta_z"] = np.expm1(np.abs(data["c_delta_z"]))
        data["c_delta_z"] *=mean_z_sign
        data["c_delta_z"] = data["c_delta_z"] * self.stats["c_delta_z_std"] + self.stats["c_delta_z_mean"]

        return data


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(y_pred, y_true))


class RegressionNet(nn.Module):
    
    def __init__(self, input_dim: int, hidden1: int = 512, hidden2: int = 256, hidden3: int = 64, output_dim: int = 4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = None,
        learning_rate: float = 1e-3,
        device: str = None
    ):
        assert(torch.cuda.is_available())
        self.device = torch.device('cuda')
        self.model = model.to(self.device)
        self.criterion = criterion if criterion else RMSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        logger.info(f"Training on: {self.device}")
    
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
        X_train_tensor = torch.FloatTensor(X_train.to_numpy()).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.to_numpy()).to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare validation data if provided
        if validation_data:
            X_val, y_val = validation_data
            X_val_tensor = torch.FloatTensor(X_val.to_numpy()).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val.to_numpy()).to(self.device)
        
        # Training metrics
        metrics = {
            'train_loss': [],
            'val_loss': [] if validation_data else None
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
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
            
            # Validation phase
            if validation_data:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()
                    metrics['val_loss'].append(val_loss)
                
                # Early stopping check
                if early_stopping_patience:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"Early stopping triggered at epoch {epoch+1}")
                            break
                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     torch.save()
            # Print progress
            if (epoch + 1) % 10 == 0:
                log_msg = f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}'
                if validation_data:
                    log_msg += f', Val Loss: {val_loss:.4f}'
                logger.info(log_msg)
        
        return metrics
    
    def evaluate(self, X_test: np.ndarray, y_test: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test.to_numpy()).to(self.device)
            y_test_tensor = torch.FloatTensor(y_test.to_numpy()).to(self.device)
            
            test_outputs = self.model(X_test_tensor)
            
            rmse_loss = self.criterion(test_outputs, y_test_tensor).item()
            
            # Convert predictions to DataFrame
            y_pred = pd.DataFrame(test_outputs.cpu().numpy(), columns=y_test.columns)
            
        return y_pred, rmse_loss
    
    def save_model(self, path: str, metadata: Dict = None):
        save_dict = {
            'model_state_dict': self.model.state_dict(),
        }
        
        if metadata:
            save_dict.update(metadata)
        
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")


class DataLoader:
    
    @staticmethod
    def load_from_folders(base_path: str, feature_filename: str, target_filename: str, 
                          folder_pattern: str = r'output_*\d') -> Tuple[pd.DataFrame, pd.DataFrame]:
        pattern = re.compile(folder_pattern)
        folder_paths = []
        
        # Find matching folders
        for folder_name in os.listdir(base_path):
            if pattern.match(folder_name):
                folder_paths.append(os.path.join(base_path, folder_name))
        
        if not folder_paths:
            folder_paths = [os.path.join(base_path, "output_28_12_2024")]
        
        folder_paths = [os.path.join(base_path, "output_19_01_2025_2")]
        X = pd.DataFrame()
        y = pd.DataFrame()
        
        # Load data from each folder
        for folder in folder_paths:
            X_tmp = pd.read_csv(os.path.join(folder, feature_filename))
            y_tmp = pd.read_csv(os.path.join(folder, target_filename))
            
            # Clean up unnamed columns
            for df in [X_tmp, y_tmp]:
                if pd.isna(df.columns[0]) or str(df.columns[0]).startswith('Unnamed:'):
                    df.drop(df.columns[0], axis=1, inplace=True)
            
            # Ensure 'y' column exists
            # if X_tmp.columns[0] != "y":
            #     X_tmp.insert(0, "y", np.ones(X_tmp.shape[0]) * 1000)
            # X_tmp.drop(columns="Tracer", inplace=True)
            logger.info(f"Loaded data from {folder}: X shape={X_tmp.shape}, y shape={y_tmp.shape}")
            
            # Concatenate data
            X = pd.concat([X, X_tmp], axis=0)
            y = pd.concat([y, y_tmp], axis=0)
        
        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)
        
        return X, y
    
    @staticmethod
    def add_exp_num(X):
        tracers_num = len(np.unique(X["Tracer"]))
        distances_num = len(np.unique(X["distances"]))
        experiment_num = []
        for exp in range(X.shape[0]//(tracers_num*distances_num)):
            experiment_num.extend(list(np.ones(tracers_num*distances_num) * (exp + 1)))
        X["experiment_num"] = experiment_num
        return X
    
    @staticmethod
    def del_outs(X, y):
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
    
    @staticmethod
    def make_means_target(X, y):
        y["c_mean_y"] = X["y"] - y["c_mean_y"]
        y["c_mean_z"] = X["z"] - y["c_mean_z"]
        return y
    
    @staticmethod
    def data_split(X, y, test_size = 0.2, valid_size = None, eval_size = None, random_seed=42):
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
    
    @staticmethod
    def preprocess_and_split(
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_size: float = None,
        val_size: float = None,
        eval_size: float = None,
        random_seed: int = 42
    ) -> Dict[str, Union[np.ndarray, pd.DataFrame, StandardScaler, TargetPreprocessor]]:
        # Filter out rows with zero standard deviations
        y = DataLoader.make_means_target(X, y)
        y.rename(columns={"c_mean_y": "c_delta_y", "c_mean_z": "c_delta_z"}, inplace=True)
        X = DataLoader.add_exp_num(X)
        X, y = DataLoader.del_outs(X, y)

        # Feature scaling
        # feature_scaler = StandardScaler()
        # X_scaled = feature_scaler.fit_transform(X)
        target_preprocessor = TargetPreprocessor()
        target_preprocessor.fit(y)

        if val_size and eval_size:
            X_train, X_test, X_val, X_eval, y_train, y_test, y_val, y_eval = DataLoader.data_split(X, y, test_size, val_size, eval_size, random_seed)
            X_train.drop(columns=["experiment_num", "Tracer"], inplace=True)
            X_test.drop(columns=["experiment_num", "Tracer"], inplace=True)
            X_val.drop(columns=["experiment_num", "Tracer"], inplace=True)
            X_eval.drop(columns=["experiment_num", "Tracer"], inplace=True)
            y_train_transformed = target_preprocessor.transform(y_train)
            y_val_transformed = target_preprocessor.transform(y_val)
            y_eval_transformed = target_preprocessor.transform(y_eval)
            return {
                'X_train': X_train,
                'X_val': X_val,
                'X_eval': X_eval,
                'X_test': X_test,
                'y_train': y_train_transformed,
                'y_val': y_val_transformed,
                'y_eval': y_eval_transformed,
                'y_test': y_test,
                # 'feature_scaler': feature_scaler,
                'target_preprocessor': target_preprocessor
            }
        elif val_size:
            X_train, X_test, X_val, y_train, y_test, y_val = DataLoader.data_split(X, y, test_size, valid_size = val_size, random_seed = random_seed)
            X_train.drop(columns=["experiment_num", "Tracer"], inplace=True)
            X_test.drop(columns=["experiment_num", "Tracer"], inplace=True)
            X_val.drop(columns=["experiment_num", "Tracer"], inplace=True)
            y_train_transformed = target_preprocessor.transform(y_train)
            y_val_transformed = target_preprocessor.transform(y_val)
            return {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train_transformed,
                'y_val': y_val_transformed,
                'y_test': y_test,
                'target_preprocessor': target_preprocessor
            }
        else:
            X_train, X_test, y_train, y_test = DataLoader.data_split(X, y, test_size, random_seed = random_seed)
            X_train.drop(columns=["experiment_num", "Tracer"], inplace=True)
            X_test.drop(columns=["experiment_num", "Tracer"], inplace=True)
            y_train_transformed = target_preprocessor.transform(y_train)
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train_transformed,
                'y_test': y_test,
                'target_preprocessor': target_preprocessor
            }


def plot_training_history(metrics: Dict[str, List[float]], save_path: str = None):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    plt.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
    if metrics.get('val_loss'):
        plt.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    else:
        plt.show()


def main():
    """Main execution function."""
    # Configuration
    config = {
        'random_seed': 42,
        'data_path': '/app/nse/ml',
        'features_filename': 'features_full.csv',
        'targets_filename': 'target_full.csv',
        'model_params': {
            'input_dim': 9,
            'hidden1': 512,
            'hidden2': 256,
            'hidden3': 64,
            'output_dim': 4
        },
        'training_params': {
            'learning_rate': 1e-4,
            'epochs': 50,
            'batch_size': 128,
            'early_stopping_patience': 20
        },
        'output_dir': './outputs'
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # Load data
    logger.info("Loading data...")
    X, y = DataLoader.load_from_folders(
        config['data_path'],
        config['features_filename'],
        config['targets_filename']
    )
    
    # Preprocess and split data
    logger.info("Preprocessing and splitting data...")
    data = DataLoader.preprocess_and_split(X, y, random_seed=config['random_seed'], 
        test_size = 0.2,
        val_size = 0.2)
    
    # Initialize model
    logger.info("Initializing model...")
    model = RegressionNet(**config['model_params'])
    
    # Initialize trainer
    trainer = ModelTrainer(model, learning_rate=config['training_params']['learning_rate'])
    
    # Train model
    logger.info("Starting training...")
    metrics = trainer.train(
        data['X_train'],
        data['y_train'],
        epochs=config['training_params']['epochs'],
        batch_size=config['training_params']['batch_size'],
        validation_data=(data['X_val'], data['y_val']),
        # early_stopping_patience=config['training_params']['early_stopping_patience']
    )
    
    # Plot and save training history
    plot_training_history(metrics, save_path=os.path.join(config['output_dir'], 'training_history.png'))
    
    # Evaluate model
    logger.info("Evaluating model...")
    y_pred, rmse_loss = trainer.evaluate(data['X_test'], data['y_test'])
    

    y_test_transformed = data['target_preprocessor'].transform(data['y_test'].copy())
    performance_visualizations(y_pred, y_test_transformed, filename_barplot = "nn_model_hist_transformed.png", 
                                                filename_compare = "nn_model_transformed.png",
                                                filename_text = "metrics_transformed.csv")

    # Inverse transform predictions
    y_pred_original = data['target_preprocessor'].inverse_transform(y_pred)
    
    logger.info(f"Final Test RMSE Loss: {rmse_loss:.4f}")
    
    # Visualize performance
    performance_visualizations(y_pred_original, data['y_test'], filename_barplot = "nn_model_hist.png", 
                                                filename_compare = "nn_model.png",
                                                filename_text = "metrics.csv")
    
    # Save model and metadata
    model_path = os.path.join(config['output_dir'], 'regression_neural_network.pth')
    trainer.save_model(
        model_path,
        metadata={
            # 'feature_scaler': data['feature_scaler'],
            'target_preprocessor': data['target_preprocessor'],
            'train_losses': metrics['train_loss'],
            'val_losses': metrics.get('val_loss'),
            'config': config
        }
    )


if __name__ == "__main__":
    main()