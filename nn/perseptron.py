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
from tqdm import tqdm
from torch.nn import _reduction as _Reduction, functional as F
from torch.utils.tensorboard import SummaryWriter

from perform_visualization import performance_visualizations
from torch.utils.data import Dataset
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
        # data["c_delta_y"] = np.log(data["c_delta_y"]+1e-6)
        
        data["c_delta_y"]*=mean_y_sign
        
        data["c_delta_z"] = (data["c_delta_z"] - self.stats["c_delta_z_mean"])/self.stats["c_delta_z_std"]
        mean_z_sign = np.sign(data["c_delta_z"])
        data["c_delta_z"]= mean_z_sign * np.log1p(np.abs(data["c_delta_z"]))
        # data["c_delta_z"]= mean_z_sign * np.log(np.abs(data["c_delta_z"])+1e-6)

        return data
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        
        # Reverse log transformation and restore standard deviations
        if 'c_std_y' in data.columns:
            data['c_std_y'] = np.exp(data['c_std_y'])
            data['c_std_y'] += self.stats[f'c_std_y_min'] - self.near_zero
        
        if 'c_std_z' in data.columns:
            data['c_std_z'] = np.expm1(data['c_std_z'])
            data['c_std_z'] += self.stats[f'c_std_z_min'] - self.near_zero
        
        if 'c_delta_y' in data.columns:
            mean_y_sign = np.sign(data["c_delta_y"])
            data["c_delta_y"] = np.abs(data["c_delta_y"])
            data["c_delta_y"] = np.expm1(data["c_delta_y"])
            data["c_delta_y"] *=mean_y_sign
        
        if 'c_delta_z' in data.columns:
            mean_z_sign = np.sign(data["c_delta_z"])
            data["c_delta_z"] = np.expm1(np.abs(data["c_delta_z"]))
            data["c_delta_z"] *=mean_z_sign
            data["c_delta_z"] = data["c_delta_z"] * self.stats["c_delta_z_std"] + self.stats["c_delta_z_mean"]

        return data

class _Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class RMSELoss(_Loss):
    __constants__ = ["reduction"]

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input_: torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        # return torch.sqrt(F.mse_loss(input, target, reduction=self.reduction, weight = batch_weight))
        return torch.sqrt(F.mse_loss(input_, target, reduction=self.reduction))


class IndexedDataset(Dataset):
    def __init__(self, X, y, weights=None):
        self.X = X
        self.y = y
        self.weights = weights if weights is not None else torch.ones(len(X))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx], idx



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
        
        folder_paths = [os.path.join(base_path, "output_19_01_2025_2"), os.path.join(base_path, "output_21_03_2025")]
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
        feature_scaler = StandardScaler()
        # X_scaled = feature_scaler.fit_transform(X)
        target_preprocessor = TargetPreprocessor()

        target_preprocessor.fit(y)

        if val_size and eval_size:
            X_train, X_test, X_val, X_eval, y_train, y_test, y_val, y_eval = DataLoader.data_split(X, y, test_size, val_size, eval_size, random_seed)
            X_train.drop(columns=["experiment_num", "Tracer"], inplace=True)
            X_test.drop(columns=["experiment_num", "Tracer"], inplace=True)
            X_val.drop(columns=["experiment_num", "Tracer"], inplace=True)
            X_eval.drop(columns=["experiment_num", "Tracer"], inplace=True)
            X_train_np = feature_scaler.fit_transform(X_train)
            X_val_np = feature_scaler.transform(X_val)
            X_eval_np = feature_scaler.transform(X_eval)
            X_test_np = feature_scaler.transform(X_test)
            X_train = pd.DataFrame(data=X_train_np, columns=X_train.columns)
            X_val = pd.DataFrame(data=X_val_np, columns=X_val.columns)
            X_eval = pd.DataFrame(data=X_eval_np, columns=X_eval.columns)
            X_test = pd.DataFrame(data=X_test_np, columns=X_test.columns)

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
                'feature_scaler': feature_scaler,
                'target_preprocessor': target_preprocessor
            }
        elif val_size:
            X_train, X_test, X_val, y_train, y_test, y_val = DataLoader.data_split(X, y, test_size, valid_size = val_size, random_seed = random_seed)
            X_train.drop(columns=["experiment_num", "Tracer"], inplace=True)
            X_test.drop(columns=["experiment_num", "Tracer"], inplace=True)
            X_val.drop(columns=["experiment_num", "Tracer"], inplace=True)
            X_train_np = feature_scaler.fit_transform(X_train)
            X_val_np = feature_scaler.transform(X_val)
            X_test_np = feature_scaler.transform(X_test)
            X_train = pd.DataFrame(data=X_train_np, columns=X_train.columns)
            X_val = pd.DataFrame(data=X_val_np, columns=X_val.columns)
            X_test = pd.DataFrame(data=X_test_np, columns=X_test.columns)

            y_train_transformed = target_preprocessor.transform(y_train)
            y_val_transformed = target_preprocessor.transform(y_val)
            return {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train_transformed,
                'y_val': y_val_transformed,
                'y_test': y_test,
                'feature_scaler': feature_scaler,
                'target_preprocessor': target_preprocessor
            }
        else:
            X_train, X_test, y_train, y_test = DataLoader.data_split(X, y, test_size, random_seed = random_seed)
            X_train.drop(columns=["experiment_num", "Tracer"], inplace=True)
            X_test.drop(columns=["experiment_num", "Tracer"], inplace=True)
            X_train_np = feature_scaler.fit_transform(X_train)
            X_test_np = feature_scaler.transform(X_test)
            X_train = pd.DataFrame(data=X_train_np, columns=X_train.columns)
            X_test = pd.DataFrame(data=X_test_np, columns=X_test.columns)


            y_train_transformed = target_preprocessor.transform(y_train)
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train_transformed,
                'y_test': y_test,
                'feature_scaler': feature_scaler,
                'target_preprocessor': target_preprocessor
            }

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = None,
        learning_rate: float = 1e-3,
        device: str = None,
        output_dir: str = './model_checkpoints'
    ):
        assert(torch.cuda.is_available())
        self.device = torch.device('cuda')
        self.model = model.to(self.device)
        # self.criterion = criterion if criterion else RMSELoss(reduction='sum')
        self.criterion = criterion if criterion else RMSELoss()

        
        # Adaptive learning rate with ReduceLROnPlateau
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,  # Reduce learning rate by half
            patience=5,  # Wait for 5 epochs without improvement
            min_lr=1e-6,  # Minimum learning rate
            verbose=True
        )
        
        # Model checkpoint directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Training on: {self.device}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: pd.DataFrame,
        sample_weights: np.ndarray,
        tb_writer: SummaryWriter,
        epochs: int = 100,
        batch_size: int = 256,
        validation_data: Tuple[np.ndarray, pd.DataFrame] = None,
        early_stopping_patience: int = None
    ) -> Dict[str, List[float]]:
        # Convert data to tensors
        X_train_tensor = torch.FloatTensor(X_train.to_numpy()).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.to_numpy()).to(self.device)
        
        # n_samples = len(X_train_tensor)
        sample_weights = torch.FloatTensor(sample_weights).to(self.device)
        # Create data loader
        dataset = IndexedDataset(X_train_tensor, y_train_tensor, sample_weights)
        # dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        # indices = torch.randperm(n_samples)
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
        best_model_path = os.path.join(self.output_dir, 'best_model.pth')
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0

            for batch_X, batch_y, batch_weights, original_indices in dataloader:
            # for idx, batch in enumerate(dataloader):
                # batch_X, batch_y = batch
                if torch.isnan(batch_X).any().item():
                    logger.info(f" NaNs in batch_X: {torch.isnan(batch_X).any().item()}")
                if torch.isnan(batch_y).any().item():
                    logger.info(f" NaNs in batch_y: {torch.isnan(batch_y).any().item()}")                

                # Forward pass
                outputs = self.model(batch_X)
                
                min_weight = float('inf')
                max_weight = float('-inf')

                for name, param in self.model.named_parameters():
                    # Only look at weights, not biases
                    if 'weight' in name:
                        # Get min and max for this layer
                        layer_min = param.data.min().item()
                        layer_max = param.data.max().item()
                        
                        # Update overall min and max
                        min_weight = min(min_weight, layer_min)
                        max_weight = max(max_weight, layer_max)
                
                # logger.info(f" Min, Max in outputs. Min, Max in weight: {torch.min(outputs).item()}, {torch.max(outputs).item()}; {min_weight}, {max_weight}")
                
                if torch.isnan(outputs).any().item():
                    logger.info(f" NaNs in outputs: {torch.isnan(outputs).any().item()}")
                    # tb_writer.add_scalar('train_batch_loss', epoch_loss/(idx+1), global_step = epoch*(int(len(dataloader)/batch_size))+idx + 1)

                    # for tag, param in self.model.named_parameters():
                    #     tb_writer.add_histogram('grad/%s'%tag, param.grad.data.cpu().numpy(), global_step = epoch*(int(len(dataloader)/batch_size))+idx + 1)
                    #     tb_writer.add_histogram('weight/%s' % tag, param.data.cpu().numpy(), global_step = epoch*(int(len(dataloader)/batch_size))+idx + 1)    
                    # return 0            
                # logger.info(f"outputs_dim {outputs.shape}, batch_y_dim {batch_y.shape}")
                loss = self.criterion(outputs, batch_y)
                # loss = self.criterion(outputs, batch_y)

                if torch.isnan(loss).any().item():
                    logger.info(f" NaNs in loss: {torch.isnan(loss).any().item()}")                
                # loss = loss.mean()
                # loss = self.criterion(outputs, batch_y)


                # weighted_loss = (loss * batch_weights).mean()
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                # loss.backward()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                epoch_loss += loss.item()


                # if idx % 3 == 0:
                #     tb_writer.add_scalar('train_batch_loss', epoch_loss/(idx+1), global_step = epoch*(int(len(dataloader)/batch_size))+idx + 1)

                #     for tag, param in self.model.named_parameters():
                #         tb_writer.add_histogram('grad/%s'%tag, param.grad.data.cpu().numpy(), global_step = epoch*(int(len(dataloader)/batch_size))+idx + 1)
                #         tb_writer.add_histogram('weight/%s' % tag, param.data.cpu().numpy(), global_step = epoch*(int(len(dataloader)/batch_size))+idx + 1)
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
                    val_loss = self.criterion(val_outputs, y_val_tensor).mean().item()
                    metrics['val_loss'].append(val_loss)
                
                # Adaptive learning rate adjustment
                self.scheduler.step(val_loss)
                # tb_writer.add_scalar('val_loss', metrics['val_loss'][-1], global_step=epoch)
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
                    logger.info(f"Saved best model with validation loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                
                # Early stopping
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                log_msg = f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}'
                if validation_data:
                    log_msg += f', Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}'
                logger.info(log_msg)
        
        return metrics
    
    def evaluate(self, X_test: np.ndarray, y_test: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        # Load the best model if it exists
        best_model_path = os.path.join(self.output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model with validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test.to_numpy()).to(self.device)
            y_test_tensor = torch.FloatTensor(y_test.to_numpy()).to(self.device)
            
            test_outputs = self.model(X_test_tensor)
            
            rmse_loss = self.criterion(test_outputs, y_test_tensor).mean().item()
            
            # Convert predictions to DataFrame
            y_pred = pd.DataFrame(test_outputs.cpu().numpy(), columns=y_test.columns)
            
        return y_pred, rmse_loss
    
    def save_model(self, path: str, metadata: Dict = None):
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        if metadata:
            save_dict.update(metadata)
        
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")


def plot_training_history(metrics: Dict[str, List[float]], save_path: str = None):
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    # Loss subplot
    plt.subplot(2, 1, 1)
    plt.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
    if metrics.get('val_loss'):
        plt.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate subplot
    plt.subplot(2, 1, 2)
    plt.plot(epochs, metrics['learning_rates'], 'g-')
    plt.title('Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    else:
        plt.show()

def plot_eda(target: pd.DataFrame, filename: str) -> None:
    fig, ax = plt.subplots()
    target.hist(bins=50, ax=ax)
    fig.savefig(filename)


def plot_feature_importance(model, X_test, target_names, feature_names, save_path):
    sns.set_theme(font = 'serif')
    sns.set_context("talk")

    logger.info("Calculating SHAP values for feature importance...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create SHAP explainer
    background = X_test.sample(min(500, len(X_test)), random_state=42).to_numpy()
    background_tensor = torch.FloatTensor(background).to(model.model[0].weight.device)
    
    # Define a prediction function for SHAP
    def predict_fn(x):
        x_tensor = torch.FloatTensor(x).to(model.model[0].weight.device)
        with torch.no_grad():
            return model(x_tensor).cpu().numpy()
    
    # Initialize the SHAP explainer
    explainer = shap.DeepExplainer(model, background_tensor)
    
    # Calculate SHAP values
    X_test_sample = X_test.sample(min(1000, len(X_test)), random_state=42).to_numpy()
    shap_values = explainer.shap_values(torch.FloatTensor(X_test_sample).to(model.model[0].weight.device))
    

    sns.set_theme(style="whitegrid")
    shape_col_t = 2        
    shape_row_t = 2
    fig, axes = plt.subplots(shape_row_t, shape_col_t, figsize=(12, 10)) 
    for idx, col in enumerate(target_names):
        row = idx // shape_col_t
        col_idx = idx % shape_row_t
        shap_values_tmp = shap_values[:, :, idx]
        shap_values_tmp_abs = np.abs(shap_values_tmp)

        combined_shap_values = np.mean(shap_values_tmp_abs, axis=0)
        feature_importance = pd.DataFrame(list(zip(feature_names, combined_shap_values)), 
                                      columns=['Feature', 'Importance'])
        feature_importance_pd = pd.Series(combined_shap_values, feature_names)
        feature_importance_pd.to_csv('feature_importance_mpl_' + col + '.csv')
        feature_importance.sort_values(by=['Importance'], ascending=False, inplace=True)
        sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'Feature importance {col}')
    plt.savefig(save_path, dpi=300)



    logger.info(f"Feature importance plot saved to {save_path}")
    
    for idx, col in enumerate(target_names):
        plt.figure(figsize=(6, 8))
        shap_values_tmp = shap_values[:, :, idx]
        shap.summary_plot(shap_values_tmp, X_test_sample, feature_names=feature_names, show=False)
        summary_path = save_path.replace('.png', f'{col}_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')

    logger.info(f"SHAP summary plot saved to {save_path}")    
    return feature_importance


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
            "hidden1": 1024,
            'hidden2': 512,
            'hidden3': 256,
            'hidden4': 64,
            'output_dim': 1
        },
        'training_params': {
            'learning_rate': 1e-5,  # Increased from 1e-4
            'epochs': 1,  # Increased from 50
            'batch_size': 6144,
            'early_stopping_patience': 20
        },
        'output_dir': './outputs_19_03_2025',
        'model_checkpoints_dir': './model_checkpoints_29_03_2025'
    }
    tb_writer = SummaryWriter(log_dir='./outputs_29_03_2025/logs/run003/')
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['model_checkpoints_dir'], exist_ok=True)
    
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
    
    plot_eda(pd.DataFrame(data["y_test"]), config['output_dir'] + "/target_test_transform.png")
    plot_eda(pd.DataFrame(data["y_train"]), config['output_dir'] + "/target_train_transform.png")


    y_pred = {}
    for target in data["y_train"].columns[:1]:

        points = np.linspace(np.min(data["y_train"][target]), np.max(data["y_train"][target]), 201)
        density, bin_edges = np.histogram(data["y_train"][target], bins=points)
        bin_indices = np.digitize(data["y_train"][target], bin_edges) - 1    
        density_answ = []
        # density = np.log1p(density.astype(float))
        for i in density:
            if i !=0:
                density_answ.append(1.0/i)
            else:
                density_answ.append(0)
        density = np.array(density_answ)
        bin_indices = np.clip(bin_indices, 0, len(density) - 1)
        sample_weights = density[bin_indices]
        # Initialize model
        logger.info(f"Initializing model for {target}...")
        model = RegressionNet(**config['model_params'])
        
        # Initialize trainer with updated configuration
        trainer = ModelTrainer(
            model, 
            learning_rate=config['training_params']['learning_rate'],
            output_dir=config['model_checkpoints_dir']
        )
        
        # Train model
        logger.info("Starting training...")
        metrics = trainer.train(
            data['X_train'],
            # data['y_train'],
            pd.DataFrame(data=np.array(data['y_train'][target]), columns=[target]),
            sample_weights= sample_weights,
            tb_writer=tb_writer,
            epochs=config['training_params']['epochs'],
            batch_size=config['training_params']['batch_size'],
            # validation_data=(data['X_val'], data['y_val']),
            validation_data=(data['X_val'], pd.DataFrame(data=np.array(data['y_val'][target]), columns=[target])),

            early_stopping_patience=config['training_params']['early_stopping_patience']
        )
        
        # Plot and save training history
        plot_training_history(metrics, save_path=os.path.join(config['output_dir'], 'training_history.png'))
        
        # Evaluate model
        logger.info(f"Evaluating model for {target}...")
        # y_pred, rmse_loss = trainer.evaluate(data['X_test'], data['y_test'])
        y_pred_tmp, rmse_loss = trainer.evaluate(data['X_test'], pd.DataFrame(data=np.array(data['y_test'][target]), columns=[target]))
        y_pred[target] = y_pred_tmp  

    
    plot_eda(pd.DataFrame(y_pred), config['output_dir'] + "/target_pred_transform.png")

    y_test_transformed = data['target_preprocessor'].transform(data['y_test'].copy())

    performance_visualizations(y_pred, y_test_transformed,transform=True, filename_barplot = config['output_dir'] + "/nn_model_hist_transformed.png", 
                                                filename_compare = config['output_dir'] + "/nn_model_transformed.png",
                                                filename_text = config['output_dir'] + "/metrics_transformed.csv")

    # Inverse transform predictions
    y_pred_original = data['target_preprocessor'].inverse_transform(pd.DataFrame(y_pred))
    plot_eda(pd.DataFrame(y_pred_original), config['output_dir'] + "/target_pred.png")
    
    logger.info(f"Final Test RMSE Loss: {rmse_loss:.4f}")
    
    # Visualize performance
    performance_visualizations(y_pred_original, data['y_test'], filename_barplot = "nn_model_hist.png", 
                                                filename_compare = "nn_model.png",
                                                filename_text = "metrics.csv")

    feature_importance = plot_feature_importance(
        model,
        data['X_test'],
        data['y_test'].columns,
        data['X_test'].columns,
        os.path.join(config['output_dir'], 'feature_importance.png')
    )
    # Save final model and metadata
    model_path = os.path.join(config['output_dir'], 'regression_neural_network.pth')
    trainer.save_model(
        model_path,
        metadata={
            'feature_scaler': data['feature_scaler'],
            'target_preprocessor': data['target_preprocessor'],
            'train_losses': metrics['train_loss'],
            'val_losses': metrics.get('val_loss'),
            'learning_rates': metrics['learning_rates'],
            'config': config
        }
    )


if __name__ == "__main__":
    main()