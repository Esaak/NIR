"""
CatBoost regression model for predicting target variables.
This module handles data loading, preprocessing, hyperparameter optimization, 
model training, and performance evaluation.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

from perform_visualization import perform_eda, perform_eda_short, performance_visualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TargetPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, near_zero: float = 1e-4):
        self.near_zero = near_zero
        self.stats = {}
    
    def fit(self, data: pd.DataFrame, *args) -> 'TargetPreprocessor':
        self.columns = data.columns
        
        # Store minimum values for standard deviations
        self.std_y_bias = np.min(data["c_std_y"])
        self.std_z_bias = np.min(data["c_std_z"])
        
        # Store standard deviations
        self.std_y_std = np.std(data["c_std_y"])
        self.std_z_std = np.std(data["c_std_z"])
        
        # Store mean values for standardization
        self.mean_y_mean = np.mean(data["c_mean_y"])
        self.mean_z_mean = np.mean(data["c_mean_z"])
        
        # Store standard deviations for standardization
        self.mean_y_std = np.std(data["c_mean_y"])
        self.mean_z_std = np.std(data["c_mean_z"])
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        
        # Apply log transformation to standard deviations
        data["c_std_y"] -= self.std_y_bias - self.near_zero
        data["c_std_z"] -= self.std_z_bias - self.near_zero
        data["c_std_y"] = np.log(data["c_std_y"])
        data["c_std_z"] = np.log1p(data["c_std_z"])
        
        # Standardize mean values
        data["c_mean_y"] = (data["c_mean_y"] - self.mean_y_mean) / self.mean_y_std
        data["c_mean_z"] = (data["c_mean_z"] - self.mean_z_mean) / self.mean_z_std
        
        return data
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame.from_dict(data)
        
        # Inverse standardization of mean values
        data["c_mean_y"] = data["c_mean_y"] * self.mean_y_std + self.mean_y_mean
        data["c_mean_z"] = data["c_mean_z"] * self.mean_z_std + self.mean_z_mean
        
        # Inverse log transformation of standard deviations
        data["c_std_y"] = np.exp(data["c_std_y"])
        data["c_std_z"] = np.expm1(data["c_std_z"])
        data["c_std_y"] += self.std_y_bias - self.near_zero
        data["c_std_z"] += self.std_z_bias - self.near_zero
        
        return data


class DataLoader:
    @staticmethod
    def load_from_folders(
        base_path: str,
        feature_filename: str,
        target_filename: str,
        folder_pattern: str = r'output_*\d'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pattern = re.compile(folder_pattern)
        folder_paths = []
        
        # Find matching folders
        for folder_name in os.listdir(base_path):
            if pattern.match(folder_name):
                folder_paths.append(folder_name)
        
        # Use default folder if no matching folders found
        if not folder_paths:
            folder_paths = ["output_28_12_2024"]
        
        folder_paths = ["output_28_12_2024"]
        
        X = pd.DataFrame()
        y = pd.DataFrame()
        
        # Load data from each folder
        for folder in folder_paths:
            X_tmp = pd.read_csv(os.path.join(base_path, folder, feature_filename))
            y_tmp = pd.read_csv(os.path.join(base_path, folder, target_filename))
            
            # Clean up unnamed columns
            for df, is_unnamed in [(X_tmp, pd.isna(X_tmp.columns[0]) or str(X_tmp.columns[0]).startswith('Unnamed:')), 
                               (y_tmp, pd.isna(y_tmp.columns[0]) or str(y_tmp.columns[0]).startswith('Unnamed:'))]:
                if is_unnamed:
                    df.drop(df.columns[0], axis=1, inplace=True)
            
            # Ensure 'y' column exists
            if X_tmp.columns[0] != "y":
                col_y = np.ones(X_tmp.shape[0]) * 1000
                X_tmp.insert(0, "y", col_y)
            
            logger.info(f"Loaded data from {folder}: X shape={X_tmp.shape}, y shape={y_tmp.shape}")
            
            # Concatenate data
            X = pd.concat([X, X_tmp], axis=0)
            y = pd.concat([y, y_tmp], axis=0)
        
        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)
        
        return X, y
    
    @staticmethod
    def preprocess_and_split(
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.2,
        eval_size: float = 0.1,
        random_state: int = 42,
        filter_zeros: bool = True,
        filter_by_histogram: bool = True
    ) -> Dict[str, Union[pd.DataFrame, TargetPreprocessor]]:
        # Filter out rows with zero standard deviations
        if filter_zeros:
            mask = (y["c_std_y"] != 0) & (y["c_std_z"] != 0)
            X = X[mask]
            y = y[mask]
        
        # Filter outliers using histogram
        if filter_by_histogram:
            points = np.linspace(0, np.max(y["c_std_y"]), 200)
            quantiles = np.histogram(y["c_std_y"], points)
            hist_mode = quantiles[1][np.argmax(quantiles[0])]
            
            cut_mask = y["c_std_y"] >= hist_mode
            X = X[cut_mask]
            y = y[cut_mask]
        
        # Initial train/test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train/validation split
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state
        )
        
        # Train/eval split for hyperparameter tuning
        X_train_final, X_eval, y_train_final, y_eval = train_test_split(
            X_train, y_train, test_size=eval_size, random_state=random_state
        )
        
        # Target preprocessing
        target_preprocessor = TargetPreprocessor()
        target_preprocessor.fit(pd.concat([y_temp, y_test]))
        
        # Transform target variables
        y_train_transformed = target_preprocessor.transform(y_train_final)
        y_valid_transformed = target_preprocessor.transform(y_valid)
        y_eval_transformed = target_preprocessor.transform(y_eval)
        
        return {
            'X_train': X_train_final,
            'X_valid': X_valid,
            'X_eval': X_eval,
            'X_test': X_test,
            'y_train': y_train_transformed,
            'y_train_original': y_train_final,
            'y_valid': y_valid_transformed,
            'y_valid_original': y_valid,
            'y_eval': y_eval_transformed,
            'y_eval_original': y_eval,
            'y_test': y_test,
            'target_preprocessor': target_preprocessor
        }


class CatBoostOptimizer:
    
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_eval: pd.DataFrame,
        y_eval: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_valid: pd.DataFrame,
        target_column: str,
        random_state: int = 42,
        n_trials: int = 100,
        early_stopping_rounds: int = 20,
        gpu_enabled: bool = True
    ):
        self.X_train = X_train
        self.y_train = y_train[target_column]
        self.X_eval = X_eval
        self.y_eval = y_eval[target_column]
        self.X_valid = X_valid
        self.y_valid = y_valid[target_column]
        self.target_column = target_column
        self.random_state = random_state
        self.n_trials = n_trials
        self.early_stopping_rounds = early_stopping_rounds
        self.gpu_enabled = gpu_enabled
        self.study = None
        self.best_params = None
    
    def objective(self, trial: optuna.Trial) -> float:
        param = {
            'learning_rate': trial.suggest_float("learning_rate", 0.0001, 0.02),
            'depth': trial.suggest_int('depth', 1, 16),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0, 3.0),
            'min_child_samples': trial.suggest_categorical('min_child_samples', [1, 4, 8, 16, 32]),
            'grow_policy': 'SymmetricTree',
            'iterations': 300,
            'use_best_model': True,
            'eval_metric': 'RMSE',
            'loss_function': 'RMSE',
            'od_type': 'iter',
            'task_type': "GPU",
            'od_wait': 20,
            'random_state': self.random_state,
            'logging_level': 'Silent'
        }
        
        # Initialize and train CatBoost model
        regressor = CatBoostRegressor(**param)
        regressor.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_eval, self.y_eval)],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=False
        )
        
        # Calculate validation loss
        predictions = regressor.predict(self.X_valid)
        loss = root_mean_squared_error(self.y_valid, predictions)
        
        return loss
    
    def optimize(self) -> Dict[str, Any]:
        logger.info(f"Starting hyperparameter optimization for {self.target_column}")
        self.study = optuna.create_study(study_name=f'catboost-{self.target_column}-seed{self.random_state}')
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        self.best_params = self.study.best_params
        logger.info(f"Best RMSE: {self.study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def get_final_params(self, iterations: int = 5000) -> Dict[str, Any]:
        if self.best_params is None:
            raise ValueError("Optimization has not been run yet. Call optimize() first.")
        
        final_params = {
            "learning_rate": self.best_params['learning_rate'],
            "depth": self.best_params['depth'],
            "l2_leaf_reg": self.best_params['l2_leaf_reg'],
            "min_child_samples": self.best_params['min_child_samples'],
            "grow_policy": 'SymmetricTree',
            "iterations": iterations,
            "use_best_model": True,
            "eval_metric": 'RMSE',
            "loss_function": 'RMSE',
            "od_type": 'iter',
            "od_wait": 20,
            "task_type": "GPU" if self.gpu_enabled else "CPU",
            "random_state": self.random_state,
            "logging_level": 'Silent'
        }
        
        return final_params


class ModelTrainer:
    
    def __init__(
        self, 
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_eval: pd.DataFrame,
        y_eval: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        target_preprocessor: TargetPreprocessor,
        params: Dict[str, Any],
        early_stopping_rounds: int = 100,
        output_dir: str = "./outputs"
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.X_test = X_test
        self.y_test = y_test
        self.target_preprocessor = target_preprocessor
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        self.output_dir = output_dir
        self.models = {}
        self.predictions = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def train_all_targets(self) -> Dict[str, CatBoostRegressor]:
        logger.info("Training models for all target variables")
        target_columns = ["c_mean_y", "c_mean_z", "c_std_y", "c_std_z"]
        
        plt.figure(figsize=(12, 8))
        
        for target in target_columns:
            logger.info(f"Training model for {target}")
            model = CatBoostRegressor(**self.params)
            
            model.fit(
                self.X_train,
                self.y_train[target],
                eval_set=[(self.X_eval, self.y_eval[target])],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=100
            )
            
            self.models[target] = model
            predictions = model.predict(self.X_test)
            self.predictions[target] = predictions
            
            # Plot training loss
            if model.get_evals_result() and "learn" in model.get_evals_result():
                loss_values = model.get_evals_result()["learn"].get("RMSE") or model.get_evals_result()["learn"].get("Logloss")
                if loss_values:
                    plt.plot(np.arange(1, len(loss_values)+1), loss_values, label=target)
        
        plt.title("Training Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "training_loss.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        return self.models
    
    def evaluate(self) -> Dict[str, float]:
        if not self.predictions:
            raise ValueError("Models have not been trained yet. Call train_all_targets() first.")
        
        # Convert predictions to DataFrame for inverse transformation
        y_pred_df = pd.DataFrame(self.predictions)
        
        # Inverse transform predictions
        y_pred_original = self.target_preprocessor.inverse_transform(y_pred_df)
        
        # Calculate metrics
        metrics = {}
        for target in y_pred_original.columns:
            rmse = root_mean_squared_error(self.y_test[target], y_pred_original[target])
            r2 = r2_score(self.y_test[target], y_pred_original[target])
            metrics[f"{target}_rmse"] = rmse
            metrics[f"{target}_r2"] = r2
            logger.info(f"{target} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Visualize performance
        performance_visualizations(y_pred_original, self.y_test)
        
        return metrics
    
    def save_models(self):
        if not self.models:
            raise ValueError("Models have not been trained yet. Call train_all_targets() first.")
        
        model_dir = os.path.join(self.output_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        
        for target, model in self.models.items():
            model_path = os.path.join(model_dir, f"catboost_{target}.cbm")
            model.save_model(model_path)
            logger.info(f"Model for {target} saved to {model_path}")
        
        # Save target preprocessor
        preprocessor_path = os.path.join(model_dir, "target_preprocessor.pkl")
        import pickle
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.target_preprocessor, f)
        logger.info(f"Target preprocessor saved to {preprocessor_path}")


def main():
    # Configuration
    config = {
        'random_seed': 42,
        'early_stopping_rounds': 100,
        'data_path': os.getcwd(),
        'features_filename': 'features_full.csv',
        'targets_filename': 'target_full.csv',
        'output_dir': './outputs',
        'optuna_trials': 100,
        'gpu_enabled': True,
        'final_iterations': 5000
    }
    
    # Set random seed for reproducibility
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
    data = DataLoader.preprocess_and_split(
        X, y, 
        random_state=config['random_seed'],
        filter_zeros=True,
        filter_by_histogram=True
    )
    
    # Optional: Exploratory Data Analysis
    # perform_eda(data['X_train'], data['y_train_original'])
    
    # Hyperparameter optimization for c_mean_y (representative target)
    logger.info("Starting hyperparameter optimization...")
    optimizer = CatBoostOptimizer(
        data['X_train'],
        data['y_train'],
        data['X_eval'],
        data['y_eval'],
        data['X_valid'],
        data['y_valid'],
        target_column='c_mean_y',
        random_state=config['random_seed'],
        n_trials=config['optuna_trials'],
        early_stopping_rounds=config['early_stopping_rounds'],
        gpu_enabled=config['gpu_enabled']
    )
    optimizer.optimize()
    
    # Get final parameters
    final_params = optimizer.get_final_params(iterations=config['final_iterations'])
    
    # Train models for all targets
    logger.info("Training final models...")
    trainer = ModelTrainer(
        data['X_train'],
        data['y_train'],
        data['X_eval'],
        data['y_eval'],
        data['X_test'],
        data['y_test'],
        data['target_preprocessor'],
        params=final_params,
        early_stopping_rounds=config['early_stopping_rounds'],
        output_dir=config['output_dir']
    )
    
    trainer.train_all_targets()
    
    # Evaluate models
    logger.info("Evaluating models...")
    metrics = trainer.evaluate()
    
    # Save models
    logger.info("Saving models...")
    trainer.save_models()
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()