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
    def data_split(X, y, test_size = 0.2, valid_size = None, eval_size = None, random_state=42):
        rng = np.random.default_rng(seed=random_state)
        
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
        test_size: float = 0.2,
        val_size: float = 0.2,
        eval_size: float = 0.1,
        random_state: int = 42
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
            X_train, X_test, X_val, X_eval, y_train, y_test, y_val, y_eval = DataLoader.data_split(X, y, test_size, val_size, eval_size, random_state)
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
                'X_valid': X_val,
                'X_eval': X_eval,
                'X_test': X_test,
                'y_train': y_train_transformed,
                'y_valid': y_val_transformed,
                'y_eval': y_eval_transformed,
                'y_test': y_test,
                'feature_scaler': feature_scaler,
                'target_preprocessor': target_preprocessor
            }
        elif val_size:
            X_train, X_test, X_val, y_train, y_test, y_val = DataLoader.data_split(X, y, test_size, valid_size = val_size, random_state = random_state)
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
                'X_valid': X_val,
                'X_test': X_test,
                'y_train': y_train_transformed,
                'y_valid': y_val_transformed,
                'y_test': y_test,
                'feature_scaler': feature_scaler,
                'target_preprocessor': target_preprocessor
            }
        else:
            X_train, X_test, y_train, y_test = DataLoader.data_split(X, y, test_size, random_state = random_state)
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
        n_trials: int = 500,
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
    
    def get_final_params(self, iterations: int = 1000) -> Dict[str, Any]:
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
            "task_type": "GPU",
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
        feature_scaler: StandardScaler,
        params: Dict[str, Dict[str, Any]],
        early_stopping_rounds: int = 100,
        output_dir: str = "./outputs_10_03_2025"
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.X_test = X_test
        self.y_test = y_test
        self.target_preprocessor = target_preprocessor
        self.feature_scaler = feature_scaler
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        self.output_dir = output_dir
        self.models = {}
        self.predictions = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def train_all_targets(self) -> Dict[str, CatBoostRegressor]:
        logger.info("Training models for all target variables")
        target_columns = ["c_delta_y", "c_delta_z", "c_std_y", "c_std_z"]
        
        plt.figure(figsize=(12, 8))
        
        for target in target_columns:
            logger.info(f"Training model for {target}")
            model = CatBoostRegressor(**self.params[target])
            
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
        y_test_transformed = self.target_preprocessor.transform(self.y_test)
        performance_visualizations(y_pred_df, y_test_transformed, filename_barplot = self.output_dir + "/cb_model_hist_transformed.png", 
                                                filename_compare = self.output_dir + "/cb_model_transformed.png",
                                                filename_text = self.output_dir + "/metrics_transformed.csv")
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
        performance_visualizations(y_pred_original, self.y_test, filename_barplot = self.output_dir + "/cb_model_hist.png", 
                                                filename_compare = self.output_dir + "/cb_model.png",
                                                filename_text = self.output_dir + "/metrics.csv")
        

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
        'random_state': 42,
        'early_stopping_rounds': 100,
        'data_path': "/app/nse/ml/",
        'features_filename': 'features_full.csv',
        'targets_filename': 'target_full.csv',
        'output_dir': './outputs_10_03_2025',
        'optuna_trials': 500,
        'gpu_enabled': True,
        'final_iterations': 1000
    }
    
    # Set random seed for reproducibility
    np.random.seed(config['random_state'])
    
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
        random_state=config['random_state']
    )
    
    # Optional: Exploratory Data Analysis
    # perform_eda(data['X_train'], data['y_train_original'])
    
    # Hyperparameter optimization 
    logger.info("Starting hyperparameter optimization...")
    final_params = {}
    for target_column in data["y_test"].columns:
        logger.info(f"Starting {target_column} optimization...")
        
        optimizer = CatBoostOptimizer(
            data['X_train'],
            data['y_train'],
            data['X_eval'],
            data['y_eval'],
            data['X_valid'],
            data['y_valid'],
            target_column= target_column,
            random_state=config['random_state'],
            n_trials=config['optuna_trials'],
            early_stopping_rounds=config['early_stopping_rounds'],
            gpu_enabled=config['gpu_enabled']
        )
        optimizer.optimize()
        
        # Get final parameters
        final_params[target_column] = optimizer.get_final_params(iterations=config['final_iterations'])
        
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
        data['feature_scaler'],
        params=final_params,
        early_stopping_rounds=config['early_stopping_rounds'],
        output_dir=config['output_dir']
    )
    
    models_final = trainer.train_all_targets()
    
    feature_importances_catboost(models_final, np.asarray(data["y_test"].columns), np.asarray(data["X_test"].columns),
                                 filename = config["output_dir"] + "/feature_importance.png")    
    # Evaluate models
    logger.info("Evaluating models...")
    metrics = trainer.evaluate()
    
    # Save models
    logger.info("Saving models...")
    trainer.save_models()
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()