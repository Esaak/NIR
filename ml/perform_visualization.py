
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error  
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn


sns.set_theme(font = 'serif')
def perform_eda(X, y):
    sns.set_context("poster")

    """Perform Exploratory Data Analysis"""
    # Feature distributions
    shape_col_f = 3
    shape_row_f = X.shape[1] // shape_col_f + int(X.shape[1] % shape_col_f != 0)

    shape_col_t = 2        
    shape_row_t = y.shape[1] // shape_col_t + int(y.shape[1] % shape_col_t != 0) 
    
    fig, axes = plt.subplots(shape_row_f, shape_col_f, figsize=(10 * shape_row_f/shape_col_f, 10))

    for idx, col in enumerate(X.columns):
        row = idx // shape_col_f
        col_idx = idx % shape_row_f
        sns.histplot(X[col], ax=axes[row, col_idx], bins = 50)
        # if col not in ['sensible_heat_flux', 'power', 'T_grad', 'distances', 'roughness', 'T']:
        #     axes[row, col_idx].set_title(f'Distribution of {col}')
        # else:
        axes[row, col_idx].set_title(f'Distribution of \n {col}')
        if col ==  'y':
               axes[row, col_idx].set_xticks(np.linspace(X[col].min(), X[col].max(), 2).astype(int)//10 * 10)
        axes[row, col_idx].set(xlabel=None, ylabel=None)
        axes[row, col_idx].ticklabel_format(style='scientific', axis='y', scilimits=[-4, 4])

    plt.tight_layout(h_pad=0.5, w_pad=0.4)
    plt.savefig('tmp.png', dpi=300)
    plt.show()
    
    # Correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', annot_kws={'size': 12}, fmt='.3f')
    plt.title('Feature Correlation Matrix')
    plt.show()
    
    # Target variable distributions
    fig, axes = plt.subplots(shape_row_t, shape_col_t, figsize=(10 * shape_row_t/shape_col_t, 10))
    for idx, col in enumerate(y.columns):
        row = idx // shape_col_t
        col_idx = idx % shape_row_t
        sns.histplot(y[col], ax=axes[row, col_idx], bins = 50)
        axes[row, col_idx].set_title(f'Distribution of {col}')
        axes[row, col_idx].set(xlabel=None, ylabel=None)
        axes[row, col_idx].ticklabel_format(style='scientific', axis='y', scilimits=[-4, 4])

    plt.tight_layout()
    plt.show()
    
    # Feature-target correlations
    plt.figure(figsize=(8, 6))
    correlations = pd.concat([X, y], axis=1).corr().iloc[:X.shape[1], X.shape[1]:]
    sns.heatmap(correlations, annot=True, cmap='coolwarm', annot_kws={'size': 16})
    plt.title('Feature-Target Correlations')
    plt.show()

def density_scatter(x, y, ax=None, bins=20, cmap='viridis', **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    
    x_copy = x.copy().to_numpy()
    # Calculate the 2D histogram
    data, x_e, y_e = np.histogram2d(x_copy, y, bins=bins, density=True)
    
    # Interpolate the density values for each point
    x_centers = 0.5 * (x_e[1:] + x_e[:-1])
    y_centers = 0.5 * (y_e[1:] + y_e[:-1])
    z = interpn((x_centers, y_centers), data, np.vstack([x_copy, y]).T, 
                method="splinef2d", bounds_error=False)
    
    # Set NaN values to 0
    z[np.where(np.isnan(z))] = 0.0
    
    # Sort the points by density so that densest points are plotted last
    idx = z.argsort()
    x_sorted, y_sorted, z_sorted = x_copy[idx], y[idx], z[idx]
    
    # Create a scatter plot with beauty improvements
    sc = ax.scatter(x_sorted, y_sorted, c=z_sorted, cmap=cmap, alpha=1, 
                   s=kwargs.get('s', 30), edgecolor=kwargs.get('edgecolor', 'none'))
    
    # Set background color and grid
    
    return ax

def perform_eda_short(X, y):
    
    """Perform Exploratory Data Analysis without correlation"""
    # Feature distributions
    shape_col_f = 3
    shape_row_f = X.shape[1] // shape_col_f + int(X.shape[1] % shape_col_f != 0)

    shape_col_t = 2        
    shape_row_t = y.shape[1] // shape_col_t + int(y.shape[1] % shape_col_t != 0) 
    
    fig, axes = plt.subplots(shape_row_f, shape_col_f, figsize=(8 * shape_row_f/shape_col_f, 8))
    for idx, col in enumerate(X.columns):
        row = idx // shape_col_f
        col_idx = idx % shape_row_f
        sns.histplot(X[col], ax=axes[row, col_idx], bins = 50)
        axes[row, col_idx].set_title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()
    
    # Target variable distributions
    fig, axes = plt.subplots(shape_row_t, shape_col_t, figsize=(8 * shape_row_t/shape_col_t, 8))
    for idx, col in enumerate(y.columns):
        row = idx // shape_col_t
        col_idx = idx % shape_row_t
        sns.histplot(y[col], ax=axes[row, col_idx], bins = 50)
        axes[row, col_idx].set_title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()
    
def performance_visualizations(y_pred, y_test, transform = False):
    sns.set_context("talk")
    metrics = {'mse': {}, 'r2': {}, 'rmse': {}}
    
    for target in y_test.columns:
        y_true = y_test[target]
        metrics['mse'][target] = [mean_squared_error(y_true, y_pred[target])]
        metrics['r2'][target] = [r2_score(y_true, y_pred[target])]
        metrics['rmse'][target] = [root_mean_squared_error(y_true, y_pred[target])]
        
    # Plot MSE comparison
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    mse_df = pd.DataFrame.from_dict(metrics['mse']).melt()
    sns.barplot(x='variable', y='value', data=mse_df)
    plt.title('Mean Squared Error by Model and Target')
    plt.xticks(rotation=45)
    
    # Plot R² comparison
    plt.subplot(1, 3, 2)
    r2_df = pd.DataFrame(metrics['r2']).melt()
    sns.barplot(x='variable', y='value', data=r2_df)
    plt.title('R² Score by Model and Target')
    plt.xticks(rotation=45)

    # Plot RMSE comparison
    plt.subplot(1, 3, 3)
    rmse_df = pd.DataFrame(metrics['rmse']).melt()
    sns.barplot(x='variable', y='value', data=rmse_df)
    plt.title('RMSE Score by Model and Target')
    plt.xticks(rotation=45)
    plt.show()
    # plt.savefig("catboost_model_hist.png")
    
    # Actual vs Predicted plots
    shape_col = 2
    shape_row = y_test.shape[1] // shape_col + int(y_test.shape[1] % shape_col != 0) 
    
    fig, axes = plt.subplots(shape_row, shape_col, figsize=(6, 6))
    # fig.suptitle(f'True vs Predicted')
    for idx, target in enumerate(y_test.columns):
        row = idx // shape_row
        col = idx % shape_col
        # sns.scatterplot(
        #                 x=y_test[target],
        #                 y=y_pred[target],
        #                 alpha=0.8,
        #                 s=10,
        #                 ax=axes[row, col],
        #             )
        # sns.kdeplot(
        #                 x=y_test[target],
        #                 y=y_pred[target],
        #                 # levels=5,
        #                 fill=True,
        #                 alpha=0.8,
        #                 cmap='viridis',
        #                 ax=axes[row, col],
        #             )
        density_scatter(x=y_test[target],
                        y=y_pred[target],
                        ax=axes[row, col])
        # axes[row, col].scatter(y_test[target], y_pred[target], alpha=0.5)
        # axes[row, col].plot([y_test[target].min(), y_test[target].max()],
        #                         [y_test[target].min(), y_test[target].max()], 'r--', lw=2)
        axes[row, col].plot([y_test[target].min(), y_test[target].max()],
                                [y_test[target].min(), y_test[target].max()], 'r--', lw=2)
        
        if target == 'c_delta_z':
            axes[row, col].set_title(f'$\Delta_z$', fontsize=36)
        elif target == 'c_delta_y':
            axes[row, col].set_title(f'$\Delta_y$', fontsize=36)
        elif target == 'c_std_z':
            axes[row, col].set_title(f'$\sigma_z$', fontsize=36)
        elif target == 'c_std_y':
            axes[row, col].set_title(f'$\sigma_y$', fontsize=36)
        else: 
            axes[row, col].set_title(f'{target}')
        
        if transform:  
            axes[row, col].set_xlabel('Истина')
            axes[row, col].set_ylabel('Прогноз')
        else:
            axes[row, col].set_xlabel('Истина, м')
            axes[row, col].set_ylabel('Прогноз, м')  
        # axes[row, col].set_xlabel('True, m')
        # axes[row, col].set_ylabel('Predicted, m')            
    plt.tight_layout()
    # plt.savefig("catboost_model.png")
    plt.show()
    
    metrics_df = pd.DataFrame.from_dict(metrics)
    metrics_df = metrics_df.map(lambda x: x[0] if isinstance(x, list) else x)
    metrics_df = metrics_df.astype(float)
    print(metrics_df)


def feature_importances_sklearn(models_dict, y, feature_columns):
    sns.set_context("talk")
    shape_col_t = 2        
    shape_row_t = 2
    fig, axes = plt.subplots(shape_row_t, shape_col_t, figsize=(10, 10)) 
    sns.set_theme(style="whitegrid")
    for idx, col in enumerate(y.columns):
        row = idx // shape_col_t
        col_idx = idx % shape_row_t
        feature_importance = models_dict[col].feature_importances_
        sorted_idx = np.argsort(feature_importance)
        sns.barplot(x = feature_importance[sorted_idx], y=np.array(feature_columns)[sorted_idx], ax=axes[row, col_idx])
        # axes[row, col_idx].set_yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
        axes[row, col_idx].set_title(f'Feature importance {col}')

def feature_importances_catboost(models_dict, target_columns, feature_columns):
    sns.set_context("talk")
    shape_col_t = 2        
    shape_row_t = 2
    fig, axes = plt.subplots(shape_row_t, shape_col_t, figsize=(10, 10)) 
    sns.set_theme(style="whitegrid")
    for idx, col in enumerate(target_columns):
        row = idx // shape_col_t
        col_idx = idx % shape_row_t
        feature_importance = models_dict[col].get_feature_importance()
        sorted_idx = np.argsort(feature_importance)
        sns.barplot(x = feature_importance[sorted_idx], y=np.array(feature_columns)[sorted_idx], ax=axes[row, col_idx])
        # axes[row, col_idx].set_yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
        axes[row, col_idx].set_title(f'Feature importance {col}')