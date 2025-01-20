
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error  

def perform_eda(X, y):
    """Perform Exploratory Data Analysis"""
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
    
    # Correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
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
    
    # Feature-target correlations
    plt.figure(figsize=(8, 6))
    correlations = pd.concat([X, y], axis=1).corr().iloc[:X.shape[1], X.shape[1]:]
    sns.heatmap(correlations, annot=True, cmap='coolwarm')
    plt.title('Feature-Target Correlations')
    plt.show()
    
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
    
def performance_visualizations(y_pred, y_test):
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
    
    
    # Actual vs Predicted plots
    shape_col = 2
    shape_row = y_test.shape[1] // shape_col + int(y_test.shape[1] % shape_col != 0) 
    
    fig, axes = plt.subplots(shape_row, shape_col, figsize=(6, 6))
    fig.suptitle(f'Actual vs Predicted')
    for idx, target in enumerate(y_test.columns):
        row = idx // shape_row
        col = idx % shape_col
        
        axes[row, col].scatter(y_test[target], y_pred[target], alpha=0.5)
        axes[row, col].plot([y_test[target].min(), y_test[target].max()],
                                [y_test[target].min(), y_test[target].max()], 'r--', lw=2)
        axes[row, col].set_title(f'{target}')
        axes[row, col].set_xlabel('Actual')
        axes[row, col].set_ylabel('Predicted')            
    plt.tight_layout()
    plt.show()
    
    metrics_df = pd.DataFrame.from_dict(metrics)
    metrics_df = metrics_df.map(lambda x: x[0] if isinstance(x, list) else x)
    metrics_df = metrics_df.astype(float)
    print(metrics_df)