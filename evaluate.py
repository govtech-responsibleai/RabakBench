"""
evaluate.py
"""
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import f1_score, precision_score, recall_score

from config import RABAKBENCH_CATEGORIES, MAP_CONFIG

def remap_azure(df_mod):
    """
    For special cases such as Azure where non_zero labels do not indicate "Unsafe" behaviour.
    Assign a negative score to such cases. For e.g., Hate level 1 does not indicate unsafe behaviour.
    """
    for k, v in MAP_CONFIG['azure'].items():
        for severity, cat in v.items():
            if len(cat) == 0:
                df_mod.loc[df_mod[k] == int(severity), k] = 0
    return df_mod

def get_mappable_columns(df_true, df_mod, moderator_name):
    """Remove categories that are not relevant from both dataframes."""
    map_config = MAP_CONFIG[moderator_name]
    
    true_subcategories = set()
    mod_categories = set()
    for k, v in map_config.items():
        if moderator_name == 'azure':
            cat_len = 0
            for _, v1 in v.items():
                true_subcategories.update(v1)
                cat_len += len(v1)
            if cat_len > 0:
                mod_categories.add(k)
        else:
            true_subcategories.update(v)
            if len(v) > 0:
                mod_categories.add(k)
    
    # Get main categories for valid sub_categories
    true_categories = [k for k, v in RABAKBENCH_CATEGORIES.items() if any(c in true_subcategories for c in v)]
    mod_categories = list(mod_categories)
    
    # If no valid categories are found, default to all categories
    if len(true_categories) == 0:
        true_categories = list(RABAKBENCH_CATEGORIES.keys())
        mod_categories = list(map_config.keys())
    
    return df_true[true_categories], df_mod[mod_categories]
    
def bootstrap_metrics(y_true, y_pred, n_iterations=1000, random_seed=42):
    """Calculate bootstrapped metrics (F1, Recall, Precision) and their standard errors."""
    np.random.seed(random_seed)
    indices = np.arange(len(y_true))
    
    metrics = {
        'F1': [],
        'R': [],
        'P': []
    }
    
    for _ in range(n_iterations):
        # Sample with replacement
        bootstrap_indices = np.random.choice(indices, size=len(indices), replace=True)
        bootstrap_true = y_true[bootstrap_indices]
        bootstrap_pred = y_pred[bootstrap_indices]
        
        # Calculate metrics for this bootstrap sample
        metrics['F1'].append(f1_score(bootstrap_true, bootstrap_pred))
        metrics['R'].append(recall_score(bootstrap_true, bootstrap_pred))
        metrics['P'].append(precision_score(bootstrap_true, bootstrap_pred))
    
        # Calculate mean, std, and 95% CI for each metric
    results = {}
    for metric_name in metrics:
        values = metrics[metric_name]
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Calculate 95% confidence interval using t-distribution
        # For bootstrap samples, we can use the percentile method or t-distribution
        # Here we use the t-distribution approach
        n = len(values)
        sem = std_val / np.sqrt(n)
        t_critical = stats.t.ppf(0.975, df=n-1)  # 95% CI (two-tailed)
        margin_of_error = t_critical * sem
        
        # Store results
        results[f'{metric_name}_score'] = mean_val
        results[f'{metric_name}_std'] = std_val
        results[f'{metric_name}_ci_lower'] = mean_val - margin_of_error
        results[f'{metric_name}_ci_upper'] = mean_val + margin_of_error
        
    return results

def map_and_evaluate(lang='en'):
    print(f"Evaluating {lang}...")
    # Read ground truth data
    filename = f'data/{lang}/rabakbench_{lang}.csv'
    if os.path.exists(filename):
        df_true = pd.read_csv(filename)
    else:
        raise FileNotFoundError(f"File not found: {filename}")
    
    results = []
    
    # Process each moderator
    for i, moderator in enumerate(MAP_CONFIG.keys()):
        print(f"===> {moderator}...")
        try:
            df_mod = pd.read_csv(f'data/{lang}/rabakbench_{lang}_{moderator}.csv')
            if moderator == 'azure':
                df_mod = remap_azure(df_mod)
        except FileNotFoundError:
            print(f"File not found: data/{lang}/rabakbench_{lang}_{moderator}.csv")
            continue
        
        # Filter out rows with errors
        df_mod = df_mod.dropna(subset=list(MAP_CONFIG[moderator].keys())) 
        
        common_rows = df_true.merge(df_mod, on='prompt_id', how='inner')['prompt_id']
        
        df_true_common = df_true[df_true['prompt_id'].isin(common_rows)]
        df_mod_common = df_mod[df_mod['prompt_id'].isin(common_rows)]
        df_true_common = df_true_common.set_index('prompt_id').reindex(common_rows).reset_index()
        df_mod_common = df_mod_common.set_index('prompt_id').reindex(common_rows).reset_index()
        
        # Calculate binary labels
        df_true_common, df_mod_common = get_mappable_columns(df_true_common, df_mod_common, moderator)
        y_true = df_true_common.astype(int).astype(bool).max(axis=1)
        y_pred = df_mod_common.astype(int).astype(bool).max(axis=1)
        
        # Get all metrics in one go, using a different seed for each moderator
        metrics = bootstrap_metrics(y_true, y_pred)
        results.append({
            'moderator': moderator,
            **metrics,
        })
    
    # Create results DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'evaluation_results_{lang}.csv', index=False)
    print("\nEvaluation Results:")
    print(results_df.to_string(index=False))
    
def main():
    for lang in ['en', 'ms', 'ta', 'zh']:
        map_and_evaluate(lang)

if __name__ == "__main__":
    main()