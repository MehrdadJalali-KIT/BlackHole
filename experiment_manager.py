import json
import os
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("bh_evaluation.log", mode="a")
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
handler.buffering = 1
logger.addHandler(handler)
logger.propagate = False

def load_checkpoint(checkpoint_file):
    try:
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"Loaded checkpoint from {checkpoint_file}")
            return checkpoint
        else:
            logger.info(f"No checkpoint found at {checkpoint_file}, starting fresh")
            return {}
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return {}

def save_checkpoint(checkpoint_file, checkpoint):
    try:
        if not checkpoint_file:
            logger.error("Checkpoint file path is empty, skipping save")
            return
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=4)
        logger.info(f"Saved checkpoint to {checkpoint_file}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {checkpoint_file}: {e}")

def save_results(results, threshold, method, run, single_file=False, task='classification'):
    if single_file:
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{'metrics_regression' if task == 'regression' else 'final_results'}.csv")
        try:
            df = pd.DataFrame(results)
            if task == 'classification' and 'Confusion_Matrix' in df.columns:
                df['Confusion_Matrix'] = df['Confusion_Matrix'].apply(lambda x: str(x))
            df = df.sort_values(['Threshold', 'Method', 'Run', 'Model'])
            if os.path.exists(output_file):
                df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df.to_csv(output_file, mode='w', header=True, index=False)
            logger.info(f"Appended {len(df)} results to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {e}")
    else:
        output_dir = f"evaluation/threshold_{threshold:.2f}/method_{method}/run_{run}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{'metrics_regression' if task == 'regression' else 'results'}.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)
            logger.info(f"Saved results to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {e}")

def aggregate_results(results, num_runs, single_file=False, task='classification'):
    try:
        df = pd.DataFrame(results)
        if df.empty:
            logger.warning("No results to aggregate")
            return

        common_metrics = [
            'Num_Edges', 'Num_Nodes', 'Modularity', 'Num_Communities',
            'Avg_Community_Size', 'Avg_Clustering', 'Graph_Density', 'Avg_Degree',
            'Nodes_Removed', 'Edges_Removed', 'Percent_Nodes_Retained', 'Percent_Edges_Retained'
        ]
        metrics = common_metrics + (
            ['Accuracy', 'Cohen_Kappa'] if task == 'classification' else ['MAE', 'RMSE', 'R2']
        )
        metrics = [m for m in metrics if m in df.columns]

        agg_results = []
        for (threshold, method, model), group in df.groupby(['Threshold', 'Method', 'Model']):
            agg_dict = {
                'Threshold': threshold,
                'Method': method,
                'Model': model
            }
            for metric in metrics:
                mean_val = group[metric].mean()
                std_val = group[metric].std()
                agg_dict[f'{metric}_Mean'] = mean_val
                agg_dict[f'{metric}_Std'] = std_val
            agg_results.append(agg_dict)

        agg_df = pd.DataFrame(agg_results)

        if single_file:
            output_dir = "evaluation_results"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{'aggregated_regression' if task == 'regression' else 'aggregated_results'}.csv")
            try:
                agg_df.to_csv(output_file, index=False)
                logger.info(f"Saved aggregated results to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save aggregated results to {output_file}: {e}")
        else:
            output_file = f"{'results_regression_aggregated' if task == 'regression' else 'results_aggregated'}.csv"
            try:
                agg_df.to_csv(output_file, index=False)
                logger.info(f"Saved aggregated results to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save aggregated results to {output_file}: {e}")
    except Exception as e:
        logger.error(f"Failed to aggregate results: {e}")