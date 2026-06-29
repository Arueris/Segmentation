from tqdm import tqdm
import os
from tbparse import SummaryReader
import pandas as pd

def read_runs(run_dir, experiment_name):
    """
    Reads the runs from the specified directory and experiment name, extracts the hyperparameters and metrics, and returns a DataFrame with the results. 
    If a summary file already exists, it loads it instead of re-reading the runs.
    Args:
        run_dir (str): The directory where the runs are stored.
        experiment_name (str): The name of the experiment to read.
    Returns:
        pd.DataFrame: A DataFrame containing the hyperparameters and metrics for each run."""
    base_dir = os.path.join(run_dir, experiment_name)
    res_summary_save_path = f"results/{experiment_name}_summary.csv"
    if os.path.exists(res_summary_save_path):
        print(f"Summary file {res_summary_save_path} already exists. Loading it.")
        return pd.read_csv(res_summary_save_path, index_col=False)
    runs = os.listdir(base_dir)
    readers = {}
    for run in tqdm(runs, desc="Reading runs"):
        run_path = os.path.join(base_dir, run)
        reader = SummaryReader(run_path)
        # run_split = run.split("_")
        # run_name = run_split[0] + "_" + run_split[1]
        readers[run] = reader
    
    results = list()
    for run, reader in readers.items():
        if len(reader.hparams) == 0:
            print(f"Run {run} has no hyperparameters. Skipping.")
            continue
        hparams = reader.hparams.set_index("tag").T
        metrics = reader.scalars.loc[reader.scalars["tag"].apply(lambda x: "hparam" in x)]
        metrics["metric"] = metrics["tag"].apply(lambda x: x.split("/")[1])
        metrics = metrics[["metric", "value"]]
        metrics = metrics.set_index("metric").T
        # combine hparams and metrics
        combined = pd.concat([hparams, metrics], axis=1)
        results.append(combined)
    results_df = pd.concat(results, axis=0)
    results_df.to_csv(res_summary_save_path, index=False)
    return results_df