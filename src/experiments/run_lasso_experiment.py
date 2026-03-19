import multiprocessing
import os
import sys
import click
import logging
import logging.handlers

import numpy as np
import datetime
import json
from src.configs.grid_search_config import GridSearchConfig
from src.experiments.time_series_lasso_grid_search import TimeSeriesLassoGridSearch
from src.utils.logging_utils import setup_logging

# hardcoding output paths as I want to keep all history and snakemake deletes old files
GRID_CONFIG_PATH = '/GRID/configs/'
INPUT_CONFIG_FILEPATH = '/configs/POD/config.json'

@click.command()
@click.option('-input_base_path', type=click.Path(exists=True), help='Path to input base path (e.g. /data/2024-05-01_2024-05-14/)')
@click.option('-input_sub_path', type=click.Path(exists=True), help='Path input sub path for dataset (e.g. /processed_recomputed_decays_with_slopes_noman_fixed_interpolation/)')
@click.option('-current_grid_filepath', type=click.Path(), help='Path to save grid search results (default: /data/2024-05-01_2024-05-14/GRID/outputs/)')
def main(input_base_path, input_sub_path, current_grid_filepath):
    logger = logging.getLogger(__name__)
    logger.info(
        f"Running Grid Search Lasso Time Series Experiment using satellites in config {input_base_path+INPUT_CONFIG_FILEPATH}\n")

    grid_configs = [input_base_path + GRID_CONFIG_PATH + file for file in os.listdir(input_base_path+GRID_CONFIG_PATH)]
    latest_config_filepath = max(grid_configs, key=os.path.getmtime)

    logger.info(f'Using latest config file: {latest_config_filepath}\n')
    latest_config_dict = json.load(open(latest_config_filepath))

    grid_config = GridSearchConfig.from_dict(latest_config_dict, input_sub_path)

    config = json.load(open(input_base_path+INPUT_CONFIG_FILEPATH))
    satellites = list(config.keys())

    experiment = TimeSeriesLassoGridSearch(satellites, grid_config)
    results_df = experiment.run_grid_search(parallel=True)

    logger.info(f"Grid Search results saved to {current_grid_filepath} at {datetime.datetime.now()}\n")
    results_df.to_parquet(current_grid_filepath, index=False)
    results_df[results_df['notes'].str.contains("success")].to_parquet(current_grid_filepath.replace('all', 'successful'), index=False)

if __name__ == '__main__':
    log_file = f'./logs/run_lasso_experiment-{datetime.datetime.now()}.log'
    listener = setup_logging(log_file)
    try:
        main()
    finally:
        listener.stop()