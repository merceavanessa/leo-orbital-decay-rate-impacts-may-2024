import os
import click
import datetime
import json
import logging
import logging.handlers

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from src.configs.grid_search_config import GridSearchConfig
from src.experiments.time_series_lasso_grid_search import TimeSeriesLassoGridSearch
from src.model_selection.fixed_rolling_split import FixedRollingSplit
from src.utils.feature_importance_utils import compute_feature_importances
from src.utils.logging_utils import setup_logging

INPUT_CONFIG_FILEPATH = '/configs/POD/config.json'
np.random.seed(42)

def backup_and_clean_results_folder(results_path, timestamp):
    if os.path.exists(results_path):
        backup_path = results_path + f'_backup_at_{timestamp}'
        os.rename(results_path, backup_path)
    os.makedirs(results_path, exist_ok=True)

@click.command()
@click.option('-input_base_path', type=click.Path(exists=True), help='Path to input base path (e.g. /data/2024-05-01_2024-05-14/)')
@click.option('-input_sub_path', type=click.Path(exists=True), help='Path input sub path for dataset (e.g. /processed_recomputed_decays_with_slopes_noman_fixed_interpolation/)')
@click.option('-current_grid_filepath', type=click.Path(exists=True), help='Path to load a specific grid search results file (e.g. /data/2024-05-01_2024-05-14/GRID/outputs/2024-06-15_16-00_grid_all.csv)')
def main(input_base_path, input_sub_path, current_grid_filepath):
    logger = logging.getLogger(__name__)

    grid_results_df = pd.read_parquet(current_grid_filepath)
    grid_results_df = grid_results_df[grid_results_df['notes'].str.contains("success")]

    grid_config_path = get_grid_config_path(current_grid_filepath)
    config_dict = load_json(grid_config_path)
    grid_config = GridSearchConfig.from_dict(config_dict, input_sub_path)

    satellite_config_dict = load_json(input_base_path + INPUT_CONFIG_FILEPATH)
    satellites = list(satellite_config_dict.keys())

    experiment = TimeSeriesLassoGridSearch(satellites, grid_config)
    data_builders = experiment.get_data_processors_for_satellites()

    scores_and_coef_dfs = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    results_base_path = input_base_path.replace('data/', 'results/')
    backup_and_clean_results_folder(results_base_path, timestamp)

    for i, data_builder in data_builders.items():
        sat = satellites[i]
        satellite_name = satellite_config_dict[sat]['name']
        best_model = select_best_model_for_satellite(grid_results_df, sat)
        logger.info(f'Selecting best model for satellite {satellite_name} ({sat})\n')

        if best_model is None:
            logger.info(f"No model configuration found for satellite {satellite_name}.")
            continue

        results = predict_for_satellite(
            i,
            data_builder,
            best_model,
            grid_config,
            satellite_name=satellite_name
        )

        results_with_val = predict_for_satellite(
            i,
            data_builder,
            best_model,
            grid_config,
            satellite_name=satellite_name,
            with_val=True
        )

        coef_df_train, shap_df_train, combined_train, shap_values_train = compute_feature_importances(
            results['model'], results['data']['train']['X']
        )

        coef_df_test, shap_df_test, combined_test, shap_values_test = compute_feature_importances(
            results['model'], results['data']['test']['X']
        )

        scores_and_coef_dfs[sat] = {
           'no_val': {
                'scores': results['scores'],
                'intercept': results['model'].named_steps['lasso'].intercept_,
                'coefficients': {
                    'train': combined_train,
                    'test': combined_test
                }
            },
            'with_val': {
                'scores': results_with_val['scores']
            }
        }

        results_path = os.path.join(results_base_path, 'satellites', sat, 'data/')
        os.makedirs(results_path, exist_ok=True)

        logger.info(f"Saving results for satellite {satellite_name} ({sat}) at {results_path}\n")
        save_pickle(scores_and_coef_dfs[sat], os.path.join(results_path, f'results_with_importance_{timestamp}.pkl'))

        for mode, shap_vals in [('train', shap_values_train), ('test', shap_values_test)]:
            save_numpy_array(shap_vals, os.path.join(results_path, f'shap_{mode}_{timestamp}.npy'))

        for scenario, results in [('no_val', results), ('with_val', results_with_val)]:
            scenario_path = os.path.join(results_path, scenario)
            os.makedirs(scenario_path, exist_ok=True)

            if scenario == 'no_val':
                save_pickle(results['model'], os.path.join(scenario_path, f'model_{timestamp}.pkl'))
            else:
                save_pickle(results_with_val['model'], os.path.join(scenario_path, f'model_{timestamp}.pkl'))

            logger.info(f"Model saved at { os.path.join(scenario_path, f'model_{timestamp}.pkl')}'\n")

            for mode, data in results['data'].items():
                mode_path = os.path.join(scenario_path, mode)
                os.makedirs(mode_path, exist_ok=True)

                for key, df in data.items():
                    save_dataframe(df, os.path.join(mode_path, f'{key}_{timestamp}.csv'))

    overall_results_path = os.path.join(results_base_path, 'summary/')
    os.makedirs(overall_results_path, exist_ok=True)
    save_pickle(scores_and_coef_dfs, os.path.join(overall_results_path, f'all_results_with_importance_{timestamp}.pkl'))

    # save grid_config for reference
    config_dict['grid_config_path'] = grid_config_path
    save_json(config_dict, os.path.join(overall_results_path, f'reference_grid_config_{timestamp}.json'))

def predict_for_satellite(
    sat_idx,
    dp,
    model_config,
    grid_config,
    satellite_name="",
    with_val=False,   # 🔑 new flag
):
    first_test_icme_index = pd.to_datetime("2024-05-10 16:00:00")

    # Extract hyperparameters and features
    best_alpha = model_config["best_alpha"]
    tol = model_config["tol"]
    max_iter = model_config["max_iter"]
    selected_features = model_config["selected_features"].tolist()

    # Train/test split
    temp_test_size = grid_config.test_size
    if grid_config.train_size + grid_config.test_size + grid_config.offset > len(dp.X):
        temp_test_size = grid_config.offset + len(dp.X) - grid_config.train_size

    splitter = FixedRollingSplit(
        train_size=int(grid_config.train_size),
        step=0,
        fixed_test_to_end=True,
        start=grid_config.offset,
        test_size=temp_test_size,
    )
    train_idx, test_idx = next(splitter.split(dp.X, dp.y))

    # Optionally split train further into train/val
    if with_val:
        half = len(train_idx) // 2
        train_sub_idx, val_sub_idx = train_idx[:half], train_idx[half:]
    else:
        train_sub_idx, val_sub_idx = train_idx, None

    # Post-CME indices
    post_cme_idx = [
        i for i in test_idx
        if (dp.y.index[i] > first_test_icme_index) and
           (i < test_idx[-1] - (grid_config.target_lags_in_minutes[sat_idx] * 2))
    ]

    # Prepare data subsets
    def extract(idx):
        return (
            dp.X.iloc[idx][selected_features],
            dp.y.iloc[idx][grid_config.target]
        ) if idx is not None and len(idx) > 0 else (None, None)

    x_train, y_train = extract(train_sub_idx)
    x_val, y_val = extract(val_sub_idx)
    x_test, y_test = extract(test_idx)
    x_test_post_cme, y_test_post_cme = extract(post_cme_idx)

    # Train model
    model = make_pipeline(
        StandardScaler(),
        Lasso(alpha=best_alpha, max_iter=max_iter, tol=tol, random_state=42)
    )
    model.fit(x_train, y_train)

    # Predictions
    def safe_predict(model, X):
        return model.predict(X) if X is not None and len(X) > 0 else None

    y_pred_train = safe_predict(model, x_train)
    y_pred_val = safe_predict(model, x_val)
    y_pred_test = safe_predict(model, x_test)
    y_pred_test_post_cme = safe_predict(model, x_test_post_cme)

    # Scores
    def compute_scores(y_true, y_pred):
        if y_true is None or y_pred is None:
            return {"MAE": None, "R2": None}
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }

    scores = {
        "train": compute_scores(y_train, y_pred_train),
        "test": compute_scores(y_test, y_pred_test),
        "post_cme_test": compute_scores(y_test_post_cme, y_pred_test_post_cme),
    }
    if with_val:
        scores["val"] = compute_scores(y_val, y_pred_val)

    # Feature importances
    lasso_coef = model.named_steps["lasso"].coef_
    feature_importances = pd.DataFrame({
        "feature": selected_features,
        "coef": lasso_coef
    })

    # Results dict
    results = {
        "satellite": satellite_name,
        "model": model,
        "data": {
            "train": {"X": x_train, "y": y_train, "y_pred": pd.DataFrame({"y_pred": y_pred_train, "time": x_train.index})},
            "test": {"X": x_test, "y": y_test, "y_pred": pd.DataFrame({"y_pred": y_pred_test, "time": x_test.index})},
            "post_cme_test": {"X": x_test_post_cme, "y": y_test_post_cme, "y_pred": pd.DataFrame({"y_pred": y_pred_test_post_cme, "time": x_test_post_cme.index})},
        },
        "scores": scores,
        "feature_importances": feature_importances,
    }
    if with_val:
        results["data"]["val"] = {"X": x_val, "y": y_val, "y_pred": pd.DataFrame({"y_pred": y_pred_val, "time": x_val.index})}

    return results

def get_grid_config_path(grid_filepath):
    datetime_grid = grid_filepath.split('/')[-1].split('_grid_')[0]
    grid_config_name = f'grid_config_{datetime_grid}.json'
    grid_config_path = grid_filepath.replace('/outputs/', '/configs/')
    grid_config_path = '/'.join(grid_config_path.split("/")[:-1]) + '/' + grid_config_name
    return grid_config_path

def select_best_model_for_satellite(grid_results_df, satellite):
    sat_df = grid_results_df[(grid_results_df['satellite'] == satellite) & grid_results_df['notes'].str.contains("success")]
    if sat_df.empty:
        return None
    return sat_df.sort_values(by='mean_cv_MAE').iloc[0]

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_numpy_array(array, path):
    np.save(path, array.values if hasattr(array, 'values') else array)

def save_dataframe(df, path):
    df.to_csv(path, index=True)

def save_pickle(obj, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    log_file = f'./logs/select-best-and-predict-{datetime.datetime.now()}.log'
    listener = setup_logging(log_file)
    try:
        main()
    finally:
        listener.stop()