import logging

import pandas as pd
import warnings
import numpy as np

from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LassoCV, Lasso
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from src.configs.data_config import DataConfig
from src.configs.grid_search_config import GridSearchConfig
from src.configs.lag_config import LagConfig
from src.model_selection.fixed_rolling_split import FixedRollingSplit
from src.model_selection.whitelisted_kbest import WhitelistedSelectKBest
from src.pipeline.dataset_builder import DatasetBuilder

class TimeSeriesLassoGridSearch:
    def __init__(self, sats, gc : GridSearchConfig):
        self.satellites = sats
        self.grid_config = gc
        self.results_df = pd.DataFrame()
        self.logger = logging.getLogger(__name__)

    def run_grid_search(self, parallel=False):
        dps = self.get_data_processors_for_satellites()

        self.logger.info(f"Data processors created for {len(dps)} satellites.")
        self.logger.info(f"Starting grid search with {len(self.satellites)} satellites, {len(self.grid_config.tols)} tols, {len(self.grid_config.ks)} ks, {len(self.grid_config.n_splits)} n_splits, {len(self.grid_config.alphas)} alphas.")

        if parallel:
            total_iters = (
                    len(self.satellites)
                    * len(self.grid_config.tols)
                    * len(self.grid_config.ks)
                    * len(self.grid_config.n_splits)
                    * len(self.grid_config.alphas)
            )

            with tqdm_joblib(tqdm(total=total_iters, desc="Grid search progress")):
                all_results = Parallel(n_jobs=-1)(
                    delayed(self.run_single_grid_search)(sat_idx, dps[sat_idx], k, n_split, alpha_range, alpha_string, tol)
                    for sat_idx in range(len(self.satellites))
                    for tol in self.grid_config.tols
                    for k in self.grid_config.ks
                    for n_split in self.grid_config.n_splits
                    for (alpha_string, alpha_range) in self.grid_config.alphas.items()
                )
        else:
            all_results = []
            for sat_idx in range(len(self.satellites)):
                for tol in self.grid_config.tols:
                    for k in self.grid_config.ks:
                        for n_split in self.grid_config.n_splits:
                            for (alpha_string, alpha_range) in self.grid_config.alphas.items():
                                result = self.run_single_grid_search(sat_idx, dps[sat_idx], k, n_split, alpha_range, alpha_string, tol)
                                all_results.append(result)

        self.results_df = pd.DataFrame(all_results)
        self.results_df = self.results_df[self.results_df['satellite'].notna()]

        self.logger.info(f"Grid search completed with {len(self.results_df)} valid results out of {len(all_results)} total runs.")
        # if len(self.results_df) > 0:
        #     self.logger.info (f"Results DataFrame shape: {self.results_df.iloc[0]}")

        del dps, all_results

        self.results_df['select_k_features'] = self.results_df['select_k_features'].fillna(-1).astype(int)
        self.results_df = self.results_df.sort_values(['satellite', 'mean_cv_MAE'])
        return self.results_df

    def get_data_processors_for_satellites(self):
        dps = {}
        for sat_idx, sat in enumerate(self.satellites):
            dp = DatasetBuilder(data_config=DataConfig(
                data_path=self.grid_config.dataset_path + "/" + sat + '.csv',
                target_column=self.grid_config.target,
                columns_to_keep=[self.grid_config.target] + self.grid_config.cols_train,
                lag_config=LagConfig(
                    use_default=True,
                    default_lag_in_minutes=self.grid_config.target_lags_in_minutes[sat_idx],
                    target_column=self.grid_config.target
                )
            ))

            dp.load_data()
            self.logger.info(f"Loaded data for satellite {sat} with {len(dp.X)} samples and {len(dp.X.columns)} features.")
            dp.preprocess_data(train_size=self.grid_config.train_size,
                               detrend=self.grid_config.detrend,
                               combine_f10_f30=False,
                               use_lagged_inputs=self.grid_config.use_lagged_inputs,
                               input_lags_in_minutes=self.grid_config.input_lags_in_minutes,
                               use_time_feature=self.grid_config.use_time_feature,
                               inputs_blacklisted_from_lagging=self.grid_config.inputs_blacklisted_from_lagging)
            dps[sat_idx] = dp
            self.logger.info(f"Preprocessed data for satellite {sat} with {len(dp.X)} samples and {len(dp.X.columns)} features after preprocessing.")

        return dps

    def run_single_grid_search(self, sat_idx, dp, k, n_splits, alpha_range, alpha_string, tol=1e-4):
        whitelist_idx = [i for i, col in enumerate(dp.X.columns) if col in self.grid_config.whitelisted_features]

        temp_test_size = self.grid_config.test_size
        if self.grid_config.train_size + self.grid_config.test_size + self.grid_config.offset > len(dp.X):
            self.logger.info(f"Train size + test size + offset must be less than the number of samples ({len(dp.X)}). train : {self.grid_config.train_size}, test : {self.grid_config.test_size}, offset : {self.grid_config.offset})")
            temp_test_size = self.grid_config.offset + len(dp.X) - self.grid_config.train_size
            self.logger.info(f"Replacing test size with the remaining samples of {len(dp.X)}: {temp_test_size}")

        splitter = FixedRollingSplit(train_size=self.grid_config.train_size,
                                     step=0, fixed_test_to_end=True,
                                     start=self.grid_config.offset, test_size=temp_test_size)
        train_idx, test_idx = next(splitter.split(dp.X, dp.y))


        self.logger.info(f"CV: \nTrain from {dp.X.iloc[train_idx[0]].name} to {dp.X.iloc[train_idx[-1]].name}. \nTest from {dp.X.iloc[test_idx[0]].name} to {dp.X.iloc[test_idx[-1]].name}.")

        self.logger.info(f"Running grid search for satellite {self.satellites[sat_idx]} with k={k}, n_splits={n_splits}, alpha_range={alpha_string}, tol={tol}")

        x_train, y_train = dp.X.iloc[train_idx], dp.y.iloc[train_idx][self.grid_config.target]
        x_test, y_test = dp.X.iloc[test_idx], dp.y.iloc[test_idx][self.grid_config.target]

        tscv = TimeSeriesSplit(n_splits=n_splits)

        converged = True
        selected_features = []
        model = make_pipeline(
            StandardScaler(),
            WhitelistedSelectKBest(
                k=k,
                whitelist_idx=whitelist_idx,
                score_func=f_regression,
            ),
            LassoCV(
                cv=tscv,
                random_state=42,
                tol=tol,
                alphas=alpha_range,
                max_iter=self.grid_config.max_iter,
                n_jobs=-1
            )
        )
        model_name = 'lassocv'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x_train, y_train)
            if any(issubclass(warning.category, ConvergenceWarning) for warning in w):
                converged = False

        if (not converged) or (model.named_steps[model_name].n_iter_ >= self.grid_config.max_iter):
            return {
                'satellite': self.satellites[sat_idx],
                'n_splits': n_splits,
                'k': k,
                'alpha_range': alpha_string,
                'tol': tol,
                'best_alpha': None,
                'n_iterations': None,
                'mean_cv_MAE': None,
                'mean_cv_r2': None,
                'std_cv_r2': None,
                'test_r2': None,
                'select_k_features': None,
                'post_lasso_features': None,
                'use_lagged_inputs': self.grid_config.use_lagged_inputs,
                'train_days': self.grid_config.train_size / 2880,
                'test_days': self.grid_config.test_size / 2880,
                'offset': self.grid_config.offset,
                'whitelisted_features': self.grid_config.whitelisted_features,
                'selected_features': selected_features,
                'detrend': self.grid_config.detrend,
                'use_time_feature': self.grid_config.use_time_feature,
                'input_lags': self.grid_config.input_lags_in_minutes,
                'max_iter': self.grid_config.max_iter,
                'notes': 'Model did not converge within max_iter. Skipping.'
            }

        best_alpha = model.named_steps[model_name].alpha_

        if (alpha_range is not None) and ((best_alpha == max(alpha_range)) or (best_alpha == min(alpha_range))):
            return {
                'satellite': self.satellites[sat_idx],
                'n_splits': n_splits,
                'k': k,
                'alpha_range': alpha_string,
                'tol': tol,
                'best_alpha': best_alpha,
                'n_iterations': model.named_steps[model_name].n_iter_,
                'mean_cv_MAE': None,
                'mean_cv_r2': None,
                'std_cv_r2': None,
                'test_r2': None,
                'select_k_features': None,
                'post_lasso_features': None,
                'use_lagged_inputs': self.grid_config.use_lagged_inputs,
                'train_days': self.grid_config.train_size / 2880,
                'test_days': self.grid_config.test_size / 2880,
                'offset': self.grid_config.offset,
                'whitelisted_features': self.grid_config.whitelisted_features,
                'selected_features': selected_features,
                'detrend': self.grid_config.detrend,
                'use_time_feature': self.grid_config.use_time_feature,
                'input_lags': self.grid_config.input_lags_in_minutes,
                'max_iter': self.grid_config.max_iter,
                'notes': 'Too low/high regularization needed. Skipping due to bounds reached for alpha.'
            }

        selected_features = list(dp.X.columns[model.named_steps['whitelistedselectkbest'].get_support(indices=True)])

        model = make_pipeline(
            StandardScaler(),
            WhitelistedSelectKBest(
                k=k,
                whitelist_idx=whitelist_idx,
                score_func=f_regression
            ),
            Lasso(alpha=best_alpha,
                  max_iter=self.grid_config.max_iter,
                  random_state=42,
                  tol=tol)
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x_train, y_train)
            if any(issubclass(warning.category, ConvergenceWarning) for warning in w):
                converged = False

        scores_mae = cross_val_score(model, x_train, y_train, cv=tscv, verbose=True, scoring='neg_mean_absolute_error')
        scores_r2 = cross_val_score(model, x_train, y_train, cv=tscv, verbose=True, scoring='r2')

        mean_cv_r2_best_alpha = scores_r2.mean()
        std_cv_r2_best_alpha = scores_r2.std()
        mae_cv_best_alpha = - scores_mae.mean()

        self.logger.info(f"CV MAE (best alpha) cross_val_scores: {mae_cv_best_alpha}")
        self.logger.info(f"Mean CV R² (best alpha) cross_val_scores: {mean_cv_r2_best_alpha}")
        self.logger.info(f"Std CV R² (best alpha) cross_val_scores: {std_cv_r2_best_alpha}")

        model_name = 'lasso'
        if model.named_steps[model_name].n_iter_ < 10 or not converged:
            return {
                'satellite': self.satellites[sat_idx],
                'n_splits': n_splits,
                'k': k,
                'alpha_range': alpha_string,
                'tol': tol,
                'best_alpha': best_alpha,
                'n_iterations': model.named_steps[model_name].n_iter_,
                'mean_cv_MAE': mae_cv_best_alpha,
                'mean_cv_r2': mean_cv_r2_best_alpha,
                'std_cv_r2': std_cv_r2_best_alpha,
                'test_r2': None,
                'select_k_features': None,
                'post_lasso_features': None,
                'use_lagged_inputs': self.grid_config.use_lagged_inputs,
                'train_days': self.grid_config.train_size / 2880,
                'test_days': self.grid_config.test_size / 2880,
                'offset': self.grid_config.offset,
                'whitelisted_features':  self.grid_config.whitelisted_features,
                'selected_features': selected_features,
                'detrend': self.grid_config.detrend,
                'use_time_feature': self.grid_config.use_time_feature,
                'input_lags': self.grid_config.input_lags_in_minutes,
                'max_iter': self.grid_config.max_iter,
                'notes': 'Model converged with n_iter < 10 or did not converge on second round. Skipping due to low iterations or lack of convergence.'
            }

        selected_features = list(dp.X.columns)
        if model.named_steps.get('whitelistedselectkbest'):
            selected_features = list(dp.X.columns[model.named_steps['whitelistedselectkbest'].get_support(indices=True)])
            self.logger.info(f"Selected {len(selected_features)} features for k={k} out of {len(dp.X.columns)}")

        coef_df = pd.DataFrame(model.named_steps[model_name].coef_, index=selected_features, columns=['Coefficient'])
        coef_df['normalized_coef'] = coef_df['Coefficient'].abs() / coef_df['Coefficient'].abs().sum()
        coef_df = coef_df[coef_df['normalized_coef'] > 0].copy()

        y_pred_train_all = model.predict(x_train)
        y_pred_test_final = model.predict(x_test)
        r2_train, r2 = r2_score(y_train, y_pred_train_all), r2_score(y_test, y_pred_test_final)

        self.logger.info(f"Train R2: {r2_train}, Test R2: {r2}")
        self.logger.info(f"Best alpha: {best_alpha:.5f}")
        self.logger.info(f"Iterations: {model.named_steps[model_name].n_iter_}")
        self.logger.info("--------------")
        return {
            'satellite': self.satellites[sat_idx],
            'n_splits': n_splits,
            'k': k,
            'alpha_range': alpha_string,
            'tol': tol,
            'best_alpha': best_alpha,
            'n_iterations': model.named_steps[model_name].n_iter_,
            'mean_cv_MAE': mae_cv_best_alpha,
            'mean_cv_r2': mean_cv_r2_best_alpha,
            'std_cv_r2': std_cv_r2_best_alpha,
            'test_r2': r2,
            'select_k_features': len(selected_features),
            'post_lasso_features': len(coef_df),
            'use_lagged_inputs': self.grid_config.use_lagged_inputs,
            'train_days': self.grid_config.train_size / 2880,
            'test_days': self.grid_config.test_size / 2880,
            'offset': self.grid_config.offset,
            'whitelisted_features': self.grid_config.whitelisted_features,
            'selected_features': selected_features,
            'detrend': self.grid_config.detrend,
            'use_time_feature': self.grid_config.use_time_feature,
            'input_lags': self.grid_config.input_lags_in_minutes,
            'max_iter': self.grid_config.max_iter,
            'notes': 'Model converged successfully.'
        }

def numpy_converter(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
