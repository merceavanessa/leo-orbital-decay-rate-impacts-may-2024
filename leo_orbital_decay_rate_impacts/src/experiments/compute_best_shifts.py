from scipy import stats
import json
import pandas as pd
import os
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import find_peaks
import numpy as np
from sklearn.metrics import median_absolute_error as mae

BASE_PATH = '../../data/2024-05-01_2024-05-14/processed/'
CONFIG_PATH = '../../data/2024-05-01_2024-05-14/configs/POD/config_pretty.json'
SAVE_PATH = './paper_plots2026/shifts/'

config = json.load(open(CONFIG_PATH))
satellites = list(config.keys())

SAT_PATHS = [BASE_PATH + satellite + '.csv' for satellite in satellites]

def compute_best_shift_from_df(df, cor_column, thr_column, target, filter='All', thr=0, method='pearson'):
    shift_times = list(range(0, 60 * 48, 5))

    strong_indices = df[df['activity_level'] == 'Strong'].index
    medium_indices = df[df['activity_level'] == 'Medium'].index
    quiet_indices = df[df['activity_level'] == 'Quiet'].index

    quiet_medium_indices = df[~df.index.isin(strong_indices)].index

    print(
        f'Data counts for cor with filter={filter}: {len(quiet_indices) if filter == "Quiet" else len(medium_indices) if filter == "Medium" else len(strong_indices) if filter == "Strong" else len(df)}')

    best_shifts = {}
    best_shift = None
    best_corr = 0
    cors_for_sat = []
    for shift in shift_times:
        shifted_df = df[[cor_column, thr_column, target]].copy()
        shifted_col_name = f'shifted_{cor_column}_{shift}_min'
        shifted_df[shifted_col_name] = shifted_df[cor_column].shift(((shift * 60) // 30))
        shifted_df[f'shifted_Bz_{shift}_min'] = shifted_df[thr_column].shift(((shift * 60) // 30))

        neg_bz_indices = shifted_df[shifted_df[f'shifted_Bz_{shift}_min'] < thr].index
        neg_bz_od_sat = shifted_df[shifted_df.index.isin(neg_bz_indices)][
            [target, shifted_col_name]].copy()

        if filter == 'Quiet & Medium':
            neg_bz_od_sat = neg_bz_od_sat[neg_bz_od_sat.index.isin(quiet_medium_indices)]
        elif filter == 'Quiet':
            neg_bz_od_sat = neg_bz_od_sat[neg_bz_od_sat.index.isin(quiet_indices)]
        elif filter == 'Medium':
            neg_bz_od_sat = neg_bz_od_sat[neg_bz_od_sat.index.isin(medium_indices)]
        elif filter == 'Strong':
            neg_bz_od_sat = neg_bz_od_sat[neg_bz_od_sat.index.isin(strong_indices)]

        if method == 'spearman':
            corr_value, p_value = stats.spearmanr(neg_bz_od_sat[target],
                                                  neg_bz_od_sat[shifted_col_name])
        elif method == 'pearson':
            if len(neg_bz_od_sat) < 2:
                cors_for_sat.append(np.nan)
                continue
            corr_value, p_value = stats.pearsonr(neg_bz_od_sat[target],
                                                 neg_bz_od_sat[shifted_col_name])
        else:
            print("No valid method specified for correlation. Use 'spearman' or 'pearson'.")

        if p_value < 0.05:
            # this p-value does not reAlly reflect reality since the data points are not independent
            cors_for_sat.append(corr_value)

            if corr_value > best_corr:
                best_corr = corr_value
                best_shift = shift
        else:
            cors_for_sat.append(np.nan)

    assert len(cors_for_sat) == len(shift_times), f"Expected {len(shift_times)} correlation values, got {len(cors_for_sat)}"
    if best_shift:
        best_shifts[cor_column] = (best_shift, best_corr, cors_for_sat)
    else:
        best_shifts[cor_column] = (None, None, cors_for_sat)

    print(f'Best shift for {cor_column} with filter={filter} and thr={thr}: {best_shifts[cor_column][0]} minutes with correlation {best_shifts[cor_column][1]}')
    return best_shifts


def annotate_orbit_disturbance(df, target, **kwargs):
    peaks, _ = find_peaks(df[target], **kwargs)
    q99 = df[target].quantile(0.99)
    q66 = df[target].quantile(0.66)
    df['activity_level'] = 'Quiet'
    for peak_pos in peaks:
        val = df[target].iloc[peak_pos]
        label = 'Strong' if val >= q99 else 'Medium' if val >= q66 else 'Quiet'

        left = peak_pos
        while (left > 0) and (df.iloc[left][target] > df.iloc[left]['lowess_decay_trend']) and (df.iloc[left]['activity_level'] == 'Quiet'):
            left -= 1

        right = peak_pos
        while (right < len(df) - 1) and (df.iloc[right][target] > df.iloc[right]['lowess_decay_trend']):
            right += 1

        df.iloc[left:right + 1, df.columns.get_loc('activity_level')] = label

    df.loc[(df[target] > df['lowess_decay_trend']) & (df['activity_level']=='Quiet'), 'activity_level'] = np.nan
    df['activity_level'].fillna('Medium', inplace=True)

    return df.copy(), peaks

def annotate(dfs, target):
    peaks = {}
    for satellite in dfs.keys():
        dfs[satellite][target] = dfs[satellite][target].interpolate().fillna(method='bfill').fillna(method='ffill')
        x = np.arange(len(dfs[satellite]))
        y = dfs[satellite][target]
        lowess_smoothed = lowess(y, x, frac=1)
        dfs[satellite]['lowess_decay_trend'] = pd.Series(lowess_smoothed[:, 1], index=dfs[satellite].index)

        mae_lowess = mae(dfs[satellite][target], dfs[satellite]['lowess_decay_trend'])
        dfs[satellite], p = annotate_orbit_disturbance(dfs[satellite], target, prominence=3*mae_lowess)
        peaks[satellite] = p

        print(f"Annotated {satellite} using Median Absolute Error = {mae_lowess:.2f}: ")
        dfs[satellite][[target, 'activity_level', 'lowess_decay_trend']].to_csv(f"./paper_plots2026/shifts/activity_levels_{satellite}.csv")
    return dfs

def read_dataframes():
    dfs = {}
    for i, satellite in enumerate(SAT_PATHS):
        satellite_name = config[satellites[i]]['name']
        df = pd.read_csv(
            BASE_PATH + satellite.split('/')[-1])
        df.index = pd.to_datetime(df['time'])
        df['Bz'] = df['Bz GSE']
        dfs[satellite_name] = df

    return dfs

if __name__ == "__main__":
    target = 'Orbital Decay (m/day)'
    dfs = read_dataframes()
    dfs = annotate(dfs, target)

    cor_cols = ['|avg B|', 'Bz']
    thr_col = 'Bz GSE'
    thrs = [9999, 0, -5]
    filters = ['All', 'Quiet', 'Medium', 'Quiet & Medium', 'Strong']

    best_shifts = {}

    for col in cor_cols:
        best_shifts[col] = {}
        for thr in thrs:
            best_shifts[col][thr] = {}
            for f in filters:
                best_shifts[col][thr][f] = {}
                for sat, od_sat2 in dfs.items():
                    best_shifts[col][thr][f][sat] = {}
                    for mode in ['pearson', 'spearman']:
                        best_shifts[col][thr][f][sat][mode] = compute_best_shift_from_df(od_sat2, col, thr_col, target, f, thr, method=mode)

    with open(SAVE_PATH + f'best_shifts_all_modes.json', 'w') as f:
        json.dump(best_shifts, f, indent=4)