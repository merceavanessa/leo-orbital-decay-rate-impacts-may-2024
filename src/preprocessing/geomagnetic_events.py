import numpy as np
import pandas as pd

class GeomagneticEventAnnotator:
    """
    Class for annotating geomagnetic storm events in time series data.
    """
    
    def __init__(self):
        """
        Initialize the GeomagneticEventAnnotator with default storm levels.
        """
        self.geomagnetic_storm_levels = {
            "G0 (Quiet to Unsettled)": (0.0, 4.99999),
            "G1 (Minor)": (5.0, 5.99999),
            "G2 (Moderate)": (6.0, 6.99999),
            "G3 (Strong)": (7.0, 7.99999),
            "G4 (Severe)": (8.0, 8.999999),
            "G5 (Extreme)": (9.0, 9.99999)
        }
    
    def set_activity_level(self, df, kp_column='Kp (LASP)'):
        """
        Set activity level based on Kp index and propagate the strongest activity levels.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the Kp index column
        kp_column : str, default='Kp (LASP)'
            Name of the column containing Kp index values
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added 'activity_level' column
        """
        df = df.copy()
        df['activity_level'] = df[kp_column].apply(
            lambda kp: next((level for level, (low, high) in self.geomagnetic_storm_levels.items() 
                            if low <= kp <= high), 'Unknown')
        )
        
        # Propagate backward the strongest activity levels until the transition to quiet level
        activity = df['activity_level'].values
        for level in ["G5 (Extreme)", "G4 (Severe)", "G3 (Strong)", "G2 (Moderate)", "G1 (Minor)"]:
            is_not_quiet = activity != 'G0 (Quiet to Unsettled)'
            is_level = activity == level
            regions = np.flatnonzero(np.diff(np.concatenate(([0], is_not_quiet.view(np.int8), [0]))))
            for start, end in zip(regions[::2], regions[1::2]):
                if is_level[start:end].any():
                    activity[start:end] = level
        df['activity_level'] = activity
        
        # Handle short gaps between events of the same level
        dt_seconds = (df.index[1] - df.index[0]).total_seconds()
        N = 4 * 60 * 3  # 3 hours worth of samples
        activity = df['activity_level'].values
        regions = np.flatnonzero(np.diff(np.concatenate(([0], activity != activity[0], [0]))))
        for start, end in zip(regions[::2], regions[1::2]):
            if end - start < N:
                left_level = activity[start - 1] if start > 0 else None
                right_level = activity[end] if end < len(activity) else None
                if left_level == right_level and left_level is not None and left_level != 'G0 (Quiet to Unsettled)':
                    activity[start:end] = left_level
        df['activity_level'] = activity
        
        return df
    
    def annotate_geomagnetic_storm_events(self, df):
        """
        Annotate geomagnetic storm events with event numbers and unique IDs.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with 'activity_level' column already set
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added event annotations
        """
        df = df.copy()
        
        for i, level in enumerate(self.geomagnetic_storm_levels.keys()):
            if 'G0' in level:
                continue
                
            activity = df['activity_level'].values
            is_level = activity == level
            regions = np.flatnonzero(np.diff(np.concatenate(([0], is_level.view(np.int8), [0]))))
            event_count = len(regions) // 2
            data_samples_per_region = [regions[i + 1] - regions[i] for i in range(0, len(regions), 2)]
            
            df.loc[df['activity_level'] == level, 'event_number'] = np.repeat(
                np.arange(1, event_count + 1), data_samples_per_region
            )
            df.loc[df['activity_level'] == level, 'unique_event_id'] = [
                int(f'{i}{n}') for n in np.repeat(np.arange(1, event_count + 1), data_samples_per_region)
            ]
            
        return df
    
    def annotate_event_windows(self, df, window_hours=24):
        """
        Annotate time windows before, during, and after geomagnetic events.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with geomagnetic events already annotated
        window_hours : int, default=24
            Number of hours before and after events to include in windows
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added event window annotations
        """
        df = df.copy()
        
        df['after_event_24h'] = False
        df['before_event_24h'] = False
        df['during_event'] = False
        df['event_window_plus_minus_24h'] = False
        
        for unique_event in df['unique_event_id'].dropna().unique():
            event_times = df.loc[df['unique_event_id'] == unique_event].index
            
            if len(event_times) == 0:
                continue
                
            event_start = event_times[0]
            event_end = event_times[-1]
            
            time_24h_before = event_start - pd.Timedelta(hours=window_hours)
            time_24_hours_after = event_end + pd.Timedelta(hours=window_hours)
            
            df.loc[(df.index >= time_24h_before) & (df.index < event_start), 'before_event_24h'] = True
            df.loc[(df.index >= event_end) & (df.index <= time_24_hours_after), 'after_event_24h'] = True
            df.loc[(df.index >= event_start) & (df.index <= event_end), 'during_event'] = True
            df.loc[(df.index >= time_24h_before) & (df.index <= time_24_hours_after), 'event_window_plus_minus_24h'] = True
            
        return df
    
    def process_dataframe(self, df, kp_column='Kp (LASP)'):
        """
        Complete processing pipeline for geomagnetic event annotation.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame with Kp index data
        kp_column : str, default='Kp (LASP)'
            Name of the column containing Kp index values
            
        Returns:
        --------
        pandas.DataFrame
            Fully processed DataFrame with all event annotations
        """
        df = self.set_activity_level(df, kp_column)
        df = self.annotate_geomagnetic_storm_events(df)
        df = self.annotate_event_windows(df)
        
        # Add decay level percentile rank
        if 'orbital_decay' in df.columns:
            df['decay_level'] = df['orbital_decay'].rank(pct=True)
            
        return df