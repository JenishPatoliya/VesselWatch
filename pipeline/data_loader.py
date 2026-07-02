"""
VesselWatch V3 — Data Loading & Cleaning
"""

import pandas as pd
import numpy as np
import gc

from .config import RAW_PATH, FILES, USE_COLS, VESSEL_TYPE_MAP


def load_raw_data():
    """Load all 7 days of AIS data and split into train/test."""
    dfs = []
    for file in FILES:
        print(f'Loading {file}...')
        temp = pd.read_csv(RAW_PATH + file,
                           usecols=USE_COLS, nrows=800000)
        temp['date'] = file.split('_')[3].replace('.csv', '')
        dfs.append(temp)

    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
    df['day'] = df['BaseDateTime'].dt.day
    df['VesselTypeLabel'] = df['VesselType'].map(
        VESSEL_TYPE_MAP).fillna('Other')

    # Train/Test Split — temporal
    train_df = df[df['day'] <= 15].copy()
    test_df = df[df['day'] >= 16].copy()
    common_vessels = set(train_df['MMSI'].unique()) & \
                     set(test_df['MMSI'].unique())

    print(f'\nTotal rows: {len(df):,}')
    print(f'Train: {len(train_df):,} rows | '
          f'{train_df["MMSI"].nunique():,} vessels')
    print(f'Test:  {len(test_df):,} rows  | '
          f'{test_df["MMSI"].nunique():,} vessels')
    print(f'Vessels in both sets: {len(common_vessels):,}')

    return train_df, test_df, common_vessels


def clean_data(data):
    """Remove invalid MMSIs, impossible speeds, duplicates, etc."""
    data = data[(data['MMSI'] >= 200000000) &
                (data['MMSI'] <= 999999999)]
    data = data[(data['SOG'] >= 0) & (data['SOG'] <= 50)]
    data['Heading'] = data['Heading'].replace(511, np.nan)
    data['VesselName'] = data['VesselName'].fillna(
        data['MMSI'].astype(str))
    data['VesselType'] = data['VesselType'].fillna(0)
    data['Length'] = data['Length'].fillna(0)
    data = data.drop_duplicates(
        subset=['MMSI', 'BaseDateTime'])
    data = data.sort_values(
        ['MMSI', 'BaseDateTime']).reset_index(drop=True)
    return data
