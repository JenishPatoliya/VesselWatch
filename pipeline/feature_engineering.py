"""
VesselWatch V3 — Feature Engineering
32 features per vessel including interaction features and Z-scores.
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

from .config import AIS_GAP_THRESHOLD_HRS, RAW_PATH


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _speed_features(data):
    """Features 1-5: Speed statistics per vessel."""
    return data.groupby('MMSI')['SOG'].agg(
        speed_mean='mean', speed_std='std',
        speed_min='min', speed_max='max',
        speed_variance=lambda x: x.var()
    ).reset_index().fillna(0)


def _loitering_features(data):
    """Features 6-8: Loitering score (low distance + high time)."""
    def calc_loitering(group):
        if len(group) < 2:
            return pd.Series({'total_distance_km': 0,
                              'total_time_hrs': 0, 'loitering_score': 0})
        group = group.sort_values('BaseDateTime')
        lats = group['LAT'].values
        lons = group['LON'].values
        dist = sum(haversine(lats[i], lons[i], lats[i + 1], lons[i + 1])
                   for i in range(len(lats) - 1))
        hrs = (group['BaseDateTime'].max() -
               group['BaseDateTime'].min()).total_seconds() / 3600
        score = 1 / (1 + (dist / (hrs + 0.001))) if hrs > 0 else 0
        return pd.Series({'total_distance_km': round(dist, 3),
                          'total_time_hrs': round(hrs, 3),
                          'loitering_score': round(score, 4)})

    return data.groupby('MMSI').apply(
        calc_loitering, include_groups=False).reset_index()


def _gap_features(data):
    """Features 9-14: AIS gap detection with 6-hour threshold."""
    def detect_gaps(group):
        if len(group) < 2:
            return pd.Series({
                'max_gap_hrs': 0, 'total_gaps': 0,
                'gap_flag': 0, 'position_jump_km': 0,
                'avg_gap_hrs': 0, 'gap_duration_std': 0
            })
        group = group.sort_values('BaseDateTime')
        diffs = group['BaseDateTime'].diff().dt.total_seconds() / 3600
        gaps = diffs[diffs > AIS_GAP_THRESHOLD_HRS]
        jump = 0
        if len(gaps) > 0:
            idx = diffs.idxmax()
            iloc = group.index.get_loc(idx)
            if iloc > 0:
                jump = haversine(
                    group['LAT'].iloc[iloc - 1],
                    group['LON'].iloc[iloc - 1],
                    group['LAT'].iloc[iloc],
                    group['LON'].iloc[iloc])
        return pd.Series({
            'max_gap_hrs': round(diffs.max(), 3),
            'total_gaps': len(gaps),
            'gap_flag': 1 if len(gaps) > 0 else 0,
            'position_jump_km': round(jump, 3),
            'avg_gap_hrs': round(diffs.mean(), 3) if len(diffs) > 0 else 0,
            'gap_duration_std': round(diffs.std(), 3) if len(diffs) > 1 else 0,
        })

    return data.groupby('MMSI').apply(
        detect_gaps, include_groups=False).reset_index()


def _port_distance_features(data):
    """Feature 15: Distance from nearest port."""
    ports = pd.read_csv(
        RAW_PATH + 'world_ports.csv'
    )[['Main Port Name', 'Latitude', 'Longitude']].dropna()

    last_pos = data.groupby('MMSI').agg(
        last_lat=('LAT', 'last'),
        last_lon=('LON', 'last')).reset_index()

    def nearest_port(lat, lon):
        d = np.sqrt((ports['Latitude'] - lat) ** 2 +
                    (ports['Longitude'] - lon) ** 2)
        return round(d.min() * 111, 2)

    last_pos['dist_from_port_km'] = last_pos.apply(
        lambda r: nearest_port(r['last_lat'], r['last_lon']),
        axis=1)
    return last_pos


def _behavioral_features(data):
    """Features 16-17: Behavioral fingerprinting against own baseline."""
    daily = data.groupby(['MMSI', data['BaseDateTime'].dt.date]
                         ).agg(daily_speed=('SOG', 'mean')).reset_index()
    baseline = daily.groupby('MMSI').agg(
        baseline_speed=('daily_speed', 'mean'),
        speed_consistency=('daily_speed', 'std')
    ).reset_index().fillna(0)
    baseline['behavioral_score'] = (
        baseline['speed_consistency'] /
        (baseline['baseline_speed'] + 0.001)).round(4)
    return baseline


def _direction_features(data):
    """Features 18-19: Course-over-ground direction changes."""
    def calc_direction_changes(group):
        if len(group) < 3:
            return pd.Series({'cog_change_mean': 0, 'cog_change_max': 0})
        group = group.sort_values('BaseDateTime')
        cog = group['COG'].values
        changes = np.abs(np.diff(cog))
        changes = np.minimum(changes, 360 - changes)
        return pd.Series({
            'cog_change_mean': round(np.mean(changes), 3),
            'cog_change_max': round(np.max(changes), 3)
        })

    return data.groupby('MMSI').apply(
        calc_direction_changes, include_groups=False).reset_index()


def _add_interaction_features(feat):
    """Features 20-26: Interaction features capturing compound patterns."""
    feat['gap_x_jump'] = feat['max_gap_hrs'] * feat['position_jump_km']
    feat['gap_x_port_dist'] = feat['max_gap_hrs'] * feat['dist_from_port_km']
    feat['speed_range'] = feat['speed_max'] - feat['speed_min']
    feat['distance_per_hour'] = (
        feat['total_distance_km'] / (feat['total_time_hrs'] + 0.001))
    feat['gap_ratio'] = (
        feat['total_gaps'] / (feat['total_time_hrs'] + 0.001))
    feat['ping_density'] = (
        feat['ping_count'] / (feat['total_time_hrs'] + 0.001))
    feat['jump_per_gap'] = (
        feat['position_jump_km'] / (feat['total_gaps'] + 1))
    return feat


def _add_vessel_type_zscores(feat):
    """Features 27-31: Z-scores relative to vessel type baseline."""
    z_features = ['max_gap_hrs', 'position_jump_km',
                  'dist_from_port_km', 'speed_mean', 'loitering_score']

    for f_name in z_features:
        z_col = f'{f_name}_zscore'
        type_mean = feat['VesselTypeLabel'].map(
            feat.groupby('VesselTypeLabel')[f_name].mean())
        type_std = feat['VesselTypeLabel'].map(
            feat.groupby('VesselTypeLabel')[f_name].std())
        feat[z_col] = (feat[f_name] - type_mean) / (type_std + 0.01)

    return feat


def compute_features(data):
    """Main function: compute all 32 features per vessel."""
    print('  Computing speed statistics...')
    speed_f = _speed_features(data)

    print('  Computing loitering scores...')
    loiter_f = _loitering_features(data)

    print('  Detecting AIS gaps...')
    gap_f = _gap_features(data)

    print('  Computing port distances...')
    last_pos = _port_distance_features(data)

    print('  Computing behavioral fingerprints...')
    baseline = _behavioral_features(data)

    print('  Computing direction changes...')
    cog_f = _direction_features(data)

    print('  Computing ping density...')
    ping_count = data.groupby('MMSI').agg(
        ping_count=('BaseDateTime', 'count')).reset_index()

    # ── Combine all features ──────────────────────────────
    feat = speed_f.copy()
    feat = feat.merge(loiter_f[['MMSI', 'total_distance_km',
                                'total_time_hrs', 'loitering_score']],
                      on='MMSI', how='left')
    feat = feat.merge(gap_f[['MMSI', 'max_gap_hrs', 'total_gaps',
                             'gap_flag', 'position_jump_km',
                             'avg_gap_hrs', 'gap_duration_std']],
                      on='MMSI', how='left')
    feat = feat.merge(last_pos[['MMSI', 'last_lat', 'last_lon',
                                'dist_from_port_km']],
                      on='MMSI', how='left')
    feat = feat.merge(baseline[['MMSI', 'behavioral_score',
                                'speed_consistency']],
                      on='MMSI', how='left')
    feat = feat.merge(cog_f, on='MMSI', how='left')
    feat = feat.merge(ping_count, on='MMSI', how='left')
    feat = feat.merge(
        data.groupby('MMSI')['VesselTypeLabel'].first().reset_index(),
        on='MMSI', how='left')
    feat = feat.merge(
        data.groupby('MMSI')['VesselName'].first().reset_index(),
        on='MMSI', how='left')

    feat = feat.fillna(0)

    print('  Adding interaction features...')
    feat = _add_interaction_features(feat)

    print('  Adding vessel-type Z-scores...')
    feat = _add_vessel_type_zscores(feat)

    return feat
