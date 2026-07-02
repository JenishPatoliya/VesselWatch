"""
VesselWatch V3 — Pseudo-Label Generation
Multi-signal composite labeling instead of simple OR logic.
"""

import numpy as np


def create_improved_labels(features_df):
    """Create smarter pseudo-labels using composite scoring and
    vessel-type-relative thresholds.

    A vessel is labeled anomalous if:
      - Strong single signal (gap>48h, jump>500km, port>500km)
      - 2+ moderate signals combined
      - Type-relative Z-score outlier (>2.5 std devs)
      - Compound suspicious pattern (gap + jump + far from port)
    """

    # Strategy 1: Absolute thresholds (strong signals)
    strong_gap = features_df['max_gap_hrs'] > 48
    strong_jump = features_df['position_jump_km'] > 500
    extreme_port = features_df['dist_from_port_km'] > 500

    # Strategy 2: Combined moderate signals (need 2+ signals)
    moderate_gap = features_df['max_gap_hrs'] > 24
    moderate_jump = features_df['position_jump_km'] > 100
    moderate_port = features_df['dist_from_port_km'] > 200
    moderate_speed = (features_df['speed_variance'] >
                      features_df['speed_variance'].quantile(0.90))
    moderate_loiter = (features_df['loitering_score'] >
                       features_df['loitering_score'].quantile(0.90))

    moderate_signals = (moderate_gap.astype(int) +
                        moderate_jump.astype(int) +
                        moderate_port.astype(int) +
                        moderate_speed.astype(int) +
                        moderate_loiter.astype(int))

    # Strategy 3: Vessel-type-relative Z-scores (2+ std devs)
    zscore_anomaly = (
        (features_df['max_gap_hrs_zscore'] > 2.5) |
        (features_df['position_jump_km_zscore'] > 2.5)
    )

    # Strategy 4: Compound suspicious patterns
    gap_with_jump = (
        (features_df['max_gap_hrs'] > 12) &
        (features_df['position_jump_km'] > 50) &
        (features_df['dist_from_port_km'] > 100)
    )

    # Final label: strong single OR 2+ moderate OR
    #              type-relative outlier OR compound pattern
    true_anomaly = (
        strong_gap | strong_jump | extreme_port |
        (moderate_signals >= 2) |
        zscore_anomaly |
        gap_with_jump
    ).astype(int)

    return true_anomaly
