"""
utils/data_loader.py
--------------------
Handles all dataset loading and version creation for the HIGGS benchmark.
Supports SAMPLE_MODE for RAM-safe development on 16GB machines.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE MODE — CRITICAL RAM SAFETY FLAG
# True  → loads 500k rows  → ~1-2GB RAM → fast development
# False → loads full 11M   → ~8GB RAM   → overnight paper run only
# NEVER change the default to False in source code.
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_MODE = True
SAMPLE_ROWS = 500_000

HIGGS_PATH = os.path.join(os.path.dirname(__file__), '..', 'HIGGS.csv')


def load_raw(filepath=HIGGS_PATH, sample_mode=SAMPLE_MODE):
    """
    Load HIGGS.csv. Returns clean DataFrame with named columns.

    Parameters
    ----------
    filepath    : str  — path to HIGGS.csv
    sample_mode : bool — True loads 500k rows, False loads all 11M

    Returns
    -------
    pd.DataFrame — columns: label, feature_1 ... feature_28
    """
    nrows    = SAMPLE_ROWS if sample_mode else None
    mode_str = f"SAMPLE MODE ({SAMPLE_ROWS:,} rows)" if sample_mode else "FULL DATA (11,000,000 rows)"

    print(f"[data_loader] Loading HIGGS.csv — {mode_str}")
    if not sample_mode:
        print("[data_loader] ⚠️  WARNING: Full load needs ~8GB RAM. Close ALL other applications first.")

    col_names = ['label'] + [f'feature_{i}' for i in range(1, 29)]
    df = pd.read_csv(filepath, header=None, names=col_names, nrows=nrows)
    df['label'] = df['label'].astype(int)

    sig = df['label'].sum()
    bg  = (df['label'] == 0).sum()
    print(f"[data_loader] Loaded: {len(df):,} rows | Signal={sig:,} | Background={bg:,} | Ratio=1:{bg/sig:.1f}")
    return df


def create_versions(df):
    """
    Create 3 imbalance versions.

    Version A — original (~1:1 balanced)
    Version B — 1:10 (keep all background, downsample signal to 1/10th background)
    Version C — 1:50 (keep all background, downsample signal to 1/50th background)

    All downsampling uses random_state=42 for reproducibility.

    Returns
    -------
    dict — keys 'A', 'B', 'C', values are DataFrames
    """
    background = df[df['label'] == 0]
    signal     = df[df['label'] == 1]
    n_bg       = len(background)

    version_A = df.copy()

    n_sig_B  = n_bg // 10
    signal_B = signal.sample(n=n_sig_B, random_state=42)
    version_B = pd.concat([background, signal_B]).sample(frac=1, random_state=42).reset_index(drop=True)

    n_sig_C  = n_bg // 50
    signal_C = signal.sample(n=n_sig_C, random_state=42)
    version_C = pd.concat([background, signal_C]).sample(frac=1, random_state=42).reset_index(drop=True)

    versions = {'A': version_A, 'B': version_B, 'C': version_C}

    for name, v in versions.items():
        sig   = v['label'].sum()
        bg    = (v['label'] == 0).sum()
        ratio = bg / sig if sig > 0 else float('inf')
        print(f"[data_loader] Version {name}: {len(v):,} rows | Signal={sig:,} | Background={bg:,} | Ratio=1:{ratio:.1f}")

    return versions


def split_version(df, version_name):
    """
    Stratified 80/20 train/test split. Preserves class ratio in both sets.

    Parameters
    ----------
    df           : pd.DataFrame
    version_name : str — label for print output

    Returns
    -------
    X_train, X_test, y_train, y_test — numpy arrays
    """
    X = df.drop('label', axis=1).values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[data_loader] Version {version_name} split → Train: {len(X_train):,} | Test: {len(X_test):,} | "
          f"Train signal ratio: {y_train.mean():.3f}")
    return X_train, X_test, y_train, y_test


def get_all_splits(sample_mode=SAMPLE_MODE):
    """
    Full pipeline: load → create 3 versions → split all.

    Returns
    -------
    dict — keys 'A','B','C', each: {'X_train','X_test','y_train','y_test','version'}
    """
    df       = load_raw(sample_mode=sample_mode)
    versions = create_versions(df)
    splits   = {}
    for name, v in versions.items():
        X_train, X_test, y_train, y_test = split_version(v, name)
        splits[name] = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'version': name
        }
    return splits


if __name__ == '__main__':
    splits = get_all_splits()
    print("\n[data_loader] All splits complete.")
