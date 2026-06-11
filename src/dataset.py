"""
dataset.py — DeepMIMO Data Pipeline
=====================================
Handles dataset loading, DFT codebook generation, beam label computation,
and PyTorch DataLoader construction.
"""

import os
import sys
import gc
import shutil
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


_SCENARIO_DEFAULTS = {
    'o1_28': {
        'N_t': 64, 'N_b_ant_shape': [64, 1],
        'n_subcarriers': 1, 'subcarrier_idx': 0, 'num_paths': 10,
        'description': 'Outdoor-1, 28 GHz mmWave',
    },
    'o1_28b': {
        'N_t': 64, 'N_b_ant_shape': [64, 1],
        'n_subcarriers': 1, 'subcarrier_idx': 0, 'num_paths': 10,
        'description': 'Outdoor-1B, 28 GHz mmWave',
    },
    'asu_campus_3p5': {
        'N_t': 64, 'N_b_ant_shape': [64, 1],
        'n_subcarriers': 1, 'subcarrier_idx': 0, 'num_paths': 10,
        'description': 'ASU Campus, 3.5 GHz outdoor',
    },
}


def dft_codebook(N_t: int = 64, N_b: int = 64) -> np.ndarray:
    """Generate a DFT beamforming codebook for a ULA."""
    beam_indices = np.arange(N_b)
    sin_vals = 2.0 * beam_indices / N_b - 1.0
    ant_indices = np.arange(N_t)
    phases = np.pi * np.outer(sin_vals, ant_indices)
    codebook = (1.0 / np.sqrt(N_t)) * np.exp(1j * phases)
    return codebook.astype(np.complex128)


def load_deepmimo_v4(
    scenario: str = 'o1_28',
    N_t: int = 64,
    N_b_ant_shape: list = None,
    n_subcarriers: int = 1,
    subcarrier_idx: int = 0,
    num_paths: int = 10,
    local_scenario_dir: str = None,
    max_users: int = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a DeepMIMO scenario and return positions and channels."""
    try:
        import deepmimo as dm
    except ImportError as e:
        raise ImportError("deepmimo is not installed. Run: pip install 'deepmimo>=4.0.0b11' --pre") from e

    if not scenario.replace('_', '').replace('-', '').isalnum():
        raise ValueError(f"Invalid scenario name: '{scenario}'")

    sc_key = scenario.lower()
    if sc_key in _SCENARIO_DEFAULTS:
        defaults = _SCENARIO_DEFAULTS[sc_key]
        if N_b_ant_shape is None:
            N_b_ant_shape = defaults['N_b_ant_shape']
        print(f"[DeepMIMO] Scenario: {defaults['description']}")
    else:
        if N_b_ant_shape is None:
            N_b_ant_shape = [N_t, 1]

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_dir = local_scenario_dir or os.path.join(project_root, 'deepmimo_scenarios')
    local_path = os.path.join(local_dir, scenario.lower())

    scenario_folder = dm.get_scenario_folder(scenario)
    if os.path.exists(local_path) and not os.path.exists(scenario_folder):
        scenarios_cache = os.path.dirname(scenario_folder)
        os.makedirs(scenarios_cache, exist_ok=True)
        target = os.path.join(scenarios_cache, scenario.lower())
        if not os.path.exists(target):
            shutil.copytree(local_path, target)
            print(f"[DeepMIMO] Copied '{scenario}' to cache.")
        scenario_folder = target

    if not os.path.exists(scenario_folder):
        print(f"[DeepMIMO] '{scenario}' not found. Attempting download...")
        result = dm.download(scenario)
        if result is None and not os.path.exists(dm.get_scenario_folder(scenario)):
            raise RuntimeError(f"Failed to download '{scenario}'. Check rate limits.")

    print(f"[DeepMIMO] Loading '{scenario}'...")
    dataset = dm.load(scenario)

    params = dm.ChannelParameters()
    params.bs_antenna['shape'] = np.array(N_b_ant_shape)
    params.ue_antenna['shape'] = np.array([1, 1])
    params.num_paths = num_paths
    params.freq_domain = 1
    params.ofdm['subcarriers'] = n_subcarriers
    params.ofdm['selected_subcarriers'] = np.array([subcarrier_idx])

    if len(dataset.datasets) > 1:
        dataset.datasets = [dataset.datasets[0]]

    dataset.compute_channels(params)

    channels_full = np.array(dataset.channel)
    positions = np.array(dataset.rx_pos)

    channels = np.ascontiguousarray(channels_full[..., 0])
    del channels_full, dataset
    gc.collect()

    print(f"[DeepMIMO] Loaded {channels.shape[0]:,} users | RAM: {channels.nbytes / 1e6:.1f} MB")

    if max_users is not None and len(channels) > max_users:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(channels), size=max_users, replace=False)
        idx.sort()
        channels = np.ascontiguousarray(channels[idx])
        positions = positions[idx]
        gc.collect()
        print(f"[DeepMIMO] Subsampled to {max_users:,} users")

    return positions, channels


def generate_synthetic_dataset(
    n_users: int = 18000,
    N_t: int = 64,
    N_r: int = 1,
    scene_size_m: float = 500.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Rayleigh-faded channels for testing."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, scene_size_m, n_users)
    y = rng.uniform(0, scene_size_m, n_users)
    z = np.zeros(n_users)
    positions = np.stack([x, y, z], axis=1)

    real_part = rng.standard_normal((n_users, N_r, N_t)) / np.sqrt(2)
    imag_part = rng.standard_normal((n_users, N_r, N_t)) / np.sqrt(2)
    channels = (real_part + 1j * imag_part).astype(np.complex128)

    print(f"[Synthetic] Generated {n_users:,} users")
    return positions, channels


def filter_valid_users(
    channels: np.ndarray,
    positions: np.ndarray,
    norm_threshold: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove users with zero-power channel vectors."""
    norms = np.linalg.norm(channels.reshape(channels.shape[0], -1), axis=1)
    valid_mask = norms > norm_threshold
    n_removed = int((~valid_mask).sum())

    if n_removed > 0:
        print(f"[Filter] Removed {n_removed:,} blocked users. Remaining: {valid_mask.sum():,}")

    return channels[valid_mask], positions[valid_mask]


def compute_beam_labels(channels: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Compute optimal beam index per user via exhaustive search."""
    if channels.ndim == 3 and channels.shape[1] == 1:
        h = channels[:, 0, :]
    elif channels.ndim == 3:
        h = channels.reshape(channels.shape[0], -1)
        codebook = np.tile(codebook, (1, channels.shape[1]))
    else:
        h = channels

    bf_gain = np.abs(h @ codebook.conj().T) ** 2
    labels = np.argmax(bf_gain, axis=1).astype(np.int64)

    print(f"[Labels] Computed beam labels | unique beams: {len(np.unique(labels))} / {codebook.shape[0]}")
    return labels


class BeamDataset(Dataset):
    """PyTorch Dataset: normalized UE positions → beam class labels."""

    def __init__(self, positions_norm: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(positions_norm, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_dataloaders(
    positions: np.ndarray,
    labels: np.ndarray,
    channels: Optional[np.ndarray] = None,
    batch_size: int = 256,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, dict]:
    """Build train/val/test DataLoaders with stratified splitting."""
    N = len(labels)

    idx_trainval, idx_test = train_test_split(
        np.arange(N), test_size=test_ratio, stratify=labels, random_state=seed,
    )
    val_of_trainval = val_ratio / (1.0 - test_ratio)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_of_trainval, stratify=labels[idx_trainval], random_state=seed,
    )

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(y, x)
    elev_arg = np.clip(z / (r + 1e-6), -1.0, 1.0)
    elevation = np.arcsin(elev_arg)

    features = np.stack([x, y, z, r, azimuth, elevation], axis=1).astype(np.float64)

    n_bad = np.sum(~np.isfinite(features))
    if n_bad > 0:
        print(f"[DataLoader] WARNING: {n_bad} NaN/Inf values — replacing with 0")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    feat_scaled = features.copy().astype(np.float32)
    feat_scaled[idx_train] = scaler.fit_transform(features[idx_train])
    feat_scaled[idx_val] = scaler.transform(features[idx_val])
    feat_scaled[idx_test] = scaler.transform(features[idx_test])

    train_ds = BeamDataset(feat_scaled[idx_train], labels[idx_train])
    val_ds = BeamDataset(feat_scaled[idx_val], labels[idx_val])
    test_ds = BeamDataset(feat_scaled[idx_test], labels[idx_test])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    ch_split = {}
    if channels is not None:
        ch_split = {
            'train': channels[idx_train],
            'val': channels[idx_val],
            'test': channels[idx_test],
        }

    print(f"[DataLoader] Split: train={len(train_ds):,} | val={len(val_ds):,} | test={len(test_ds):,}")

    return train_loader, val_loader, test_loader, scaler, ch_split


if __name__ == '__main__':
    positions, channels = generate_synthetic_dataset(n_users=5000)
    channels, positions = filter_valid_users(channels, positions)
    codebook = dft_codebook(N_t=64, N_b=64)
    labels = compute_beam_labels(channels, codebook)
    train_dl, val_dl, test_dl, scaler, _ = make_dataloaders(positions, labels, channels, batch_size=64)
    X_b, y_b = next(iter(train_dl))
    print(f"Batch shapes — X: {X_b.shape}, y: {y_b.shape}")
    print("dataset.py smoke test PASSED")
