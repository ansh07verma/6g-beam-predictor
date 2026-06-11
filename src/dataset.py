"""
dataset.py — DeepMIMO Data Pipeline for 6G Beam Prediction
===========================================================
Handles dataset loading, DFT codebook generation, beam label computation,
and PyTorch DataLoader construction.

DeepMIMO API
------------
Uses DeepMIMO v4 (deepmimo>=4.0.0b11), compatible with Python 3.10+.
v4 uses a flat attribute-based API:
    dataset.channel     → (N_users, N_r, N_t, N_subcarriers)
    dataset.rx_pos      → (N_users, 3)

Supported Scenarios
-------------------
  o1_28          — Outdoor 1, 28 GHz mmWave (PRIMARY — matches PRD spec)
                   ~18,000 UE grid, clean rectangular topology
                   Expected Top-1 accuracy: 88–92%

  asu_campus_3p5 — ASU Campus outdoor, 3.5 GHz (FALLBACK — already downloaded)
                   131,931 UEs, complex campus multipath
                   Expected Top-1 accuracy: ~73%

To use o1_28:
  1. Download via notebooks/DeepMIMO_O1_28_Downloader.ipynb in Google Colab
  2. Place extracted folder at: deepmimo_scenarios/o1_28/
  3. Run: python src/train.py --scenario o1_28 --device cpu

Channel configuration:
    BS antennas : 64-element ULA  (shape=[64, 1])
    UE antennas : 1               (shape=[1, 1])
    Subcarriers : 64 (OFDM), subcarrier 0 extracted for baseline

Usage
-----
    from src.dataset import dft_codebook, load_deepmimo_v4, make_dataloaders

    # ── Real data ───────────────────────────────────────────────────────────
    positions, channels = load_deepmimo_v4('o1_28')          # 28 GHz mmWave
    # OR
    positions, channels = load_deepmimo_v4('asu_campus_3p5') # 3.5 GHz fallback
    channels, positions = filter_valid_users(channels, positions)
    codebook            = dft_codebook(N_t=64, N_b=64)
    labels              = compute_beam_labels(channels, codebook)
    train_dl, val_dl, test_dl, scaler, ch_split = make_dataloaders(
                                positions, labels, channels)

    # ── Synthetic (no download) ─────────────────────────────────────────────
    positions, channels = generate_synthetic_dataset(n_users=18000, N_t=64)
"""

import os
import sys
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DFT Codebook  (single source of truth — also imported by evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

def dft_codebook(N_t: int = 64, N_b: int = 64) -> np.ndarray:
    """Generate a DFT beamforming codebook for a Uniform Linear Array (ULA).

    Each beam vector (row) is a ULA steering vector:
        f_i[n] = (1/sqrt(N_t)) * exp(j*pi*n*sin(theta_i))
    where theta_i = arcsin(2*i/N_b - 1) scans from -90° to +90°.

    Parameters
    ----------
    N_t : int  — number of transmit antennas (ULA elements)
    N_b : int  — number of beams (codebook size)

    Returns
    -------
    codebook : np.ndarray, shape (N_b, N_t), dtype complex128
        Row i is beam vector f_i.  All rows are unit-norm.

    Example
    -------
    >>> cb = dft_codebook(64, 64)
    >>> cb.shape
    (64, 64)
    >>> np.allclose(np.linalg.norm(cb, axis=1), 1.0)
    True
    """
    beam_indices = np.arange(N_b)
    sin_vals    = 2.0 * beam_indices / N_b - 1.0   # uniform scan in [-1, +1]
    ant_indices = np.arange(N_t)

    phases   = np.pi * np.outer(sin_vals, ant_indices)         # (N_b, N_t)
    codebook = (1.0 / np.sqrt(N_t)) * np.exp(1j * phases)     # (N_b, N_t)
    return codebook.astype(np.complex128)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DeepMIMO v4 Loader
# ─────────────────────────────────────────────────────────────────────────────

# ── Per-scenario default configs ─────────────────────────────────────────
# These presets ensure correct antenna arrays and OFDM parameters for each
# published DeepMIMO scenario.
# n_subcarriers=1: DeepMIMO allocates (N, N_r, N_t, n_subcarriers) internally.
# Setting this to 1 instead of 64 gives a 64× RAM reduction at load time with
# zero accuracy impact — we only use subcarrier 0 for narrowband beam prediction.
_SCENARIO_DEFAULTS = {
    'o1_28': {
        'N_t': 64, 'N_b_ant_shape': [64, 1],
        'n_subcarriers': 1, 'subcarrier_idx': 0, 'num_paths': 10,
        'description': 'Outdoor-1, 28 GHz mmWave (PRD spec)',
    },
    'o1_28b': {
        'N_t': 64, 'N_b_ant_shape': [64, 1],
        'n_subcarriers': 1, 'subcarrier_idx': 0, 'num_paths': 10,
        'description': 'Outdoor-1B, 28 GHz mmWave (extended)',
    },
    'asu_campus_3p5': {
        'N_t': 64, 'N_b_ant_shape': [64, 1],
        'n_subcarriers': 1, 'subcarrier_idx': 0, 'num_paths': 10,
        'description': 'ASU Campus, 3.5 GHz outdoor (fallback)',
    },
}


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
    """Load a DeepMIMO scenario using the v4 API and return numpy arrays.

    Supports both 'o1_28' (28 GHz, PRD spec) and 'asu_campus_3p5' (3.5 GHz
    fallback). Auto-downloads the scenario if not found locally; for
    rate-limited accounts, place the manually downloaded folder at
    deepmimo_scenarios/<scenario_name>/ and it will be found automatically.

    Parameters
    ----------
    scenario           : str  — DeepMIMO scenario name ('o1_28' or 'asu_campus_3p5')
    N_t                : int  — number of BS transmit antennas (ULA)
    N_b_ant_shape      : list — [rows, cols] antenna shape; default [N_t, 1]
    n_subcarriers      : int  — OFDM subcarrier count
    subcarrier_idx     : int  — which subcarrier to extract (0 = baseline)
    num_paths          : int  — max multipath components
    local_scenario_dir : str  — override path to scenarios directory
                                (e.g. 'deepmimo_scenarios/')

    Returns
    -------
    positions : np.ndarray, shape (N_users, 3)        — UE (x,y,z) in metres
    channels  : np.ndarray, shape (N_users, N_r, N_t) — complex channel vectors

    Raises
    ------
    ImportError   — if deepmimo package is not installed
    RuntimeError  — if scenario is not found and download fails
    """
    try:
        import deepmimo as dm
    except ImportError as e:
        raise ImportError(
            "deepmimo is not installed. Run: pip install 'deepmimo>=4.0.0b11' --pre"
        ) from e

    # Apply scenario-specific defaults
    sc_key = scenario.lower()
    if sc_key in _SCENARIO_DEFAULTS:
        defaults = _SCENARIO_DEFAULTS[sc_key]
        if N_b_ant_shape is None:
            N_b_ant_shape = defaults['N_b_ant_shape']
        print(f"[DeepMIMO] Scenario: {defaults['description']}")
    else:
        if N_b_ant_shape is None:
            N_b_ant_shape = [N_t, 1]

    # ── Resolve scenario directory ────────────────────────────────────────
    # Priority: 1) local_scenario_dir arg  2) deepmimo_scenarios/ in project root
    #           3) default DeepMIMO cache   4) auto-download
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_dir    = local_scenario_dir or os.path.join(project_root, 'deepmimo_scenarios')
    local_path   = os.path.join(local_dir, scenario.lower())

    # If found in project deepmimo_scenarios/, point DeepMIMO there
    if not scenario.replace('_', '').replace('-', '').isalnum():
        raise ValueError(f"Invalid scenario name: '{scenario}'. Only alphanumeric, underscore, and hyphen allowed.")
    
    scenario_folder = dm.get_scenario_folder(scenario)
    if os.path.exists(local_path) and not os.path.exists(scenario_folder):
        import shutil
        scenarios_cache = os.path.dirname(scenario_folder)
        os.makedirs(scenarios_cache, exist_ok=True)
        target = os.path.join(scenarios_cache, scenario.lower())
        if not os.path.exists(target):
            shutil.copytree(local_path, target)
            print(f"[DeepMIMO] Copied '{scenario}' from project folder to cache.")
        scenario_folder = target

    # Auto-download if still not found
    if not os.path.exists(scenario_folder):
        print(f"[DeepMIMO] '{scenario}' not found locally. Attempting download...")
        result = dm.download(scenario)
        if result is None and not os.path.exists(dm.get_scenario_folder(scenario)):
            raise RuntimeError(
                f"Failed to download '{scenario}'.\n"
                "Your account may have hit the download rate limit.\n"
                "Solution: Open notebooks/DeepMIMO_O1_28_Downloader.ipynb in\n"
                "Google Colab, run it, download the zip to your PC, and place\n"
                f"the extracted folder at: deepmimo_scenarios/{scenario.lower()}/"
            )

    print(f"[DeepMIMO v4] Loading '{scenario}' ...")
    dataset = dm.load(scenario)

    # ── Configure antenna array and channel parameters ────────────────────
    params = dm.ChannelParameters()
    params.bs_antenna['shape']   = np.array(N_b_ant_shape)
    params.ue_antenna['shape']   = np.array([1, 1])
    params.num_paths             = num_paths
    params.freq_domain           = 1
    params.ofdm['subcarriers']   = n_subcarriers
    params.ofdm['selected_subcarriers'] = np.array([subcarrier_idx])

    print(f"[DeepMIMO v4] Computing channels (BS: {N_b_ant_shape} ULA, "
          f"subcarrier {subcarrier_idx}/{n_subcarriers}) ...")
          
    # ── Memory Safety for Massive Scenarios (like O1_28) ──────────────────
    if len(dataset.datasets) > 1:
        print(f"[DeepMIMO v4] WARNING: Scenario contains {len(dataset.datasets)} TX-RX pairs.")
        print(f"              Keeping only the first pair to prevent out-of-memory errors.")
        dataset.datasets = [dataset.datasets[0]]
        
    dataset.compute_channels(params)

    # v4 API: flat attribute access
    channels_full = np.array(dataset.channel)    # (N, N_r, N_t, n_subcarriers)
    positions     = np.array(dataset.rx_pos)     # (N, 3)

    # Squeeze subcarrier dim → (N, N_r, N_t).
    # Use ascontiguousarray so the result owns its memory, then immediately
    # delete channels_full — without this both the 4D original and the 3D
    # slice live in RAM simultaneously, doubling peak usage.
    channels = np.ascontiguousarray(channels_full[..., 0])
    del channels_full, dataset
    import gc; gc.collect()

    print(f"[DeepMIMO v4] Loaded {channels.shape[0]:,} users | "
          f"channel shape per user: {channels.shape[1:]} | "
          f"RAM: {channels.nbytes / 1e6:.1f} MB")

    # Memory Safety: Cap the maximum number of users if requested
    if max_users is not None and len(channels) > max_users:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(channels), size=max_users, replace=False)
        idx.sort()
        channels  = np.ascontiguousarray(channels[idx])
        positions = positions[idx]
        gc.collect()
        print(f"[DeepMIMO v4] Subsampled to {max_users:,} users | "
              f"RAM now: {channels.nbytes / 1e6:.1f} MB")

    return positions, channels


# ─────────────────────────────────────────────────────────────────────────────
# 3. Synthetic Fallback (Rayleigh — for pipeline testing only)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_dataset(
    n_users: int = 18000,
    N_t: int = 64,
    N_r: int = 1,
    scene_size_m: float = 500.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Rayleigh-faded channels for pipeline testing.

    WARNING: i.i.d. Rayleigh channels have no spatial structure.
    A model trained on this data cannot exceed ~1/N_b accuracy.
    Always use real DeepMIMO data for the actual project deliverables.

    Parameters
    ----------
    n_users      : int   — number of simulated UEs
    N_t          : int   — number of transmit antennas
    N_r          : int   — number of receive antennas (1 for single-antenna UE)
    scene_size_m : float — scenario spatial extent in metres
    seed         : int   — random seed

    Returns
    -------
    positions : np.ndarray, shape (n_users, 3)
    channels  : np.ndarray, shape (n_users, N_r, N_t), dtype complex128
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, scene_size_m, n_users)
    y = rng.uniform(0, scene_size_m, n_users)
    z = np.zeros(n_users)
    positions = np.stack([x, y, z], axis=1)

    real_part = rng.standard_normal((n_users, N_r, N_t)) / np.sqrt(2)
    imag_part = rng.standard_normal((n_users, N_r, N_t)) / np.sqrt(2)
    channels  = (real_part + 1j * imag_part).astype(np.complex128)

    print(f"[Synthetic] Generated {n_users:,} users | "
          f"channel shape per user: ({N_r}, {N_t})")
    return positions, channels


# ─────────────────────────────────────────────────────────────────────────────
# 4. Filtering
# ─────────────────────────────────────────────────────────────────────────────

def filter_valid_users(
    channels: np.ndarray,
    positions: np.ndarray,
    norm_threshold: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove users with zero-power channel vectors (blocked/no-path users).

    A user is blocked if the Frobenius norm of their channel matrix is
    below norm_threshold. These correspond to los=-1 in the DeepMIMO dataset.

    Parameters
    ----------
    channels       : np.ndarray, shape (N_users, N_r, N_t)
    positions      : np.ndarray, shape (N_users, 3)
    norm_threshold : float — minimum channel norm to retain user

    Returns
    -------
    channels_filt  : np.ndarray — filtered channels
    positions_filt : np.ndarray — filtered positions
    """
    norms = np.linalg.norm(
        channels.reshape(channels.shape[0], -1), axis=1
    )
    valid_mask = norms > norm_threshold
    n_removed  = int((~valid_mask).sum())

    if n_removed > 0:
        print(f"[Filter] Removed {n_removed:,} blocked users "
              f"(norm < {norm_threshold:.1e}). "
              f"Remaining: {valid_mask.sum():,}")

    return channels[valid_mask], positions[valid_mask]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Beam Label Generation
# ─────────────────────────────────────────────────────────────────────────────

def compute_beam_labels(
    channels: np.ndarray,
    codebook: np.ndarray,
) -> np.ndarray:
    """Compute the optimal beam index per user via exhaustive search.

    For each user i:
        b*_i = argmax_j  |h_i^H @ f_j|^2

    Parameters
    ----------
    channels : np.ndarray, shape (N_users, N_r, N_t) or (N_users, N_t)
        Complex channel matrices. N_r=1 is squeezed automatically.
    codebook : np.ndarray, shape (N_b, N_t)
        DFT codebook — rows are unit-norm beam vectors.

    Returns
    -------
    labels : np.ndarray, shape (N_users,), dtype int64
        Optimal beam index in [0, N_b-1].
    """
    if channels.ndim == 3 and channels.shape[1] == 1:
        h = channels[:, 0, :]           # (N, N_t) — squeeze single RX ant
    elif channels.ndim == 3:
        N_r = channels.shape[1]
        h = channels.reshape(channels.shape[0], -1)
        codebook = np.tile(codebook, (1, N_r))
    else:
        h = channels

    # Beamforming power: |H @ F^H|^2  →  (N_users, N_b)
    bf_gain = np.abs(h @ codebook.conj().T) ** 2
    labels  = np.argmax(bf_gain, axis=1).astype(np.int64)

    print(f"[Labels] Computed beam labels | "
          f"unique beams: {len(np.unique(labels))} / {codebook.shape[0]}")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 6. PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BeamDataset(Dataset):
    """PyTorch Dataset: normalized UE positions → beam class labels.

    Parameters
    ----------
    positions_norm : np.ndarray, shape (N, 3) — StandardScaler-normalized
    labels         : np.ndarray, shape (N,)   — integer beam indices
    """

    def __init__(self, positions_norm: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(positions_norm, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 7. DataLoader Factory
# ─────────────────────────────────────────────────────────────────────────────

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
    """Build train / val / test DataLoaders from positions and beam labels.

    Applies StandardScaler (fit on train only) and stratified splitting
    so every beam class is proportionally represented in each split.

    Parameters
    ----------
    positions  : np.ndarray, shape (N, 3)
    labels     : np.ndarray, shape (N,)   — integer beam indices [0, N_b-1]
    channels   : np.ndarray or None       — kept for achievable-rate evaluation
    batch_size : int   — default 256
    val_ratio  : float — fraction for validation (default 0.10)
    test_ratio : float — fraction for test      (default 0.20)
    seed       : int   — reproducibility seed
    num_workers: int   — DataLoader worker processes (0 = main process)

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    scaler   : StandardScaler — fitted on train positions
    ch_split : dict           — {'train':.., 'val':.., 'test':..} channel arrays
    """
    N = len(labels)

    # ── Stratified split ──────────────────────────────────────────────────
    idx_trainval, idx_test = train_test_split(
        np.arange(N), test_size=test_ratio,
        stratify=labels, random_state=seed,
    )
    val_of_trainval = val_ratio / (1.0 - test_ratio)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_of_trainval,
        stratify=labels[idx_trainval], random_state=seed,
    )

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    azimuth   = np.arctan2(y, x)                          # always safe: [-pi, pi]
    # Clip to [-1,1] BEFORE arcsin to prevent NaN when r is near 0
    elev_arg  = np.clip(z / (r + 1e-6), -1.0, 1.0)
    elevation = np.arcsin(elev_arg)                       # safe: [-pi/2, pi/2]

    # Feature engineering: euclidean (x,y,z) + spherical (r, az, el)
    features = np.stack([x, y, z, r, azimuth, elevation], axis=1).astype(np.float64)

    # Sanity check — any NaN/Inf here would cause silent GPU crashes
    n_bad = np.sum(~np.isfinite(features))
    if n_bad > 0:
        print(f"[DataLoader] WARNING: {n_bad} NaN/Inf values in features — replacing with 0")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Scale features
    scaler = StandardScaler()
    feat_scaled = features.copy().astype(np.float32)
    feat_scaled[idx_train] = scaler.fit_transform(features[idx_train])
    feat_scaled[idx_val]   = scaler.transform(features[idx_val])
    feat_scaled[idx_test]  = scaler.transform(features[idx_test])

    # ── Datasets & Loaders ────────────────────────────────────────────────
    train_ds = BeamDataset(feat_scaled[idx_train], labels[idx_train])
    val_ds   = BeamDataset(feat_scaled[idx_val],   labels[idx_val])
    test_ds  = BeamDataset(feat_scaled[idx_test],  labels[idx_test])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    ch_split: dict = {}
    if channels is not None:
        ch_split = {
            'train': channels[idx_train],
            'val':   channels[idx_val],
            'test':  channels[idx_test],
        }

    print(f"[DataLoader] Split: train={len(train_ds):,} | "
          f"val={len(val_ds):,} | test={len(test_ds):,}")
    print(f"             Scaler: mean~{scaler.mean_.mean():.3f}, "
          f"std~{scaler.scale_.mean():.3f}")

    return train_loader, val_loader, test_loader, scaler, ch_split


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("  dataset.py — smoke test")
    print("=" * 60)

    use_real = '--real' in sys.argv

    if use_real:
        positions, channels = load_deepmimo_v4('asu_campus_3p5')
    else:
        positions, channels = generate_synthetic_dataset(n_users=5000)

    channels, positions = filter_valid_users(channels, positions)

    codebook = dft_codebook(N_t=64, N_b=64)
    print(f"Codebook shape  : {codebook.shape}")
    print(f"Unit-norm check : {np.allclose(np.linalg.norm(codebook, axis=1), 1.0)}")

    labels = compute_beam_labels(channels, codebook)
    print(f"Label range     : [{labels.min()}, {labels.max()}]")

    train_dl, val_dl, test_dl, scaler, _ = make_dataloaders(
        positions, labels, channels, batch_size=64)

    X_b, y_b = next(iter(train_dl))
    print(f"Batch shapes — X: {X_b.shape}, y: {y_b.shape}")
    print("dataset.py smoke test PASSED ✓")
