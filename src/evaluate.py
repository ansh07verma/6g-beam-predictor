"""
evaluate.py — Evaluation Metrics for 6G Beam Prediction
=========================================================
Computes the three metrics specified in FR-E1 through FR-E4:
  - Top-1 accuracy on the test set
  - Top-5 accuracy on the test set
  - Achievable rate ratio:  R_predicted / R_optimal  at SNR = 20 dB
    (also reports R_random for comparison)

Additionally generates two evaluation plots (FR-E5, FR-E6):
  - Training curve: loss & Top-1 accuracy vs epoch  → results/training_curve.png
  - Spatial beam map: UE positions colored by beam   → results/beam_map.png
  - Achievable rate CDF: predicted / optimal / random → results/rate_cdf.png

Usage
-----
    # From repository root (loads checkpoints/best_mlp.pt automatically):
    python src/evaluate.py
    python src/evaluate.py --real --checkpoint checkpoints/best_mlp.pt

    # Programmatic:
    from src.evaluate import evaluate_all, plot_training_curve
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')           # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)



# ─────────────────────────────────────────────────────────────────────────────
# 1. Top-k Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def topk_accuracy(
    model:  nn.Module,
    loader: DataLoader,
    k:      int,
    device: torch.device,
) -> float:
    """Compute Top-k accuracy of a model on a DataLoader.

    A prediction is "correct" if the ground-truth label appears in the
    model's top-k predicted classes.

    Parameters
    ----------
    model  : nn.Module    — trained BeamPredictor (in eval mode after call)
    loader : DataLoader   — test or validation DataLoader
    k      : int          — number of top predictions to consider (e.g. 1 or 5)
    device : torch.device

    Returns
    -------
    accuracy : float — fraction of samples correctly classified in top-k
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            logits = model(X_batch)                            # (B, N_b)
            # Top-k indices: shape (B, k)
            topk_indices = logits.topk(k, dim=1).indices
            # Check if ground truth appears anywhere in top-k
            correct = topk_indices.eq(
                y_batch.unsqueeze(1).expand_as(topk_indices)
            ).any(dim=1)

            total_correct += correct.sum().item()
            total_samples += len(y_batch)

    return total_correct / max(total_samples, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Achievable Rate
# ─────────────────────────────────────────────────────────────────────────────

def achievable_rate_batch(
    channels:     np.ndarray,
    beam_indices: np.ndarray,
    codebook:     np.ndarray,
    snr_db:       float = 20.0,
) -> np.ndarray:
    """Compute per-user achievable rate R = log2(1 + SNR * |h^H f|^2).

    Parameters
    ----------
    channels     : np.ndarray, shape (N, N_r, N_t) or (N, N_t)
        Complex channel matrices.  N_r=1 case is squeezed automatically.
    beam_indices : np.ndarray, shape (N,)
        Integer beam indices in [0, N_b-1] selecting the applied beam.
    codebook     : np.ndarray, shape (N_b, N_t)
        DFT codebook from dataset.dft_codebook() — rows are beam vectors.
    snr_db       : float
        Signal-to-noise ratio in decibels (default 20 dB).

    Returns
    -------
    rates : np.ndarray, shape (N,) — achievable rate in bps/Hz per user
    """
    snr_linear = 10.0 ** (snr_db / 10.0)

    # Squeeze single RX antenna
    if channels.ndim == 3 and channels.shape[1] == 1:
        h = channels[:, 0, :]           # (N, N_t)
    elif channels.ndim == 3:
        h = channels.reshape(channels.shape[0], -1)
    else:
        h = channels                    # already (N, N_t)

    # Gather selected beam vectors for each user: (N, N_t)
    f_selected = codebook[beam_indices, :]    # (N, N_t)

    # Normalize channels so average power is 1. This ensures snr_db represents
    # the average received SNR across the dataset, canceling out DeepMIMO pathloss
    # which is otherwise ~100-140 dB and drives Shannon capacity to 0.0
    h_power = np.mean(np.abs(h)**2)
    h_norm  = h / (np.sqrt(h_power) + 1e-12)

    # Beamforming gain |h^H f|^2 per user
    bf_gain = np.abs(np.sum(np.conj(h_norm) * f_selected, axis=1)) ** 2  # (N,)

    # Shannon capacity formula
    rates = np.log2(1.0 + snr_linear * bf_gain)
    return rates


def evaluate_rates(
    model:       nn.Module,
    test_loader: DataLoader,
    channels_test: np.ndarray,
    codebook:    np.ndarray,
    snr_db:      float = 20.0,
    device:      torch.device = torch.device('cpu'),
) -> dict:
    """Compute achievable rates for predicted, optimal, and random beams.

    Parameters
    ----------
    model          : trained BeamPredictor
    test_loader    : DataLoader for the test split
    channels_test  : np.ndarray, shape (N_test, N_r, N_t)
    codebook       : np.ndarray, shape (N_b, N_t)
    snr_db         : float — evaluation SNR in dB (default 20)
    device         : torch.device

    Returns
    -------
    results : dict with keys:
        'pred_beams'   : np.ndarray — predicted beam index per user
        'optimal_beams': np.ndarray — optimal beam index per user
        'rates_pred'   : np.ndarray — achievable rate with predicted beam
        'rates_opt'    : np.ndarray — achievable rate with optimal beam
        'rates_rand'   : np.ndarray — achievable rate with random beam
        'mean_pred'    : float
        'mean_opt'     : float
        'mean_rand'    : float
        'rate_ratio'   : float — mean(rates_pred) / mean(rates_opt)
    """
    model.eval()
    pred_beams    = []
    optimal_beams = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch.to(device))
            preds  = logits.argmax(dim=1).cpu().numpy()
            pred_beams.append(preds)
            optimal_beams.append(y_batch.numpy())

    pred_beams    = np.concatenate(pred_beams)
    optimal_beams = np.concatenate(optimal_beams)

    N = len(pred_beams)
    rng = np.random.default_rng(42)
    rand_beams = rng.integers(0, codebook.shape[0], size=N)

    rates_pred = achievable_rate_batch(channels_test, pred_beams,    codebook, snr_db)
    rates_opt  = achievable_rate_batch(channels_test, optimal_beams, codebook, snr_db)
    rates_rand = achievable_rate_batch(channels_test, rand_beams,    codebook, snr_db)

    mean_pred  = float(rates_pred.mean())
    mean_opt   = float(rates_opt.mean())
    mean_rand  = float(rates_rand.mean())
    rate_ratio = mean_pred / max(mean_opt, 1e-10)

    return {
        'pred_beams':    pred_beams,
        'optimal_beams': optimal_beams,
        'rand_beams':    rand_beams,
        'rates_pred':    rates_pred,
        'rates_opt':     rates_opt,
        'rates_rand':    rates_rand,
        'mean_pred':     mean_pred,
        'mean_opt':      mean_opt,
        'mean_rand':     mean_rand,
        'rate_ratio':    rate_ratio,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Full Evaluation Driver
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(
    model:          nn.Module,
    test_loader:    DataLoader,
    channels_test:  np.ndarray,
    codebook:       np.ndarray,
    snr_db:         float = 20.0,
    device:         torch.device = torch.device('cpu'),
) -> dict:
    """Compute and print all evaluation metrics (FR-E1 through FR-E4).

    Parameters
    ----------
    model          : trained BeamPredictor
    test_loader    : DataLoader for the test split
    channels_test  : np.ndarray, shape (N_test, N_r, N_t)
    codebook       : np.ndarray, shape (N_b, N_t)
    snr_db         : float — evaluation SNR in dB
    device         : torch.device

    Returns
    -------
    metrics : dict — all computed metrics
    """
    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)

    # ── Top-k accuracy ────────────────────────────────────────────────────
    top1 = topk_accuracy(model, test_loader, k=1, device=device)
    top5 = topk_accuracy(model, test_loader, k=5, device=device)

    print(f"  Top-1 Accuracy : {top1*100:.2f}%  (target: > 85%)")
    print(f"  Top-5 Accuracy : {top5*100:.2f}%  (target: > 97%)")

    # ── Achievable rate ───────────────────────────────────────────────────
    rate_results = evaluate_rates(model, test_loader, channels_test,
                                   codebook, snr_db, device)

    print(f"\n  Achievable Rate @ SNR = {snr_db} dB")
    print(f"  ─────────────────────────────────────────")
    print(f"  Predicted beam : {rate_results['mean_pred']:.4f} bps/Hz")
    print(f"  Optimal beam   : {rate_results['mean_opt']:.4f} bps/Hz")
    print(f"  Random beam    : {rate_results['mean_rand']:.4f} bps/Hz")
    print(f"  Rate Ratio     : {rate_results['rate_ratio']:.4f}  (target: > 0.95)")
    print("=" * 55)

    # ── Pass/fail summary ─────────────────────────────────────────────────
    checks = {
        'top1_pass':  top1 >= 0.85,
        'top5_pass':  top5 >= 0.97,
        'ratio_pass': rate_results['rate_ratio'] >= 0.95,
    }
    print("\n  Acceptance Criteria")
    print(f"  Top-1 ≥ 85% :  {'✓ PASS' if checks['top1_pass']  else '✗ FAIL'}")
    print(f"  Top-5 ≥ 97% :  {'✓ PASS' if checks['top5_pass']  else '✗ FAIL'}")
    print(f"  Ratio ≥ 0.95:  {'✓ PASS' if checks['ratio_pass'] else '✗ FAIL'}")
    print("=" * 55 + "\n")

    metrics = {
        'top1_acc':   top1,
        'top5_acc':   top5,
        **rate_results,
        **checks,
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 4. Visualization Functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curve(
    history:  dict,
    save_dir: str = 'results',
) -> str:
    """Plot training loss and validation Top-1 accuracy vs epoch.

    Creates a dual-axis plot: left y-axis = loss, right y-axis = accuracy.
    Saves to ``save_dir/training_curve.png``.

    Parameters
    ----------
    history  : dict — output of train() with 'train_loss' and 'val_acc' keys
    save_dir : str  — directory to save the PNG

    Returns
    -------
    save_path : str — absolute path to the saved figure
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = np.arange(1, len(history['train_loss']) + 1)
    train_loss = np.array(history['train_loss'])
    val_acc    = np.array(history['val_acc']) * 100.0

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f1117')
    ax1.set_facecolor('#0f1117')

    color_loss = '#7c83fd'
    color_acc  = '#f7b731'

    # Training loss (left axis)
    ln1 = ax1.plot(epochs, train_loss, color=color_loss, linewidth=2,
                   label='Train Loss', marker='o', markersize=3, alpha=0.9)
    ax1.set_xlabel('Epoch', fontsize=13, color='white')
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=13, color=color_loss)
    ax1.tick_params(axis='y', labelcolor=color_loss, colors='white')
    ax1.tick_params(axis='x', colors='white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#444')

    # Val accuracy (right axis)
    ax2 = ax1.twinx()
    ax2.set_facecolor('#0f1117')
    ln2 = ax2.plot(epochs, val_acc, color=color_acc, linewidth=2,
                   label='Val Top-1 Acc', marker='s', markersize=3, alpha=0.9)
    ax2.set_ylabel('Top-1 Accuracy (%)', fontsize=13, color=color_acc)
    ax2.tick_params(axis='y', labelcolor=color_acc, colors='white')

    # 85% target line
    ax2.axhline(85, color='#ff6b6b', linestyle='--', linewidth=1.2,
                label='85% target', alpha=0.7)

    # Best epoch marker
    best_epoch = history.get('best_epoch', 0)
    if best_epoch > 0 and best_epoch <= len(val_acc):
        ax2.axvline(best_epoch, color='#26de81', linestyle=':', linewidth=1.5,
                    label=f'Best epoch ({best_epoch})', alpha=0.8)

    # Combined legend
    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right',
               facecolor='#1a1d2e', edgecolor='#444', labelcolor='white',
               fontsize=10)

    plt.title('Training Curve — 6G Beam Prediction MLP',
              fontsize=14, color='white', pad=12)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'training_curve.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Training curve saved → {save_path}")
    return save_path


def plot_beam_map(
    positions:   np.ndarray,
    beam_labels: np.ndarray,
    title:       str  = 'Spatial Beam Prediction Map',
    save_dir:    str  = 'results',
    filename:    str  = 'beam_map.png',
    N_b:         int  = 64,
) -> str:
    """Scatter plot of UE positions colored by predicted (or optimal) beam index.

    Parameters
    ----------
    positions   : np.ndarray, shape (N, 3) — raw or normalized UE positions
    beam_labels : np.ndarray, shape (N,)   — beam index per user (0..N_b-1)
    title       : str  — plot title
    save_dir    : str  — output directory
    filename    : str  — output filename
    N_b         : int  — number of beams (for colormap range)

    Returns
    -------
    save_path : str
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')

    scatter = ax.scatter(
        positions[:, 0], positions[:, 1],
        c=beam_labels, cmap='hsv',
        vmin=0, vmax=N_b - 1,
        s=6, alpha=0.7, linewidths=0,
    )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Beam Index', fontsize=12, color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.outline.set_edgecolor('#444')

    ax.set_xlabel('X Position (m)', fontsize=13, color='white')
    ax.set_ylabel('Y Position (m)', fontsize=13, color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    ax.set_title(title, fontsize=14, color='white', pad=12)
    plt.tight_layout()

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Beam map saved → {save_path}")
    return save_path


def plot_rate_cdf(
    rates_pred:  np.ndarray,
    rates_opt:   np.ndarray,
    rates_rand:  np.ndarray,
    snr_db:      float = 20.0,
    save_dir:    str   = 'results',
) -> str:
    """Plot CDF of achievable rates for predicted / optimal / random beams.

    Parameters
    ----------
    rates_pred  : per-user rates with predicted beam
    rates_opt   : per-user rates with optimal beam
    rates_rand  : per-user rates with random beam
    snr_db      : float — SNR used for labeling
    save_dir    : str   — output directory

    Returns
    -------
    save_path : str
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')

    def _cdf(data):
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    for rates, label, color in [
        (rates_opt,  'Optimal Beam',   '#26de81'),
        (rates_pred, 'Predicted Beam', '#f7b731'),
        (rates_rand, 'Random Beam',    '#ff6b6b'),
    ]:
        x, y = _cdf(rates)
        ax.plot(x, y * 100, label=label, color=color, linewidth=2)

    ax.set_xlabel('Achievable Rate (bps/Hz)', fontsize=13, color='white')
    ax.set_ylabel('CDF (%)',                   fontsize=13, color='white')
    ax.set_title(f'Achievable Rate CDF @ SNR = {snr_db} dB',
                 fontsize=14, color='white', pad=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.legend(facecolor='#1a1d2e', edgecolor='#444', labelcolor='white',
               fontsize=11)
    ax.grid(True, color='#2a2d3e', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'rate_cdf.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Rate CDF saved → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate BeamPredictor on the 6G beam prediction task.'
    )
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best_mlp.pt',
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--real', action='store_true', default=True,
                        help='Use DeepMIMO asu_campus_3p5 dataset (default)')
    parser.add_argument('--no-real', action='store_false', dest='real',
                        help='Use synthetic data instead')
    parser.add_argument('--scenario', type=str, default='asu_campus_3p5')
    parser.add_argument('--snr-db', type=float, default=20.0,
                        help='Evaluation SNR in dB (default: 20)')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory for saved plots')
    parser.add_argument('--n-synthetic', type=int, default=18000)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    from src.dataset import (
        dft_codebook, generate_synthetic_dataset, load_deepmimo_v4,
        filter_valid_users, compute_beam_labels, make_dataloaders,
    )
    from src.model import BeamPredictor

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load data ──────────────────────────────────────────────────────────
    if args.real:
        positions, channels = load_deepmimo_v4(args.scenario)
    else:
        positions, channels = generate_synthetic_dataset(args.n_synthetic)

    channels, positions = filter_valid_users(channels, positions)
    codebook = dft_codebook(N_t=64, N_b=64)
    labels   = compute_beam_labels(channels, codebook)

    _, _, test_loader, scaler, ch_split = make_dataloaders(
        positions, labels, channels, batch_size=256)

    # ── Load checkpoint ────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found at '{args.checkpoint}'.")
        print("Run `python src/train.py` first to generate a checkpoint.")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg  = ckpt.get('config', {})
    
    input_dim = cfg.get('input_dim', 3)
    hidden_dims = cfg.get('hidden_dims', [512, 256, 128])
    num_classes = cfg.get('num_classes', 64)
    dropout = cfg.get('dropout', 0.2)
    n_res_blocks = cfg.get('n_res_blocks', 2)
    
    if not isinstance(input_dim, int) or not (1 <= input_dim <= 1000):
        raise ValueError(f"Invalid input_dim: {input_dim}")
    if not isinstance(hidden_dims, list) or not all(isinstance(d, int) and 1 <= d <= 10000 for d in hidden_dims):
        raise ValueError(f"Invalid hidden_dims: {hidden_dims}")
    if not isinstance(num_classes, int) or not (1 <= num_classes <= 10000):
        raise ValueError(f"Invalid num_classes: {num_classes}")
    if not isinstance(dropout, (int, float)) or not (0 <= dropout <= 1):
        raise ValueError(f"Invalid dropout: {dropout}")
    if not isinstance(n_res_blocks, int) or not (0 <= n_res_blocks <= 100):
        raise ValueError(f"Invalid n_res_blocks: {n_res_blocks}")
    
    model = BeamPredictor(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout,
        n_res_blocks=n_res_blocks,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[Evaluate] Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, "
          f"val_acc = {ckpt.get('val_acc', 0)*100:.2f}%")

    # ── Evaluate ───────────────────────────────────────────────────────────
    channels_test = ch_split.get('test', channels[:100])   # fallback
    metrics = evaluate_all(model, test_loader, channels_test,
                           codebook, snr_db=args.snr_db, device=device)

    # ── Plots ──────────────────────────────────────────────────────────────
    os.makedirs(args.results_dir, exist_ok=True)

    # Training curve (needs history file)
    history_path = 'checkpoints/training_history.json'
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curve(history, save_dir=args.results_dir)

    # Beam map (use original positions for spatial interpretation)
    # Re-create test positions without normalization for plotting
    plot_beam_map(
        positions[-len(metrics['pred_beams']):],    # approximate test positions
        metrics['pred_beams'],
        title='Predicted Beam Map — Test Set',
        save_dir=args.results_dir,
        N_b=codebook.shape[0],
    )

    # Rate CDF
    plot_rate_cdf(
        metrics['rates_pred'],
        metrics['rates_opt'],
        metrics['rates_rand'],
        snr_db=args.snr_db,
        save_dir=args.results_dir,
    )
