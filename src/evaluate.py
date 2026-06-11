"""
evaluate.py — Evaluation Metrics & Visualization
==================================================
Computes Top-1/Top-5 accuracy, achievable rate ratio, and generates plots.
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def topk_accuracy(model: nn.Module, loader: DataLoader, k: int, device: torch.device) -> float:
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            logits = model(X_batch)
            topk_indices = logits.topk(k, dim=1).indices
            correct = topk_indices.eq(y_batch.unsqueeze(1).expand_as(topk_indices)).any(dim=1)

            total_correct += correct.sum().item()
            total_samples += len(y_batch)

    return total_correct / max(total_samples, 1)


def achievable_rate_batch(
    channels: np.ndarray,
    beam_indices: np.ndarray,
    codebook: np.ndarray,
    snr_db: float = 20.0,
) -> np.ndarray:
    snr_linear = 10.0 ** (snr_db / 10.0)

    if channels.ndim == 3 and channels.shape[1] == 1:
        h = channels[:, 0, :]
    elif channels.ndim == 3:
        h = channels.reshape(channels.shape[0], -1)
    else:
        h = channels

    f_selected = codebook[beam_indices, :]
    h_power = np.mean(np.abs(h)**2)
    h_norm = h / (np.sqrt(h_power) + 1e-12)

    bf_gain = np.abs(np.sum(np.conj(h_norm) * f_selected, axis=1)) ** 2
    return np.log2(1.0 + snr_linear * bf_gain)


def evaluate_rates(
    model: nn.Module,
    test_loader: DataLoader,
    channels_test: np.ndarray,
    codebook: np.ndarray,
    snr_db: float = 20.0,
    device: torch.device = torch.device('cpu'),
) -> dict:
    model.eval()
    pred_beams = []
    optimal_beams = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch.to(device))
            pred_beams.append(logits.argmax(dim=1).cpu().numpy())
            optimal_beams.append(y_batch.numpy())

    pred_beams = np.concatenate(pred_beams)
    optimal_beams = np.concatenate(optimal_beams)

    N = len(pred_beams)
    rand_beams = np.random.default_rng(42).integers(0, codebook.shape[0], size=N)

    rates_pred = achievable_rate_batch(channels_test, pred_beams, codebook, snr_db)
    rates_opt = achievable_rate_batch(channels_test, optimal_beams, codebook, snr_db)
    rates_rand = achievable_rate_batch(channels_test, rand_beams, codebook, snr_db)

    mean_pred = float(rates_pred.mean())
    mean_opt = float(rates_opt.mean())
    mean_rand = float(rates_rand.mean())

    return {
        'pred_beams': pred_beams,
        'optimal_beams': optimal_beams,
        'rand_beams': rand_beams,
        'rates_pred': rates_pred,
        'rates_opt': rates_opt,
        'rates_rand': rates_rand,
        'mean_pred': mean_pred,
        'mean_opt': mean_opt,
        'mean_rand': mean_rand,
        'rate_ratio': mean_pred / max(mean_opt, 1e-10),
    }


def evaluate_all(
    model: nn.Module,
    test_loader: DataLoader,
    channels_test: np.ndarray,
    codebook: np.ndarray,
    snr_db: float = 20.0,
    device: torch.device = torch.device('cpu'),
) -> dict:
    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)

    top1 = topk_accuracy(model, test_loader, k=1, device=device)
    top5 = topk_accuracy(model, test_loader, k=5, device=device)

    print(f"  Top-1 Accuracy : {top1*100:.2f}%  (target: > 85%)")
    print(f"  Top-5 Accuracy : {top5*100:.2f}%  (target: > 97%)")

    rate_results = evaluate_rates(model, test_loader, channels_test, codebook, snr_db, device)

    print(f"\n  Achievable Rate @ SNR = {snr_db} dB")
    print(f"  ─────────────────────────────────────────")
    print(f"  Predicted beam : {rate_results['mean_pred']:.4f} bps/Hz")
    print(f"  Optimal beam   : {rate_results['mean_opt']:.4f} bps/Hz")
    print(f"  Random beam    : {rate_results['mean_rand']:.4f} bps/Hz")
    print(f"  Rate Ratio     : {rate_results['rate_ratio']:.4f}  (target: > 0.95)")
    print("=" * 55)

    checks = {
        'top1_pass': top1 >= 0.85,
        'top5_pass': top5 >= 0.97,
        'ratio_pass': rate_results['rate_ratio'] >= 0.95,
    }
    print("\n  Acceptance Criteria")
    print(f"  Top-1 ≥ 85% :  {'✓ PASS' if checks['top1_pass'] else '✗ FAIL'}")
    print(f"  Top-5 ≥ 97% :  {'✓ PASS' if checks['top5_pass'] else '✗ FAIL'}")
    print(f"  Ratio ≥ 0.95:  {'✓ PASS' if checks['ratio_pass'] else '✗ FAIL'}")
    print("=" * 55 + "\n")

    return {'top1_acc': top1, 'top5_acc': top5, **rate_results, **checks}


def plot_training_curve(history: dict, save_dir: str = 'results') -> str:
    os.makedirs(save_dir, exist_ok=True)

    epochs = np.arange(1, len(history['train_loss']) + 1)
    train_loss = np.array(history['train_loss'])
    val_acc = np.array(history['val_acc']) * 100.0

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f1117')
    ax1.set_facecolor('#0f1117')

    ax1.plot(epochs, train_loss, color='#7c83fd', linewidth=2, label='Train Loss', marker='o', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=13, color='white')
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=13, color='#7c83fd')
    ax1.tick_params(axis='y', labelcolor='#7c83fd', colors='white')
    ax1.tick_params(axis='x', colors='white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#444')

    ax2 = ax1.twinx()
    ax2.set_facecolor('#0f1117')
    ax2.plot(epochs, val_acc, color='#f7b731', linewidth=2, label='Val Top-1 Acc', marker='s', markersize=3)
    ax2.set_ylabel('Top-1 Accuracy (%)', fontsize=13, color='#f7b731')
    ax2.tick_params(axis='y', labelcolor='#f7b731', colors='white')
    ax2.axhline(85, color='#ff6b6b', linestyle='--', linewidth=1.2, label='85% target', alpha=0.7)

    best_epoch = history.get('best_epoch', 0)
    if best_epoch > 0:
        ax2.axvline(best_epoch, color='#26de81', linestyle=':', linewidth=1.5, label=f'Best epoch ({best_epoch})', alpha=0.8)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower right', facecolor='#1a1d2e', edgecolor='#444', labelcolor='white', fontsize=10)

    plt.title('Training Curve — 6G Beam Prediction MLP', fontsize=14, color='white', pad=12)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'training_curve.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Training curve saved → {save_path}")
    return save_path


def plot_beam_map_comparison(
    positions: np.ndarray,
    optimal_labels: np.ndarray,
    pred_labels: np.ndarray,
    save_dir: str = 'results',
    filename: str = 'beam_map_comparison.png',
    N_b: int = 64,
) -> str:
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#ffffff')
    gs = fig.add_gridspec(3, 2, height_ratios=[0.08, 1, 0.15], hspace=0.4, wspace=0.15)

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, '6G Beam Assignment: Physical Location vs. Antenna Beam',
                  ha='center', va='center', fontsize=18, fontweight='bold', color='#1a1a2e')

    cmap = 'viridis'

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.scatter(positions[:, 0], positions[:, 1], c=optimal_labels, cmap=cmap, vmin=0, vmax=N_b - 1, s=3, alpha=0.7)
    ax1.set_title('Ground Truth (Optimal Beam)', fontsize=15, fontweight='bold', pad=12, color='#2c3e50')
    ax1.set_xlabel('X Position (meters)', fontsize=12, color='#555')
    ax1.set_ylabel('Y Position (meters)', fontsize=12, color='#555')
    ax1.grid(True, linestyle='--', alpha=0.2, color='#aaa')

    ax2 = fig.add_subplot(gs[1, 1])
    sc2 = ax2.scatter(positions[:, 0], positions[:, 1], c=pred_labels, cmap=cmap, vmin=0, vmax=N_b - 1, s=3, alpha=0.7)
    ax2.set_title('AI Model Prediction', fontsize=15, fontweight='bold', pad=12, color='#2c3e50')
    ax2.set_xlabel('X Position (meters)', fontsize=12, color='#555')
    ax2.set_ylabel('Y Position (meters)', fontsize=12, color='#555')
    ax2.grid(True, linestyle='--', alpha=0.2, color='#aaa')

    ax_cbar = fig.add_subplot(gs[2, :])
    cbar = fig.colorbar(sc2, cax=ax_cbar, orientation='horizontal')
    cbar.set_label('Beam Index (0 to 63)', fontsize=12, color='#333')

    explanation = (
        "HOW TO READ THIS: Each dot is a user device. Color shows the assigned antenna beam. "
        "Nearby users share similar colors. When AI (right) matches Ground Truth (left), the model works."
    )
    fig.text(0.5, 0.02, explanation, ha='center', va='bottom', fontsize=11.5, color='#2c3e50',
             bbox=dict(boxstyle='round,pad=1', facecolor='#e8f4f8', edgecolor='#3498db', linewidth=2, alpha=0.95),
             transform=fig.transFigure)

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Beam map comparison saved → {save_path}")
    return save_path


def plot_rate_cdf(
    rates_pred: np.ndarray,
    rates_opt: np.ndarray,
    rates_rand: np.ndarray,
    snr_db: float = 20.0,
    save_dir: str = 'results',
) -> str:
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')

    def _cdf(data):
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    for rates, label, color in [
        (rates_opt, 'Optimal Beam', '#26de81'),
        (rates_pred, 'Predicted Beam', '#f7b731'),
        (rates_rand, 'Random Beam', '#ff6b6b'),
    ]:
        x, y = _cdf(rates)
        ax.plot(x, y * 100, label=label, color=color, linewidth=2)

    ax.set_xlabel('Achievable Rate (bps/Hz)', fontsize=13, color='white')
    ax.set_ylabel('CDF (%)', fontsize=13, color='white')
    ax.set_title(f'Achievable Rate CDF @ SNR = {snr_db} dB', fontsize=14, color='white', pad=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.legend(facecolor='#1a1d2e', edgecolor='#444', labelcolor='white', fontsize=11)
    ax.grid(True, color='#2a2d3e', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'rate_cdf.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Rate CDF saved → {save_path}")
    return save_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate BeamPredictor.')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_mlp.pt', help='Path to checkpoint')
    parser.add_argument('--real', action='store_true', default=True, help='Use DeepMIMO dataset')
    parser.add_argument('--no-real', action='store_false', dest='real', help='Use synthetic data')
    parser.add_argument('--scenario', type=str, default='o1_28', help='DeepMIMO scenario')
    parser.add_argument('--snr-db', type=float, default=20.0, help='Evaluation SNR in dB')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory for plots')
    parser.add_argument('--n-synthetic', type=int, default=18000, help='Number of synthetic users')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    from src.dataset import (
        dft_codebook, generate_synthetic_dataset, load_deepmimo_v4,
        filter_valid_users, compute_beam_labels, make_dataloaders,
    )
    from src.model import BeamPredictor

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.real:
        positions, channels = load_deepmimo_v4(args.scenario)
    else:
        positions, channels = generate_synthetic_dataset(args.n_synthetic)

    channels, positions = filter_valid_users(channels, positions)
    codebook = dft_codebook(N_t=64, N_b=64)
    labels = compute_beam_labels(channels, codebook)

    _, _, test_loader, scaler, ch_split = make_dataloaders(positions, labels, channels, batch_size=256)

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found at '{args.checkpoint}'. Run training first.")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = ckpt.get('config', {})

    input_dim = cfg.get('input_dim', 6)
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
    print(f"[Evaluate] Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, val_acc = {ckpt.get('val_acc', 0)*100:.2f}%")

    channels_test = ch_split.get('test', channels[:100])
    metrics = evaluate_all(model, test_loader, channels_test, codebook, snr_db=args.snr_db, device=device)

    os.makedirs(args.results_dir, exist_ok=True)

    history_path = 'checkpoints/training_history.json'
    if os.path.exists(history_path):
        with open(history_path) as f:
            plot_training_curve(json.load(f), save_dir=args.results_dir)

    test_positions = positions[-len(metrics['pred_beams']):]
    plot_beam_map_comparison(test_positions, metrics['optimal_beams'], metrics['pred_beams'], save_dir=args.results_dir, N_b=codebook.shape[0])
    plot_rate_cdf(metrics['rates_pred'], metrics['rates_opt'], metrics['rates_rand'], snr_db=args.snr_db, save_dir=args.results_dir)
