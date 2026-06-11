"""
train.py — Training Loop for 6G Beam Prediction
=================================================
Trains the BeamPredictor MLP and saves the best checkpoint.
"""

import os
import sys
import time
import argparse
import json
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


DEFAULT_CONFIG: Dict[str, Any] = {
    'use_real': True,
    'scenario': 'o1_28',
    'n_synthetic': 18000,
    'input_dim': 6,
    'hidden_dims': [512, 256, 128],
    'num_classes': 64,
    'dropout': 0.2,
    'n_res_blocks': 2,
    'epochs': 200,
    'batch_size': 256,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'label_smoothing': 0.05,
    'log_every': 1,
    'seed': 42,
    'checkpoint_dir': 'checkpoints',
    'checkpoint_name': 'best_mlp.pt',
    'history_name': 'training_history.json',
    'use_amp': True,
    'grad_clip': 1.0,
}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
    use_amp: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
        else:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            preds = logits.argmax(dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += len(y_batch)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1), total_correct / max(total_samples, 1)


def train(config: Dict[str, Any] = None) -> Dict[str, list]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    if not isinstance(cfg['epochs'], int) or not (1 <= cfg['epochs'] <= 10000):
        raise ValueError(f"Invalid epochs: {cfg['epochs']}")
    if not isinstance(cfg['batch_size'], int) or not (1 <= cfg['batch_size'] <= 100000):
        raise ValueError(f"Invalid batch_size: {cfg['batch_size']}")
    if not isinstance(cfg['lr'], (int, float)) or not (0 < cfg['lr'] <= 10):
        raise ValueError(f"Invalid lr: {cfg['lr']}")

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    requested_device = cfg.get('device', 'auto')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if requested_device == 'auto' else torch.device(requested_device)
    print(f"[Train] Device: {device}")

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)} | AMP={'enabled' if cfg.get('use_amp') else 'disabled'}")

    from src.dataset import (
        dft_codebook, generate_synthetic_dataset, load_deepmimo_v4,
        filter_valid_users, compute_beam_labels, make_dataloaders,
    )

    if cfg['use_real']:
        print(f"[Train] Loading DeepMIMO '{cfg['scenario']}'...")
        positions, channels = load_deepmimo_v4(
            scenario=cfg['scenario'], max_users=cfg.get('max_users'), seed=cfg['seed']
        )
    else:
        print(f"[Train] Using synthetic dataset ({cfg['n_synthetic']:,} users)")
        positions, channels = generate_synthetic_dataset(n_users=cfg['n_synthetic'], seed=cfg['seed'])

    channels, positions = filter_valid_users(channels, positions)
    codebook = dft_codebook(N_t=cfg['num_classes'], N_b=cfg['num_classes'])
    labels = compute_beam_labels(channels, codebook)

    train_loader, val_loader, test_loader, scaler, ch_split = make_dataloaders(
        positions, labels, channels, batch_size=cfg['batch_size'], seed=cfg['seed'],
    )

    from src.model import BeamPredictor, count_parameters

    model = BeamPredictor(
        input_dim=cfg['input_dim'],
        hidden_dims=cfg['hidden_dims'],
        num_classes=cfg['num_classes'],
        dropout=cfg['dropout'],
        n_res_blocks=cfg.get('n_res_blocks', 2),
    ).to(device)
    count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get('label_smoothing', 0.05))

    use_amp = cfg.get('use_amp', True) and device.type == 'cuda'
    if use_amp:
        print("[Train] Mixed precision (bfloat16) enabled")

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'best_val_acc': 0.0, 'best_epoch': 0}

    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    best_ckpt_path = os.path.join(cfg['checkpoint_dir'], cfg['checkpoint_name'])

    print(f"\n{'─' * 60}")
    print(f"  Training for {cfg['epochs']} epochs | batch size {cfg['batch_size']}")
    print(f"{'─' * 60}")

    t_start = time.time()

    for epoch in range(1, cfg['epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip=cfg.get('grad_clip', 1.0), use_amp=use_amp)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            history['best_epoch'] = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': {**cfg, **model.config},
            }, best_ckpt_path)

        if epoch % cfg['log_every'] == 0 or epoch == 1:
            elapsed = time.time() - t_start
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:>3}/{cfg['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | LR: {lr_now:.2e} | Elapsed: {elapsed:.1f}s")

    total_time = time.time() - t_start
    print(f"{'─' * 60}")
    print(f"Training complete in {total_time:.1f}s")
    print(f"Best Val Accuracy: {history['best_val_acc']*100:.2f}% (Epoch {history['best_epoch']})")
    print(f"Checkpoint saved → {best_ckpt_path}")

    history_path = os.path.join(cfg['checkpoint_dir'], cfg['history_name'])
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved → {history_path}")

    return history


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train BeamPredictor MLP for 6G beam prediction.')
    parser.add_argument('--real', action='store_true', default=True, help='Use DeepMIMO dataset')
    parser.add_argument('--no-real', action='store_false', dest='real', help='Use synthetic data')
    parser.add_argument('--scenario', type=str, default='o1_28', help='DeepMIMO scenario name')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num-classes', type=int, default=64, help='Codebook size')
    parser.add_argument('--n-synthetic', type=int, default=18000, help='Number of synthetic users')
    parser.add_argument('--max-users', type=int, default=None, help='Max users to load')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to train on')
    parser.add_argument('--no-amp', action='store_true', default=False, help='Disable mixed precision')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    config = {
        'use_real': args.real,
        'scenario': args.scenario,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_classes': args.num_classes,
        'n_synthetic': args.n_synthetic,
        'max_users': args.max_users,
        'checkpoint_dir': args.checkpoint_dir,
        'seed': args.seed,
        'device': args.device,
        'use_amp': not args.no_amp,
    }

    train(config)
