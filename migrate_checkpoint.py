"""
migrate_checkpoint.py — ZenoZero checkpoint migration tool

Migrates a checkpoint saved by the OLD CompactReversiNet (no residual
connections, 64-unit value head) to the NEW architecture (residual
skip connections, 128-unit value head).

Architecture diff:
  OLD                               NEW
  ─────────────────────────────     ──────────────────────────────
  conv1  Conv2d(4→128, 3×3)         conv1  Conv2d(4→128, 3×3)   ✓ copy
  bn1    BatchNorm2d(128)            bn1    BatchNorm2d(128)      ✓ copy
                                    proj1  Conv2d(4→128, 1×1)    ↺ init  (new)
  conv2  Conv2d(128→128, 3×3)       conv2  Conv2d(128→128, 3×3) ✓ copy
  bn2    BatchNorm2d(128)            bn2    BatchNorm2d(128)      ✓ copy
  conv3  Conv2d(128→128, 3×3)       conv3  Conv2d(128→128, 3×3) ✓ copy
  bn3    BatchNorm2d(128)            bn3    BatchNorm2d(128)      ✓ copy
  conv4  Conv2d(128→128, 3×3)       conv4  Conv2d(128→128, 3×3) ✓ copy
  bn4    BatchNorm2d(128)            bn4    BatchNorm2d(128)      ✓ copy

  p_conv Conv2d(128→2, 1×1)         p_conv Conv2d(128→2, 1×1)   ✓ copy
  p_bn   BatchNorm2d(2)             p_bn   BatchNorm2d(2)        ✓ copy
  p_fc   Linear(128, 65)            p_fc   Linear(128, 65)       ✓ copy

  v_conv Conv2d(128→1, 1×1)         v_conv Conv2d(128→1, 1×1)   ✓ copy
  v_bn   BatchNorm2d(1)             v_bn   BatchNorm2d(1)        ✓ copy
  v_fc1  Linear(64, 64)             v_fc1  Linear(64, 128)       ↺ reinit (shape mismatch)
  v_fc2  Linear(64, 1)              v_fc2  Linear(128, 1)        ↺ reinit (shape mismatch)

What transfers:  all 4 conv blocks + BNs + full policy head + v_conv/v_bn
What reinits:    proj1 (new), v_fc1, v_fc2 (widened bottleneck)
Optimizer state: always dropped (incompatible param count) — Adam restarts from warmup

Usage:
  python3 migrate_checkpoint.py checkpoints/iter_0100.pt
  python3 migrate_checkpoint.py checkpoints/iter_0100.pt --out checkpoints/iter_0100_v3.pt
  python3 migrate_checkpoint.py checkpoints/iter_0100.pt --verify
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ── Minimal architecture definitions ─────────────────────────────────────────
# Defined inline so the script is self-contained — no dependency on the
# (possibly modified) reversi_phase5_topology_core.py in the local directory.

import torch.nn.functional as F
from typing import Tuple, List, Optional


class _OldNet(nn.Module):
    """Old architecture — no residuals, 64-unit value head."""
    NUM_ACTIONS = 65

    def __init__(self, board_size: int = 8, channels: int = 128):
        super().__init__()
        S = board_size
        self.conv1 = nn.Conv2d(4, channels, 3, padding=1); self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1); self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1); self.bn3 = nn.BatchNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, channels, 3, padding=1); self.bn4 = nn.BatchNorm2d(channels)
        self.p_conv = nn.Conv2d(channels, 2, 1); self.p_bn = nn.BatchNorm2d(2)
        self.p_fc   = nn.Linear(2 * S * S, self.NUM_ACTIONS)
        self.v_conv = nn.Conv2d(channels, 1, 1); self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1  = nn.Linear(S * S, 64)
        self.v_fc2  = nn.Linear(64, 1)


class _NewNet(nn.Module):
    """New architecture — residual blocks, 128-unit value head."""
    NUM_ACTIONS = 65

    def __init__(self, board_size: int = 8, channels: int = 128):
        super().__init__()
        S = board_size
        self.conv1 = nn.Conv2d(4, channels, 3, padding=1); self.bn1 = nn.BatchNorm2d(channels)
        self.proj1 = nn.Conv2d(4, channels, 1, bias=False)   # new — projection for block-1 skip
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1); self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1); self.bn3 = nn.BatchNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, channels, 3, padding=1); self.bn4 = nn.BatchNorm2d(channels)
        self.p_conv = nn.Conv2d(channels, 2, 1); self.p_bn = nn.BatchNorm2d(2)
        self.p_fc   = nn.Linear(2 * S * S, self.NUM_ACTIONS)
        self.v_conv = nn.Conv2d(channels, 1, 1); self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1  = nn.Linear(S * S, 128)    # widened: 64 → 128
        self.v_fc2  = nn.Linear(128, 1)         # widened: 64 → 128


# ── Migration logic ───────────────────────────────────────────────────────────

def migrate(
    src_path:  str,
    dst_path:  str,
    channels:  int  = 128,
    verify:    bool = False,
    verbose:   bool = True,
) -> dict:
    """
    Load old checkpoint, copy compatible weights into new architecture,
    save migrated checkpoint.

    Returns: dict with keys transferred, reinitialised, skipped.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    if not src_path.exists():
        print(f"✗  Source not found: {src_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load old checkpoint ───────────────────────────────────────────────────
    print(f"\n  Loading: {src_path}")
    ckpt = torch.load(src_path, map_location='cpu', weights_only=False)

    old_weights = ckpt.get('model_state_dict')
    if old_weights is None:
        print("✗  Checkpoint has no 'model_state_dict' key.", file=sys.stderr)
        sys.exit(1)

    src_iteration  = ckpt.get('iteration', '?')
    src_thresholds = ckpt.get('thresholds', {
        'H_v_thresh': 0.15, 'G_thresh': 0.85, 'Var_Q_thresh': 0.01
    })
    src_log        = ckpt.get('log', [])
    src_config     = ckpt.get('config', {})

    print(f"  Source iteration: {src_iteration}")
    print(f"  Keys in source:   {sorted(old_weights.keys())[:6]} ...")

    # Detect architecture: presence of 'proj1.weight' means already new arch
    if 'proj1.weight' in old_weights:
        print("\n  ⚠  Source checkpoint already has new architecture (proj1 found).")
        print("     If shapes match, migration is a no-op — saving as-is.")
        new_net = _NewNet(8, channels)
        new_net.load_state_dict(old_weights)
        _save(dst_path, new_net, src_iteration, src_thresholds, src_log, src_config)
        return {'transferred': list(old_weights.keys()), 'reinitialised': [], 'skipped': []}

    # ── Build new network (randomly initialised) ──────────────────────────────
    new_net    = _NewNet(8, channels)
    new_weights = new_net.state_dict()

    transferred   = []
    reinitialised = []
    skipped       = []

    for key, new_tensor in new_weights.items():
        if key in old_weights:
            old_tensor = old_weights[key]
            if old_tensor.shape == new_tensor.shape:
                new_weights[key] = old_tensor.clone()
                transferred.append(key)
            else:
                # Shape mismatch — keep random init, log it
                reinitialised.append(
                    f"{key}  old={tuple(old_tensor.shape)}→new={tuple(new_tensor.shape)}"
                )
        else:
            # Key not in old checkpoint — new layer, keep random init
            reinitialised.append(f"{key}  (new layer, not in source)")

    # Apply
    new_net.load_state_dict(new_weights)

    # ── Initialisation quality for new layers ─────────────────────────────────
    # proj1: initialise as near-identity projection so the residual connection
    # starts as a no-op (doesn't disrupt pre-trained conv1 features immediately).
    # Conv2d(4→128, 1×1): initialise weights to small values, zero bias.
    with torch.no_grad():
        nn.init.kaiming_normal_(new_net.proj1.weight, mode='fan_out', nonlinearity='relu')
        new_net.proj1.weight.data *= 0.1   # scale down — starts near-zero skip contribution

    # v_fc1 (64→128): Kaiming normal for ReLU activation
    with torch.no_grad():
        nn.init.kaiming_normal_(new_net.v_fc1.weight, nonlinearity='relu')
        nn.init.zeros_(new_net.v_fc1.bias)
        nn.init.kaiming_normal_(new_net.v_fc2.weight, nonlinearity='relu')
        nn.init.zeros_(new_net.v_fc2.bias)

    # ── Save migrated checkpoint ──────────────────────────────────────────────
    _save(dst_path, new_net, src_iteration, src_thresholds, src_log, src_config)

    # ── Report ────────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n  ── Transferred ({len(transferred)} layers) ──────────────────────")
        for k in transferred:
            t = new_weights[k]
            print(f"    ✓  {k:<45} {str(tuple(t.shape))}")

        print(f"\n  ── Reinitialised ({len(reinitialised)} layers) ─────────────────")
        for r in reinitialised:
            print(f"    ↺  {r}")

        if skipped:
            print(f"\n  ── Skipped ({len(skipped)}) ─────────────────────────────────")
            for s in skipped:
                print(f"    ─  {s}")

        print(f"\n  ── Summary ──────────────────────────────────────────────────")
        total = len(transferred) + len(reinitialised) + len(skipped)
        print(f"    Transferred:   {len(transferred)}/{total}  layers")
        print(f"    Reinitialised: {len(reinitialised)}/{total}  layers  "
              f"(proj1 + v_fc1 + v_fc2)")
        print(f"    Optimizer:     dropped  (Adam will restart from warmup)")
        print(f"    Saved →        {dst_path}")

    # ── Optional verification ─────────────────────────────────────────────────
    if verify:
        _verify(src_path, dst_path, transferred, channels)

    return {
        'transferred':   transferred,
        'reinitialised': reinitialised,
        'skipped':       skipped,
    }


def _save(
    dst_path:   Path,
    net:        _NewNet,
    iteration:  int,
    thresholds: dict,
    log:        list,
    config:     dict,
):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict':     net.state_dict(),
        'optimizer_state_dict': None,   # intentionally dropped — incompatible param count
        'iteration':            iteration,
        'config':               config,
        'thresholds':           thresholds,
        'log':                  log,
        'migration':            {
            'migrated': True,
            'note': 'Migrated from pre-residual architecture. '
                    'Optimizer state dropped — Adam restarts from LR warmup. '
                    'Reinitialised: proj1, v_fc1, v_fc2.',
        },
    }, dst_path)


def _verify(src_path: Path, dst_path: Path, transferred_keys: list, channels: int):
    """
    Load both checkpoints, confirm transferred weights are byte-identical.
    """
    print(f"\n  ── Verification ──────────────────────────────────────────────")
    old_ckpt = torch.load(src_path, map_location='cpu', weights_only=False)
    new_ckpt = torch.load(dst_path, map_location='cpu', weights_only=False)
    old_w = old_ckpt['model_state_dict']
    new_w = new_ckpt['model_state_dict']

    all_ok = True
    for key in transferred_keys:
        if key not in old_w or key not in new_w:
            print(f"    ✗  {key}  —  missing in one side")
            all_ok = False
            continue
        match = torch.allclose(old_w[key].float(), new_w[key].float(), atol=1e-7)
        status = "✓" if match else "✗"
        if not match:
            all_ok = False
        print(f"    {status}  {key}")

    # Check new layers are NOT zero (would indicate init failure)
    for key in ['proj1.weight', 'v_fc1.weight', 'v_fc2.weight']:
        if key in new_w:
            norm = new_w[key].float().norm().item()
            ok = norm > 1e-6
            print(f"    {'✓' if ok else '✗'}  {key}  norm={norm:.4f}  (should be >0)")
            if not ok:
                all_ok = False

    print(f"\n    {'All checks passed ✓' if all_ok else 'SOME CHECKS FAILED ✗'}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Migrate ZenoZero checkpoint to new residual architecture.'
    )
    p.add_argument('source',
                   help='Path to old checkpoint (e.g. checkpoints/iter_0100.pt)')
    p.add_argument('--out', default=None,
                   help='Output path. Default: <source_stem>_v3.pt in same dir')
    p.add_argument('--channels', type=int, default=128,
                   help='Network channel width (default: 128)')
    p.add_argument('--verify', action='store_true',
                   help='After saving, verify transferred weights are byte-identical')
    p.add_argument('--quiet', action='store_true',
                   help='Suppress per-layer output')
    cfg = p.parse_args()

    src = Path(cfg.source)
    if cfg.out:
        dst = Path(cfg.out)
    else:
        dst = src.parent / f"{src.stem}_v3.pt"

    print(f"\n{'='*65}")
    print(f"  ZenoZero Checkpoint Migration")
    print(f"{'='*65}")
    print(f"  Source:   {src}")
    print(f"  Target:   {dst}")
    print(f"  Channels: {cfg.channels}")
    print(f"  Verify:   {cfg.verify}")
    print(f"{'='*65}")

    result = migrate(
        src_path  = str(src),
        dst_path  = str(dst),
        channels  = cfg.channels,
        verify    = cfg.verify,
        verbose   = not cfg.quiet,
    )

    print(f"\n{'='*65}")
    print(f"  Migration complete.")
    print(f"  Next step — resume training from migrated checkpoint:")
    print(f"\n  python3 reversi_phase5_training.py \\")
    print(f"    --resume {dst} \\")
    print(f"    --iterations 200 \\")
    print(f"    --lr 3e-4 \\")
    print(f"    --lr-warmup-iters 5 \\")
    print(f"    --checkpoint-dir checkpoints_v3 \\")
    print(f"    --log-file training_log_v3.jsonl")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()