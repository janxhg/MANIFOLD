#!/usr/bin/env python3
"""
GFN Unified Training Script
============================

Usage:
    python scripts/train.py --model configs/model/gfn_medium.yaml \\
                            --training configs/training/math_oracle.yaml \\
                            --hardware configs/hardware/rtx_4090.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import os
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn import Manifold, GFNLoss, RiemannianAdam
from gfn.losses import hamiltonian_loss, curiosity_loss, geodesic_regularization
from gfn.math_dataset import MathDataset
from gfn.mixed_dataset import MixedHFDataset
from gfn.safety import GPUMonitor



def load_config(path: str) -> dict:
    """Load YAML configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def setup_device(hw_cfg: dict) -> torch.device:
    """Setup compute device based on config."""
    cfg = hw_cfg.get('hardware', {})
    if torch.cuda.is_available() and cfg.get('gpu_enabled', True):
        return torch.device('cuda')
    return torch.device('cpu')


def build_model(model_cfg: dict, device: torch.device) -> nn.Module:
    """Instantiate Manifold model."""
    cfg = model_cfg['model']
    
    # Physics configuration for Implicit Readout/Functional Embedding
    physics_config = {
        "embedding": {"type": "functional", "mode": "binary", "coord_dim": 32},
        "readout": {"type": "implicit", "coord_dim": 32},
        "active_inference": {
            "enabled": True, 
            "plasticity": 0.1,
            "dynamic_time": {"enabled": True}
        },
        "fractal": {"enabled": True, "threshold": 0.5},
        "singularities": {"enabled": True, "strength": 5.0},
        "symmetries": {"enabled": True}
    }
    
    if 'physics_config' in cfg:
        physics_config.update(cfg['physics_config'])

    model = Manifold(
        vocab_size=cfg['vocab_size'],
        dim=cfg['dim'],
        depth=cfg['depth'],
        rank=cfg['rank'],
        heads=cfg.get('heads', 4),
        integrator_type=cfg.get('integrator', 'leapfrog'), 
        use_scan=cfg.get('use_scan', True),
        physics_config=physics_config
    )
    return model.to(device)


def build_dataset(train_cfg: dict):
    """Build dataset based on training config."""
    cfg = train_cfg['training']
    if cfg['dataset'] == 'math':
        return MathDataset(
            size=cfg.get('dataset_size'),
            max_digits=cfg.get('max_digits', 8),
            file_path=cfg.get('dataset_path')
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg['dataset']}")


def run_demo(model: nn.Module, dataset, device: torch.device):
    """Run live inference demo."""
    model.eval()
    test_cases = ["42+9=", "131-31=", "50*5=", "999+1=", "123*10="]
    
    print("\n[LIVE DEMO]")
    with torch.no_grad():
        for prompt in test_cases:
            ids = [dataset.char_to_id[c] for c in prompt]
            input_seq = torch.tensor([ids]).to(device)
            logits, state = model(input_seq)
            
            generated = list(ids)
            curr_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            generated.append(curr_token.item())
            
            for _ in range(12):
                logits, state = model(curr_token, state=state)
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                tok_id = next_token.item()
                if tok_id == dataset.char_to_id['<EOS>']:
                    break
                generated.append(tok_id)
                curr_token = next_token.unsqueeze(0)
            
            result = dataset.decode(generated).split('=')[-1]
            print(f"  {prompt} {result}")
    
    print("-----------\n")
    model.train()


def train(model_cfg: dict, train_cfg: dict, hw_cfg: dict):
    """Execution of the training loop."""
    device = setup_device(hw_cfg)
    model = build_model(model_cfg, device)
    
    dataset = build_dataset(train_cfg)
    train_params = train_cfg['training']
    hw_params = hw_cfg['hardware']
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model Parameters: {param_count:.2f}M")
    print(f"Device: {device}")
    
    safety = GPUMonitor(threshold_temp=hw_params.get('max_temp_c', 80))
    safety.start()
    
    optimizer = RiemannianAdam(
        model.parameters(),
        lr=train_params.get('learning_rate', 3e-4),
        weight_decay=train_params.get('weight_decay', 0.01),
        retraction='normalize', 
        max_norm=10.0
    )
    
    task_criterion = nn.MSELoss()
    
    lambda_h = train_params.get('lambda_h', 0.01)
    lambda_c = train_params.get('lambda_c', 0.05)
    lambda_g = train_params.get('lambda_g', 0.001)
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_params['batch_size'],
        num_workers=hw_params.get('num_workers', 4),
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    
    ckpt_path = Path(train_cfg['checkpoint']['path'])
    ckpt_path.mkdir(parents=True, exist_ok=True)
    
    model.train()
    
    try:
        for epoch in range(train_params['epochs']):
            epoch_loss = 0.0
            epoch_t0 = time.time()
            
            for batch_idx, inputs in enumerate(dataloader):
                if batch_idx >= train_params.get('steps_per_epoch', 1000): break
                
                safety.check()
                inputs = inputs.to(device, non_blocking=True)
                
                # Prepare targets for coordinate regression (next token prediction)
                target_tokens = torch.roll(inputs, -1, dims=1)
                
                # O(1) Target Generation (Accessing Binary Coordinates Directly)
                coord_dim = 32
                mask = 2**torch.arange(coord_dim).to(device)
                bits = (target_tokens.unsqueeze(-1) & mask) > 0
                target_coords = bits.float() * 2 - 1
                
                optimizer.zero_grad()
                
                logits, (x_final, v_final), christoffels = model(inputs)
                
                # Align predictions (t -> t+1)
                pred_valid = logits[:, :-1, :]
                targ_valid = target_coords[:, :-1, :]
                
                loss_mse = task_criterion(pred_valid, targ_valid)
                total_loss = loss_mse
                
                # Physics regularization
                loss_phy = 0.0
                if christoffels:
                    loss_phy += geodesic_regularization(None, christoffels, lambda_g)
                    
                if v_final is not None:
                    loss_phy += curiosity_loss([v_final], lambda_c)
                
                loss = total_loss + loss_phy
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % train_params.get('log_interval', 10) == 0:
                    dt = time.time() - epoch_t0
                    speed = (batch_idx * inputs.shape[0]) / max(dt, 0.01)
                    print(f"Ep {epoch} | Step {batch_idx} | Total Loss: {loss.item():.4f} | MSE: {loss_mse.item():.4f} | Speed: {speed:.1f} tok/s")
            
            avg_loss = epoch_loss / train_params.get('steps_per_epoch', 1000)
            print(f"Epoch {epoch} Completed. Avg Loss: {avg_loss:.4f}")
            
            if epoch % train_cfg['checkpoint'].get('save_interval', 1) == 0:
                torch.save(model.state_dict(), ckpt_path / f"epoch_{epoch}.pt")

    except KeyboardInterrupt:
        torch.save(model.state_dict(), ckpt_path / "interrupted.pt")
    finally:
        safety.stop()


def main():
    parser = argparse.ArgumentParser(description="GFN Unified Training")
    parser.add_argument('--model', type=str, required=True, help="Path to model config YAML")
    parser.add_argument('--training', type=str, required=True, help="Path to training config YAML")
    parser.add_argument('--hardware', type=str, required=True, help="Path to hardware config YAML")
    parser.add_argument('--reset-optimizer', action='store_true', help="Reset optimizer state when resuming")
    args = parser.parse_args()
    
    model_cfg = load_config(args.model)
    train_cfg = load_config(args.training)
    hw_cfg = load_config(args.hardware)
    
    train(model_cfg, train_cfg, hw_cfg)


if __name__ == "__main__":
    main()
