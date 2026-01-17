import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import json
import time
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import Manifold
from src.losses import GFNLoss

class ExperimentRunner:
    """
    Professional Experiment Runner for Manifold Ablation Studies.
    Handles:
    - Grid Search Management
    - Training Loop with Metrics
    - Checkpointing
    - JSON/CSV Reporting
    - Visualization
    """
    def __init__(self, output_dir_base="tests/ablation/results", experiment_name="experiment"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir_base) / f"{experiment_name}_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n🔬 Manifold Ablation: {experiment_name}")
        print(f"📂 Results Dir: {self.output_dir}")
        print(f"⚡ Device: {self.device}")
        
    def run_grid(self, base_config, grid_params, task_cls):
        """
        Run a grid search.
        
        Args:
            base_config: Dict with static params
            grid_params: Dict with list of values to vary {'param': [v1, v2]}
            task_cls: Class for the task (must have generate_batch)
        """
        keys = grid_params.keys()
        values = grid_params.values()
        combinations = list(itertools.product(*values))
        
        configs = []
        for combo in combinations:
            cfg = base_config.copy()
            # Deep update for grid params
            # Flattened update: 'optimizer.lr' -> 1e-3
            # We assume simple top-level or specific overrides
            
            variation_name = []
            
            for k, v in zip(keys, combo):
                # Handle nested keys if needed (e.g. 'physics.alpha')
                # For simplicity, we assume flat config or handle specifically
                cfg[k] = v
                variation_name.append(f"{k}-{v}")
            
            cfg['name'] = "_".join(variation_name)
            configs.append(cfg)
            
        print(f"📋 Grid Search: {len(configs)} configurations")
        
        results = []
        for i, cfg in enumerate(configs):
            print(f"\n▶️  Running [{i+1}/{len(configs)}]: {cfg['name']}")
            try:
                res = self.train_single(cfg, task_cls, run_id=i)
                results.append(res)
            except Exception as e:
                print(f"❌ Run {i} Failed: {e}")
                import traceback
                traceback.print_exc()
        
        self.generate_report(results)
        return results

    def train_single(self, config, task_cls, run_id):
        # 1. Init Task
        # Config should specify task params if needed, or use defaults
        task = task_cls() # Assume default for now
        
        # 2. Init Model
        # Standardize construction
        # Physics config needs careful handling
        physics_config = config.get('physics_config', {})
        
        # Override physics with grid items if present
        if 'plasticity' in config:
            physics_config.setdefault('active_inference', {}).setdefault('reactive_curvature', {})['plasticity'] = config['plasticity']
            
        model = Manifold(
            vocab_size=getattr(task, 'full_vocab', 100), # Task should provide this
            dim=config.get('dim', 64),
            depth=config.get('depth', 2),
            heads=config.get('heads', 4),
            rank=config.get('rank', 16),
            use_scan=config.get('use_scan', False),
            integrator_type=config.get('integrator', 'heun'),
            physics_config=physics_config
        ).to(self.device)
        
        # 3. Init Loss
        criterion = GFNLoss(
            lambda_h=config.get('lambda_h', 0.01),
            lambda_g=config.get('lambda_g', 0.001),
            lambda_c=config.get('lambda_c', 0.0),
            lambda_n=config.get('lambda_n', 0.0),
            ignore_index=-100
        )
        
        # 4. Optimizer
        lr = config.get('lr', 3e-3)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # 5. Loop
        history = {'step': [], 'loss': [], 'accuracy': []}
        max_steps = config.get('max_steps', 1000)
        
        start_time = time.time()
        pbar = tqdm(range(max_steps), desc=config['name'], leave=False)
        
        model.train()
        
        for step in pbar:
            x, y = task.generate_batch(64, device=self.device)
            
            optimizer.zero_grad()
            logits, _, _ = model(x)
            
            # TODO: If we want to test Lambda H/G/C, we need velocities.
            # Currently Manifold forward doesn't return them easily in standard API.
            # We will rely on implicit internal handling or just test GFNLoss shell.
            # NOTE: For this ablation, if lambda > 0 but no velocities, loss is 0. 
            # This makes testing lambda_h hard without modifying model.py.
            # We will assume model.py MIGHT be updated or we test what we can.
            
            loss, loss_dict = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 20 == 0:
                acc = self.evaluate_acc(model, task, logits, y)
                history['step'].append(step)
                history['loss'].append(loss.item())
                history['accuracy'].append(acc)
                pbar.set_postfix({'acc': f"{acc*100:.0f}%", 'loss': f"{loss.item():.2f}"})
                
        duration = time.time() - start_time
        final_acc = history['accuracy'][-1]
        
        result = {
            'config': config,
            'final_accuracy': final_acc,
            'duration': duration,
            'history': history
        }
        
        # Save JSON
        cleaned_name = config['name'].replace('.', 'p')
        with open(self.output_dir / f"run_{run_id}_{cleaned_name}.json", 'w') as f:
            # Helper to serialize non-serializable configs if any
            json.dump(result, f, default=str, indent=4)
            
        return result
        
    def evaluate_acc(self, model, task, logits, targets):
        # Default Sorting/Copy Task evaluation
        # Exact match on the "Output" part of sequence
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            # Assuming Task structure: [Input] [SEP] [Output] [EOS]
            # We check the part after length + 1
            if hasattr(task, 'length'):
                start = task.length + 1
                p = preds[:, start:]
                t = targets[:, start:]
            else:
                p = preds
                t = targets
                
            # Exact match row-wise
            correct = (p == t).all(dim=1).float().mean().item()
        return correct

    def generate_report(self, results):
        print("\n📊 Generating Report...")
        
        # CSV
        rows = []
        for r in results:
            row = r['config'].copy()
            # remove complex nested objects
            if 'physics_config' in row: del row['physics_config']
            
            row['accuracy'] = r['final_accuracy']
            row['duration'] = r['duration']
            rows.append(row)
            
        df = pd.DataFrame(rows)
        csv_path = self.output_dir / "summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"Summary saved: {csv_path}")
        print(df[['name', 'accuracy', 'duration']].to_string(index=False))
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        palette = sns.color_palette("husl", len(results))
        
        for i, r in enumerate(results):
            plt.plot(r['history']['step'], r['history']['accuracy'], 
                     label=r['config']['name'], linewidth=2, alpha=0.8, color=palette[i])
                     
        plt.title("Manifold Ablation Study", fontsize=16)
        plt.ylabel("Accuracy", fontsize=12)
        plt.xlabel("Step", fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "comparison.png", dpi=300)
        print("Plot saved.")
