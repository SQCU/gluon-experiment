# gluon_utils.py
import csv
import yaml
import argparse
from pathlib import Path
from collections import deque
from typing import Dict, List, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW
from scipy.optimize import minimize
import numpy as np
from typing import Optional

# --- Component 1: The Adam Hooking Profiler ---

class GluonProfiler:
    """
    Wraps a standard optimizer (like AdamW) to collect the necessary statistics
    for fitting Gluon's (L0, L1)-smoothness constants.

    It works by decorating the optimizer's `step()` method. At each step, it
    records the gradient norm, the difference in gradients between steps, and
    the magnitude of the parameter update.

    Args:
        optimizer: The PyTorch optimizer instance to wrap (e.g., AdamW).
        log_dir: The directory to save statistics CSV files.
        flush_interval: How many steps to buffer in memory before writing to disk.
    """
    def __init__(self, optimizer: Optimizer, log_dir: str = "./gluon_stats", flush_interval: int = 1000):
        self._optimizer = optimizer
        self.log_dir = Path(log_dir)
        self.flush_interval = flush_interval

        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._param_states: Dict[int, Dict[str, Any]] = {}
        self._buffers: Dict[str, deque] = {}

    def _initialize_state_for_param(self, p: nn.Parameter, group_name: str):
        """Lazy initialization of state and log files for a parameter."""
        if id(p) not in self._param_states:
            self._param_states[id(p)] = {
                'prev_grad': torch.zeros_like(p.grad),
            }
            if group_name not in self._buffers:
                self._buffers[group_name] = deque()
                # Write header to new CSV file
                log_path = self.log_dir / f"{group_name}.csv"
                with open(log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["grad_diff_norm", "param_update_norm", "next_grad_norm"])

    def step(self, closure=None):
        """
        Performs a single optimization step and records profiling statistics.
        This method should be called in place of optimizer.step().
        """
        # 1. Store pre-update parameter values (X_k) for each group
        params_before_step = {}
        for i, group in enumerate(self._optimizer.param_groups):
            group_name = group.get('name', f'group_{i}')
            params_before_step[group_name] = [p.clone().detach() for p in group['params']]

        # 2. Execute the actual optimizer step (updates parameters in-place to X_k+1)
        loss = self._optimizer.step(closure)

        # 3. Collect and buffer statistics
        for i, group in enumerate(self._optimizer.param_groups):
            group_name = group.get('name', f'group_{i}')
            
            for p_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                self._initialize_state_for_param(p, group_name)
                state = self._param_states[id(p)]
                
                # Retrieve X_k
                p_before = params_before_step[group_name][p_idx]

                # Calculate norms
                param_update_norm = torch.linalg.norm(p.detach() - p_before).item()
                grad_diff_norm = torch.linalg.norm(p.grad.detach() - state['prev_grad']).item()
                next_grad_norm = torch.linalg.norm(p.grad.detach()).item()

                # Avoid division by zero in analysis later
                if param_update_norm > 1e-8:
                    self._buffers[group_name].append(
                        (grad_diff_norm, param_update_norm, next_grad_norm)
                    )

                # Update state for next step
                state['prev_grad'] = p.grad.clone().detach()

        # 4. Flush buffers to disk if they are full
        for name, buf in self._buffers.items():
            if len(buf) >= self.flush_interval:
                self.flush(name)
        
        return loss

    def flush(self, group_name: Optional[str] = None):
        """Writes buffered statistics to their corresponding CSV files."""
        names_to_flush = [group_name] if group_name else self._buffers.keys()
        for name in names_to_flush:
            buf = self._buffers.get(name)
            if not buf:
                continue
            
            log_path = self.log_dir / f"{name}.csv"
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                while buf:
                    writer.writerow(buf.popleft())
        print(f"[GluonProfiler] Flushed statistics for groups: {list(names_to_flush)}")


# --- Component 2: The gluanalyze() Fitting Script ---

def _fit_l0_l1(data: np.ndarray, lambda_penalty: float = 1.0):
    """The core fitting logic for a single parameter group."""
    grad_diff_norms, param_update_norms, next_grad_norms = data.T
    
    # L_hat is the "true" measured smoothness at each step
    L_hat = grad_diff_norms / (param_update_norms + 1e-10)

    def objective_function(params):
        l0, l1 = params
        # L_approx is the model's prediction of smoothness
        L_approx = l0 + l1 * next_grad_norms
        
        error = L_hat - L_approx
        mse = np.mean(error**2)
        
        # Hinge-like penalty for underestimation (from paper's Appendix E.2)
        underestimation_penalty = np.mean(np.maximum(0, error)**2)
        
        return mse + lambda_penalty * underestimation_penalty

    # Initial guess and bounds (L0 and L1 must be non-negative)
    initial_guess = [0.0, 1.0]
    bounds = [(0, None), (0, None)]
    
    result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    return result.x[0], result.x[1] # l0, l1

def gluanalyze(log_dir: str, lambda_penalty: float = 1.0) -> Dict[str, Any]:
    """
    Analyzes profiling logs to fit L0/L1 constants for each parameter group.

    Args:
        log_dir: The directory containing the statistics CSV files.
        lambda_penalty: Weight for the underestimation penalty in fitting.

    Returns:
        A dictionary containing the fitted hyperparameters, ready to be saved as a config.
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_path}")
        
    config = {
        "hyperparameters": {
            "default": {},
            "overrides": {}
        }
    }
    
    print(f"[gluanalyze] Analyzing logs in {log_path}...")
    for csv_file in log_path.glob("*.csv"):
        group_name = csv_file.stem
        print(f"  - Processing group: {group_name}")
        
        try:
            data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
            if data.shape[0] < 10: # Need some data to fit
                print(f"    - Skipping {group_name}, not enough data points.")
                continue
            
            l0, l1 = _fit_l0_l1(data, lambda_penalty)
            print(f"    - Fitted L0={l0:.4f}, L1={l1:.4f}")
            
            # For simplicity, we put everything in overrides. Can be refined later.
            config["hyperparameters"]["overrides"][group_name] = {"l0": l0, "l1": l1}

        except Exception as e:
            print(f"    - Failed to process {group_name}: {e}")
            
    return config


# --- Component 3: Dumping and Loading Config Handlers ---

def save_gluon_config(config: Dict[str, Any], path: str):
    """Saves a Gluon configuration dictionary to a YAML file."""
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    header = (
        "# GlazyGloptimizer configuration generated by gluanalyze\n"
        "# This file contains the fitted (L0, L1)-smoothness constants for your model.\n"
    )
    
    with open(config_path, 'w') as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"[GluonConfig] Saved configuration to {config_path}")


def load_gluon_config(path: str) -> Dict[str, Any]:
    """Loads a Gluon configuration from a YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"[GluonConfig] Loaded configuration from {config_path}")
    return config

# --- Component 4: The `gluon_that_model` Helper Function ---

def gluon_that_model(model: nn.Module, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Constructs the parameter group list needed by the `gluon.Gluon` optimizer
    based on a loaded configuration file.

    Args:
        model: The torch.nn.Module to be optimized.
        config: The loaded Gluon configuration dictionary.

    Returns:
        A list of parameter group dicts for the optimizer.
    """
    hyperparams = config["hyperparameters"]
    overrides = hyperparams.get("overrides", {})
    
    # Create a mapping from override keys to parameter lists
    grouped_params: Dict[str, List[nn.Parameter]] = {key: [] for key in overrides}
    default_params: List[nn.Parameter] = []
    
    param_to_group_map = {}

    # First, assign each parameter to a group based on its name
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        assigned_group = None
        for group_name in overrides:
            if group_name in name: # Simple substring matching
                assigned_group = group_name
                break # Assign to the first matching group
        
        if assigned_group:
            grouped_params[assigned_group].append(p)
            param_to_group_map[id(p)] = assigned_group
        else:
            default_params.append(p)
            param_to_group_map[id(p)] = "default"

    # Now, build the final list for the optimizer
    optimizer_param_groups = []
    
    # Add the override groups
    for name, params in grouped_params.items():
        if not params: continue
        group_config = {
            "params": params,
            "name": name, # Store name for the profiler
            **hyperparams.get("default", {}), # Start with defaults
            **overrides[name] # Apply overrides
        }
        optimizer_param_groups.append(group_config)
        
    # Add the default group
    if default_params:
        optimizer_param_groups.append({
            "params": default_params,
            "name": "default",
            **hyperparams.get("default", {})
        })

    print("[gluon_that_model] Created parameter groups from config.")
    return optimizer_param_groups

# --- CLI for Analysis ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gluon Optimizer Utility Script")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    analyze_parser = subparsers.add_parser("analyze", help="Analyze profiler logs to generate a Gluon config.")
    analyze_parser.add_argument("--log_dir", type=str, required=True, help="Directory containing the profiler CSV logs.")
    analyze_parser.add_argument("--output_path", type=str, default="gluon_config.yaml", help="Path to save the generated YAML config.")
    analyze_parser.add_argument("--lambda_penalty", type=float, default=1.0, help="Weight for the underestimation penalty during fitting.")

    args = parser.parse_args()

    if args.command == "analyze":
        config = gluanalyze(args.log_dir, args.lambda_penalty)
        save_gluon_config(config, args.output_path)