    
# glazy_gloptimizer.py
import torch
import atexit
import logging
from pathlib import Path
from torch.optim import Optimizer
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union, Any
import hashlib

# --- Imports from our own library ---
from .gluon import Gluon
from .gluon_utils import (
    GluonProfiler,
    gluanalyze,
    save_gluon_config,
    load_gluon_config,
    gluon_that_model,
)

class GlazyGloptimizer(Optimizer):
    """
    An automated, "just works" optimizer that implements the two-phase workflow
    for using the Gluon optimizer.

    On the first run with a new model, it operates in **Profiling Mode**:
    1. It uses a standard AdamW optimizer as a proxy.
    2. It wraps AdamW with the GluonProfiler to collect smoothness statistics.
    3. When the training script exits, it automatically analyzes the stats,
       fits the L0/L1 constants, and saves a `gluon_config.yaml`.

    On all subsequent runs, it operates in **Gluon Mode**:
    1. It detects the saved config file.
    2. It loads the L0/L1 constants.
    3. It instantiates and runs the actual, high-performance Gluon optimizer,
       fully tuned for the specific model architecture.

    Args:
        model (torch.nn.Module): The model to be optimized. This is required
            to create named parameter groups.
        log_dir (str, optional): Directory to store logs, stats, and the config.
            Defaults to "./glazy_gluon".
        **kwargs: Hyperparameters for the AdamW optimizer used during the
            profiling phase (e.g., `lr`, `weight_decay`, `betas`).
    """
    def __init__(self, params, log_dir: str = "./glazy_gluon", **kwargs):
        """
        Final, corrected __init__ signature. Conforms to the standard
        torch.optim.Optimizer API: __init__(self, params, **defaults).

        Args:
            params: The standard iterable of parameters or parameter groups.
            log_dir (str, optional): Directory to store logs and configs.
            **kwargs: Hyperparameters for the AdamW profiler and base optimizer.
        """
        # The `params` argument can be an iterator, so we convert it to a list
        # to ensure we can inspect it and pass it on.
        param_groups = list(params)

        # We must call the base Optimizer's __init__ first.
        # It handles parsing the parameter groups and setting up self.param_groups.
        # We pass the user's kwargs as the defaults.
        super().__init__(param_groups, kwargs)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_logging()

        # Generate a unique path for the config based on the parameters
        model_hash = self._get_model_hash()
        self.config_path = self.log_dir / f"config_{model_hash}.yaml"

        self.is_profiling = not self.config_path.exists()
        self._internal_optimizer: Optimizer

        if self.is_profiling:
            self._initialize_profiling_mode()
        else:
            self._initialize_gluon_mode()
        
        # We need to sync our param_groups with the internal optimizer's.
        # This is a bit of a trick to make this wrapper work seamlessly.
        self.param_groups = self._internal_optimizer.param_groups


    def _get_model_hash(self) -> str:
        """
        Creates a simple but STABLE and DETERMINISTIC hash based on the shapes
        and names of the parameters being optimized. This is robust to the
        order in which parameter groups are passed and is deterministic
        across Python processes.
        """
        param_signatures = []
        for group in self.param_groups:
            group_name = group.get('name', 'default')
            for i, p in enumerate(group['params']):
                shape_str = "_".join(str(s) for s in p.shape)
                param_sig = f"{group_name}_{i}_{shape_str}"
                param_signatures.append(param_sig)
        
        param_signatures.sort()
        
        final_signature = "|".join(param_signatures)
        
        # --- THE CRITICAL FIX ---
        # Use hashlib.sha256 for a deterministic hash instead of the
        # built-in, non-deterministic hash().

        # 1. Encode the string to bytes, as hashing functions operate on bytes.
        signature_bytes = final_signature.encode('utf-8')
        
        # 2. Create a sha256 hash object and update it with the bytes.
        sha256_hash = hashlib.sha256(signature_bytes)
        
        # 3. Get the hexadecimal representation of the hash.
        #    We can truncate it to a reasonable length for the filename.
        deterministic_hash = sha256_hash.hexdigest()
        
        return deterministic_hash[:16] # Return the first 16 characters for a clean filename

    def _setup_logging(self):
        """Sets up a logger that writes detailed info to a file."""
        self.logger = logging.getLogger("GlazyGloptimizer")
        self.logger.setLevel(logging.INFO)
        # Prevent logs from propagating to the root logger
        self.logger.propagate = False
        
        # Avoid adding handlers if they already exist (e.g., in a notebook)
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_dir / "glazy_log.txt")
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _initialize_profiling_mode(self):
        """Sets up the optimizer for its first-run profiling phase."""
        print("="*60)
        print("✨ GlazyGloptimizer: Config not found. Starting in Profiling Mode.")
        print(f"   - Statistics will be saved to: {self.log_dir / 'stats'}")
        print(f"   - A detailed log will be written to: {self.log_dir / 'glazy_log.txt'}")
        print("="*60)
        self.logger.info("Initializing in Profiling Mode.")

        # Ensure param groups have names for logging.
        for i, group in enumerate(self.param_groups):
            group.setdefault('name', f'group_{i}')

        # We need to extract the AdamW-specific kwargs from our defaults
        adamw_kwargs = {
            k: v for k, v in self.defaults.items() 
            if k in ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad']
        }
        
        adam_optimizer = torch.optim.AdamW(self.param_groups, **adamw_kwargs)
        
        self.profiler_stats_dir = self.log_dir / "stats"
        self._internal_optimizer = GluonProfiler(adam_optimizer, log_dir=str(self.profiler_stats_dir))

        atexit.register(self._on_exit_profiling)

    def _on_exit_profiling(self):
        """This function is called automatically when the profiling script ends."""
        print("\n" + "="*60)
        print("✨ GlazyGloptimizer: Training script finished.")
        print("   - Flushing final statistics and running analysis...")
        print("="*60)
        self.logger.info("Script exit detected. Starting post-run analysis.")

        # Ensure all buffered data is written to disk
        self._internal_optimizer.flush()

        try:
            config = gluanalyze(log_dir=str(self.profiler_stats_dir))
            save_gluon_config(config, path=str(self.config_path))
            self.logger.info(f"Successfully generated and saved config to {self.config_path}")
            print(f"✅ Analysis complete. Config saved to: {self.config_path}")
            print("   Please re-run your training script to automatically use the tuned Gluon optimizer.")
        except Exception as e:
            self.logger.error(f"Failed to generate config during analysis: {e}")
            print(f"❌ Error during analysis: {e}. Config not saved. Please check 'glazy_log.txt'.")

    def _initialize_gluon_mode(self):
        """Sets up the optimizer for its second-run, high-performance phase."""
        print("="*60)
        print(f"✨ GlazyGloptimizer: Config found at {self.config_path}.")
        print("   Initializing the tuned Gluon optimizer.")
        print("="*60)
        self.logger.info("Initializing in Gluon Mode.")
        
        config = load_gluon_config(path=str(self.config_path))
        
        # Add default fallback algorithm for parameters not covered in config
        config.setdefault("hyperparameters", {}).setdefault("default", {})
        config["hyperparameters"]["default"].setdefault("algorithm", "adamw")
        config["hyperparameters"]["default"].setdefault("lr", 1e-4) # Sensible default for AdamW

        gluon_param_groups = self._gluonify_param_groups(config)

        self._internal_optimizer = Gluon(gluon_param_groups)

    def _gluonify_param_groups(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        A new helper function that applies a loaded config directly to the
        existing parameter groups, without needing the full model object.
        """
        hyperparams = config.get("hyperparameters", {})
        overrides = hyperparams.get("overrides", {})
        default_hparams = hyperparams.get("default", {})

        new_param_groups = []
        for group in self.param_groups:
            group_name = group.get('name', 'default')
            
            # Find the right config for this group
            group_config = overrides.get(group_name, default_hparams)
            
            if "l0" in group_config and "l1" in group_config:
                group['algorithm'] = 'gluon'
                group['l0'] = group_config['l0']
                group['l1'] = group_config['l1']
            else:
                group['algorithm'] = 'adamw'
            
            new_param_groups.append(group)
        
        print("[GlazyGloptimizer] Applied Gluon config to parameter groups.")
        return new_param_groups

    # --- Pass-through methods to the internal optimizer ---

    def zero_grad(self, set_to_none: bool = True):
        self._internal_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        return self._internal_optimizer.step(closure)

    def __repr__(self):
        mode = "Profiling" if self.is_profiling else "Gluon"
        return f"GlazyGloptimizer(mode={mode}, internal_optimizer={self._internal_optimizer.__class__.__name__})"

  