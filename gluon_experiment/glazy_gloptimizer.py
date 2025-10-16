    
# glazy_gloptimizer.py
import torch
import atexit
import logging
from pathlib import Path
from torch.optim import Optimizer

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
    def __init__(self, model: torch.nn.Module, log_dir: str = "./glazy_gluon", **kwargs):
        self.model = model
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_logging()

        # Generate a unique path for the config based on the model architecture
        model_hash = self._get_model_hash()
        self.config_path = self.log_dir / f"config_{model_hash}.yaml"

        self.is_profiling = not self.config_path.exists()
        self._internal_optimizer: Optimizer

        if self.is_profiling:
            self._initialize_profiling_mode(kwargs)
        else:
            self._initialize_gluon_mode()

        # The base Optimizer class requires a `param_groups` attribute. We'll
        # delegate this to our internal optimizer.
        # This is a bit of a hack to make the wrapper conform to the Optimizer API.
        # The `defaults` dict is also required, but can be empty.
        super().__init__(self._internal_optimizer.param_groups, {})


    def _get_model_hash(self) -> str:
        """Creates a simple hash based on the model's parameter names."""
        # A simple but effective way to identify a model architecture.
        param_names = "_".join(self.model.state_dict().keys())
        return str(abs(hash(param_names)))

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

    def _initialize_profiling_mode(self, adamw_kwargs):
        """Sets up the optimizer for its first-run profiling phase."""
        print("="*60)
        print("✨ GlazyGloptimizer: Config not found. Starting in Profiling Mode.")
        print(f"   - Statistics will be saved to: {self.log_dir / 'stats'}")
        print(f"   - A detailed log will be written to: {self.log_dir / 'glazy_log.txt'}")
        print("="*60)
        self.logger.info("Initializing in Profiling Mode.")

        # Create simple named parameter groups for AdamW
        param_groups = []
        for name, module in self.model.named_modules():
            # Filter out the top-level module and empty modules
            if "." in name and list(module.parameters(recurse=False)):
                param_groups.append({
                    "params": module.parameters(recurse=False),
                    "name": name.replace('.', '_') # Make names file-system friendly
                })
        
        # Add any remaining parameters to a default group
        all_grouped_params = {p for group in param_groups for p in group['params']}
        remaining_params = [p for p in self.model.parameters() if p not in all_grouped_params and p.requires_grad]
        if remaining_params:
            param_groups.append({"params": remaining_params, "name": "default_params"})

        adam_optimizer = torch.optim.AdamW(param_groups, **adamw_kwargs)
        
        # Wrap AdamW with our profiler
        self.profiler_stats_dir = self.log_dir / "stats"
        self._internal_optimizer = GluonProfiler(adam_optimizer, log_dir=str(self.profiler_stats_dir))

        # Register the analysis function to run automatically on clean exit
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

        gluon_param_groups = gluon_that_model(self.model, config)
        
        # Add the 'gluon' algorithm tag where L0/L1 are defined, as this is
        # used by the Gluon optimizer's internal dispatch logic.
        for group in gluon_param_groups:
            if "l0" in group and "l1" in group:
                group["algorithm"] = "gluon"

        self._internal_optimizer = Gluon(gluon_param_groups)

    # --- Pass-through methods to the internal optimizer ---

    def zero_grad(self, set_to_none: bool = True):
        self._internal_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        return self._internal_optimizer.step(closure)

    def __repr__(self):
        mode = "Profiling" if self.is_profiling else "Gluon"
        return f"GlazyGloptimizer(mode={mode}, internal_optimizer={self._internal_optimizer.__class__.__name__})"

  