# special thanks 2 https://arxiv.org/abs/2505.13416
import math
import torch
import torch.distributed as dist
from itertools import chain
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

# --- Imports from the vendored dion library ---
# These are assumed to be in the same directory or a submodule.
from .muon import Muon
from .newton_schulz_triton import newton_schulz_triton
from .opt_utils import (
    AsyncTask,
    AsyncRuntime,
    to_local,
    create_param_batches,
    pad_batch,
)
from .scalar_opts import lion_update_foreach, adamw_update_foreach

# --- Gluon-specific implementations ---

class Gluon(Muon):
    """
    Distributed Gluon optimizer, extending the Muon implementation.

    This optimizer replaces the heuristic learning rate adjustment of Muon with
    the theoretically-grounded adaptive stepsize calculation from the paper
    "Gluon: Making Muon & Scion Great Again!" (arXiv:2505.13416).

    It requires per-parameter-group L0 and L1 smoothness constants to be provided
    at initialization.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. 
            For Gluon, used *only* for scaling weight decay.
            For Muon, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        mu: Momentum factor for Muon algorithm.
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor.
        epsilon: Small value to avoid division by zero.
        nesterov: Whether to use Nesterov momentum.
        adjust_lr: How to adjust the learning rate for Muon updates ("spectral_norm" or "rms_norm" or None).
            "spectral_norm": Adjust based on spectral norm, for learning rate transfer across model scale.
            "rms_norm": Adjust based on RMS norm, for learning rate compatibility with Adam/AdamW.
            None: Do not adjust the learning rate.
        flatten: Whether to flatten 3D+ tensors to 2D for Muon updates.
            True: Tensors with 3+ dimensions are flattened to 2D. Use this for convolutional layers.
            False: Tensors are not flattened. 3D+ tensors are treated as batches of 2D matrices.
        use_triton: Whether to use Triton kernel for Newton-Schulz. Ignored if custom function is provided.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.
            Signature is `func(input: Tensor, epsilon: float) -> Tensor`.

    Muon optimizer algorithm by Keller Jordan: https://kellerjordan.github.io/posts/muon/
    FSDP2 Muon uses all-to-all communications: https://www.essential.ai/blog/infra
    Distributed Muon implementation by way of samsja: https://github.com/samsja/dion
        (by way of microsoft/dion, ad infinitum)
    """
    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 1.0,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        flatten: bool = False,
        use_triton: bool = False,
        newton_schulz_func: Optional[Callable] = None,
    ):
        # Validate that user has provided L0/L1 for all Gluon groups *before* init
        # This requires converting params from an iterator to a list if needed
        param_groups = list(params)
        for group in param_groups:
            if group.get("algorithm", "gluon") == "gluon":
                if "l0" not in group or "l1" not in group:
                    raise ValueError(
                        "Parameter groups with algorithm='gluon' must include 'l0' and 'l1' keys."
                    )
        # Call the parent __init__ only ONCE with the fully formed groups and defaults.
        # The `adjust_lr` parameter is intentionally not passed, so it uses its default (None)
        # inside the Muon init, which is fine as we never use it.
        # Default arguments for each param group
        defaults = dict(
            lr=lr,
            mu=mu,
            betas=betas,    #gee thanks gemini
            weight_decay=weight_decay,
            algorithm="gluon",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            flatten=flatten,
            #adjust_lr=adjust_lr,
            #dont initialize adjust_lr... 
            # non-passing defaults to None in the superclass...
        )
        super().__init__(params, **defaults)

        # clonnet linted this tab depth
        # We can also set the default algorithm type more explicitly here
        for group in self.param_groups:
            group.setdefault("algorithm", "gluon")

    def _create_muon_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "gluon",
    ) -> Generator["AsyncTask", None, None]:
        """
        (OVERRIDE) Helper function to create batches of Gluon matrices and
        generate AsyncTask objects so we can process multiple batches concurrently.
        """
        for group in param_groups:
            if group["algorithm"] != algo_name:
                continue

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            # Hyperparameters for this specific group
            mu = torch.tensor(group["mu"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            nesterov = group["nesterov"]
            flatten = group["flatten"]
            # [GLUON] Get L0, L1, and the base LR for weight decay
            l0_val = group["l0"]
            l1_val = group["l1"]
            base_lr_for_wd = torch.tensor(group["lr"])

            # Create batches of parameters of size self._world_size
            for params in create_param_batches(group_params, batch_size=self._world_size):
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, algo_name) for p in params]
                momentums = [s["momentum"] for s in states]
                
                # Copy sharding logic directly from Muon implementation
                sharded_mesh_dim, sharded_tensor_dim = None, None
                if isinstance(params[0], DTensor):
                    if not isinstance(self._distributed_mesh, DeviceMesh):
                        raise RuntimeError(
                            "Must create optimizer with DeviceMesh if using DTensor parameters."
                        )

                    # Find the sharded placement and get its mesh and tensor dimensions
                    # Skip any Shard() placements on size-1 mesh dimension = Replicate()
                    shard_placements = [
                        (i, p)
                        for i, p in enumerate(params[0].placements)
                        if p.is_shard() and params[0].device_mesh.size(i) > 1
                    ]
                    if len(shard_placements) == 1:
                        sharded_mesh_dim = shard_placements[0][0]
                        sharded_tensor_dim = shard_placements[0][1].dim
                    elif len(shard_placements) > 1:
                        raise NotImplementedError(
                            "Muon does not support parameters with multiple sharded dimensions."
                        )

                    # Check that the sharded mesh dimension matches optimizer's device mesh
                    if (
                        sharded_mesh_dim is not None
                        and params[0].device_mesh.get_group(sharded_mesh_dim)
                        != self._process_group
                    ):
                        raise RuntimeError(
                            f"Got DTensor sharded over mesh dimension {sharded_mesh_dim} different from the optimizer's device mesh"
                        )

                # Yield an AsyncTask that calls our new Gluon update function
                yield AsyncTask(
                    gluon_update_batch_async(
                        X=pad_batch(params, self._world_size),
                        G=pad_batch(gradients, self._world_size),
                        M=pad_batch(momentums, self._world_size),
                        l0=l0_val,
                        l1=l1_val,
                        base_lr_for_wd=base_lr_for_wd,
                        momentum=mu,
                        weight_decay=weight_decay,
                        epsilon=epsilon,
                        nesterov=nesterov,
                        flatten=flatten,
                        device_rank=self._device_rank,
                        world_size=self._world_size,
                        shard_dim=sharded_tensor_dim,
                        process_group=self._process_group,
                        newton_schulz_func=self._newton_schulz_func,
                    )
                )

# We reuse these functions directly from the muon.py implementation
from .muon import (
    muon_update_pre_orthogonalize,
    muon_update_newton_schulz
)


def gluon_update_batch_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    l0: float,  # Single value, not a list
    l1: float,
    base_lr_for_wd: Tensor,
    momentum: Tensor,
    weight_decay: Tensor,
    epsilon: Tensor,
    nesterov: bool,
    flatten: bool,
    device_rank: int,
    world_size: int,
    shard_dim: Optional[int] = None,
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
) -> Generator[None, None, None]:
    """
    Batched version of the Gluon update. This function orchestrates the
    computation, communication, and final weight update.
    """

    assert len(X) == len(G)
    assert len(X) == len(M)
    assert len(X) == world_size

    # Create tensors once here
    #this part is changed to transdfuce the function operands to the next function
    L0s_tensor = torch.tensor(l0, device=X[0].device)
    L1s_tensor = torch.tensor(l1, device=X[0].device)

    # Update momentum and compute the inputs for orthogonalization
    U = muon_update_pre_orthogonalize(
        G=to_local(G),
        M=to_local(M),
        momentum=momentum,
        nesterov=nesterov,
    )

    # Get one whole matrix for each device to orthogonalize
    if shard_dim is not None:
        # Use all-to-all to transform from a batch of shards to a single whole matrix
        # https://www.essential.ai/blog/infra
        assert (
            process_group is not None
        ), "process_group must be provided for sharded DTensors"
        assert isinstance(X[0], DTensor), "X should contain DTensors"
        assert not isinstance(U[0], DTensor), "U should contain local shards"
        assert (
            X[0].size(shard_dim) % world_size == 0
        ), f"Shard dimension {shard_dim} size {X[0].size(shard_dim)} is not divisible by world size {world_size}."

        # Allocate buffers to receive shards of one whole matrix from other devices
        single_matrix_shards = [torch.empty_like(u) for u in U]

        # Redistribute the shards to form one unique full tensor on each device
        work = dist.all_to_all(
            single_matrix_shards, U, group=process_group, async_op=True
        )
        yield
        work.wait()

        # Concatentate shards to form a whole matrix to orthogonalize
        single_matrix = torch.cat(single_matrix_shards, dim=shard_dim)
        single_matrix = muon_update_newton_schulz(
            single_matrix,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        # Split result back into shards
        # Contiguous is needed for all-to-all to work correctly
        single_matrix_shards = [
            x.contiguous()
            for x in torch.tensor_split(single_matrix, world_size, dim=shard_dim)
        ]

        # Redistribute the orthogonalized tensor back to original layout
        work = dist.all_to_all(
            U, single_matrix_shards, group=process_group, async_op=True
        )
        yield
        work.wait()

    else:
        # Matrices are not sharded, so we can directly orthogonalize
        # Get a single matrix corresponding to this device
        single_matrix = U[device_rank]
        assert not isinstance(single_matrix, DTensor)

        single_matrix = muon_update_newton_schulz(
            single_matrix,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        if process_group is not None and process_group.size() > 1:
            # Allocate empty tensors to receive updates from other devices
            U = [torch.empty_like(u) for u in U]

            # All gather orthogonalized results from other devices into buffer
            work = dist.all_gather(
                U, single_matrix.contiguous(), group=process_group, async_op=True
            )
            yield
            work.wait()

        else:
            # Single GPU case, no need to gather
            assert world_size == 1
            U = [single_matrix]

    # [GLUON]: Calculate adaptive layer-wise stepsize t_i for each param in the batch.
    # This is the core difference between Gluon and Muon.
    T = []
    local_G = to_local(G)
    for i in range(world_size):
        grad_norm = torch.linalg.norm(local_G[i])
        # Add epsilon to denominator for stability when grad_norm is very small
        t_i = grad_norm / (L0s_tensor + L1s_tensor * grad_norm + epsilon)
        T.append(t_i)

    # Step 4: [GLUON] Update model parameters using the adaptive stepsizes.
    gluon_update_post_orthogonalize(
        X=to_local(X),
        U=U,
        T=T,
        base_lr_for_wd=base_lr_for_wd,
        weight_decay=weight_decay,
    )

def gluon_update_post_orthogonalize(
    X: List[Tensor],
    U: List[Tensor],
    T: List[Tensor],
    base_lr_for_wd: Tensor,
    weight_decay: Tensor,
):
    """
    (GLUON VERSION) Applies weight decay and the final weight update after the LMO step.

    This version accepts a list of adaptive stepsizes 'T' instead of a single 'adjusted_lr',
    allowing each parameter in the batch to have its own theoretically-grounded stepsize.

    Args:
        X: List of local parameter tensors (modified in place).
        U: List of normalized momentum tensors from the LMO step.
        T: List of computed adaptive stepsizes (t_i) for each parameter.
        base_lr_for_wd: The base learning rate, used *only* for scaling weight decay.
        weight_decay: The weight decay factor.
    """
    # Apply weight decay, scaled by the provided base learning rate.
    # This decouples the regularization strength from the adaptive stepsize.
    if weight_decay > 0:
        torch._foreach_mul_(X, 1 - base_lr_for_wd * weight_decay)

    # Perform the final weight update using the per-parameter adaptive stepsize t_i.
    # U is multiplied element-wise by T before being subtracted from X.
    U = torch._foreach_mul(U, T)
    torch._foreach_sub_(X, U)