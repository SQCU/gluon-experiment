# swappomatic_scalar_opts.py

import torch
from torch import Tensor
from typing import List

# @torch.compile(fullgraph=True) # Compilation can be added after verification
def swappomatic_adamw_update_foreach(
    X: List[Tensor],         # Model weights (modified in place)
    G: List[Tensor],         # Gradient
    M: List[Tensor],         # Momentum buffer (modified in place)
    V: List[Tensor],         # Variance buffer (modified in place)
    U_N: Tensor,             # Update norm buffer (scalar tensor, modified in place)
    lr: Tensor,              # Learning rate (scalar tensor)
    beta1: Tensor,           # Beta 1 (scalar tensor)
    beta2: Tensor,           # Beta 2 (scalar tensor)
    weight_decay: Tensor,    # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
):
    """
    Swappomatic-aware AdamW optimizer algorithm (foreach implementation).

    In addition to the standard AdamW update, this function performs two telemetry side-effects:
    1. It calculates the L2 norm of the policy-driven update vector (excluding weight decay)
   and writes it in-place to the `U_N` buffer.

        Args:
        ...
        U_N (Tensor): A pre-allocated scalar tensor buffer. The calculated norm of
            the update vector will be written here in-place.
        ...
    """
    batch_size = len(X)
    assert batch_size == len(G)
    assert batch_size == len(M)
    assert batch_size == len(V)
    assert U_N.numel() == 1, "Update norm buffer U_N must be a scalar tensor."

    M_dtype = M[0].dtype
    V_dtype = V[0].dtype

    # Update momentum and variance (same as original)
    G = [g.to(dtype=M_dtype) for g in G]
    torch._foreach_lerp_(M, G, [1 - beta1] * batch_size)
    G_square = torch._foreach_mul(G, G)
    G_square = [g.to(dtype=V_dtype) for g in G_square]
    torch._foreach_lerp_(V, G_square, [1 - beta2] * batch_size)

    # Bias correction (same as original)
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()

    # Compute the denominator for the weight update (same as original)
    denom = torch._foreach_sqrt(V)
    torch._foreach_div_(denom, bias_correction2_sqrt)
    torch._foreach_add_(denom, [epsilon] * batch_size)

    # Adjust learning rate to include bias correction 1 (same as original)
    adj_lr = lr / bias_correction1
    
    # --- SWAPPOMATIC MODIFICATION START ---

    # 1. Calculate the final update vectors *before* applying weight decay
    #    This represents the pure, pre-regularization gradient-based step.
    #    update_vectors = adj_lr * M / denom
    update_vectors = torch._foreach_div(M, denom)
    torch._foreach_mul_(update_vectors, adj_lr)

    # 2. Compute the L2 norm of these vectors and write it to the buffer.
    #    torch.linalg.vector_norm is the most efficient way to do this.
    norm_val = torch.linalg.vector_norm(update_vectors)
    U_N.fill_(norm_val) # In-place write to the output buffer.

    # --- SWAPPOMATIC MODIFICATION END ---

    # Apply weight decay (same as original)
    torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Final weight update, using the pre-calculated update_vectors
    # X = X - update_vectors
    torch._foreach_sub_(X, update_vectors)


# @torch.compile(fullgraph=True)
def swappomatic_gluon_update_post_orthogonalize(
    X: List[Tensor],         # Model weights (modified in place)
    U: List[Tensor],         # Orthogonalized momentum
    T: List[Tensor],         # Adaptive stepsizes
    U_N: Tensor,             # Update norm buffer (scalar tensor, modified in place)
    lr: Tensor,              # Base learning rate
    weight_decay: Tensor,    # Weight decay factor
):
    """
    Swappomatic-aware final Gluon update step.

    Performs the standard post-orthogonalization update and writes the L2 norm
    of the update vector to the U_N buffer.
    
    Args:
        ...
        U_N (Tensor): A pre-allocated scalar tensor buffer. The calculated norm of
            the update vector will be written here in-place.
        ...
    """

    # 1. Calculate the final update vectors.
    #    update_vectors = lr * T * U
    T_scaled = torch._foreach_mul(T, lr)
    U = torch._foreach_mul(U, T_scaled)
    # --- SWAPPOMATIC MODIFICATION START ---
    # 2. Compute the L2 norm and write it to the buffer.
    norm_val = torch.linalg.vector_norm(U)
    U_N.fill_(norm_val) # In-place write to the output buffer.
    # --- SWAPPOMATIC MODIFICATION END ---
    
    # Apply weight decay, scaled by the provided base learning rate.
    # This decouples the regularization strength from the adaptive stepsize.
    if weight_decay > 0:
        torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Final weight update, using the pre-calculated update_vectors
    # X = X - update_vectors
    torch._foreach_sub_(X, U)

def swappomatic_adamw_update_foreach_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    V: List[Tensor],  # Variance buffer (modified in place)
    U_N: Tensor,      # Update norm buffer (scalar tensor, modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
) -> Generator[None, None, None]:
    """
    Async wrapper for the instrumented AdamW update.
    This is what the overridden _create_adamw_tasks will yield.
    """
    swappomatic_adamw_update_foreach(X, G, M, V, U_N, lr, beta1, beta2, weight_decay, step, epsilon)
    yield # Mimics the async pattern of the parent library

def swappomatic_gluon_update_batch_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    U_N: Tensor,
    l0: float,  # Single value, not a list
    l1: float,
    lr: Tensor,
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
    swappomatic_gluon_update_post_orthogonalize(
        X=to_local(X),
        U=U,
        T=T,
        U_N=U_N,
        lr=lr,
        weight_decay=weight_decay,
    )