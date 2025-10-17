# swappomatic_gloptimizer.py
import torch
from torch.optim import Optimizer
from torch import Tensor
from typing import List, Dict, Any

from .gluon import Gluon # Subclassing our direct parent
from .swappomatic_scalar_opts import swappomatic_adamw_update_foreach, swappomatic_adamw_update_foreach_async, swappomatic_gluon_post_orthogonalize, swappomatic_gluon_update_batch_async

class SwappomaticGloptimizer(Gluon):
    def __init__(self, params, **defaults):
        """
        Initializes the Swappomatic Gloptimizer.

        As a subclass of Gluon, this optimizer starts with AdamW as the default "Actor"
        and learns a better, Gluon-based "Understudy" policy in the background for each
        parameter group. It autonomously promotes the Understudy based on a distributed
        consensus protocol that measures policy superiority.
        ...
        During initialization, this method iterates through the parameter groups and injects
        the necessary state for the Swappomatic protocol. For each group, it adds:
        - `algorithm`: Set to 'adamw' by default (the primordial Actor).
        - `cfct_l0`, `cfct_l1`: The Understudy's trainable meta-model parameters.
        - `projection_seed`: A deterministic integer seed for the on-the-fly gradient sketch.
        - `grad_sketch_prev`, `grad_norm_sq_prev`: The O(d) sketch history buffers.
        - `telemetry_buffer_UN`, `telemetry_buffer_MDN`: The O(1) in-place telemetry buffers.
        - `last_actor_norm`: A float caching the previous step's actor update norm.
        - `ema_R`, `var_R_state`: Buffers for the behavioral statistics.
        - `local_cost`: A float for the group's contribution to the swap decision.
        ...
        """
        super().__init__(params, **defaults)
        for group in self.param_groups:
            # --- PERSISTENT GROUP STATE ---
            # Stored here, owned by the group.
            device = group['params'][0].device
            sketch_dim = self.defaults['sketch_dim']
            
            group['projection_seed'] = torch.randint(0, 2**32, (1,)).item()
            group['grad_sketch_prev'] = torch.zeros(sketch_dim, device=device)
            group['grad_norm_sq_prev'] = torch.zeros(1, device=device)
            group['last_actor_norm'] = 0.0 # Will store the scalar norm from step k-1
            group['U_N'] = torch.zeros(1, device=device)
            # ... (ema_R, var_R_state, etc.) ...

    @torch.no_grad()
    def step(self, closure=None):
        """
        ...
        This method implements an 'off policy'/'online statistics' optimizer switching scheduler, orchestrated in five blocks:
        1. control flow logic for yeeting out of date optimizers with hysteretic or covariate shifted parameters
        2. A gradient norm extraction + low rank compression of live `p.grad` tensors to
        pass along a representation useable to approximate the norm of gradient differences across steps.
        this is miserably complicated in one or two ways but replaces an adam-sized momentum cache with ~128*1 parameters per parameter *group*
        i.e. one 512x512 dim linear projection matrix -> 128x1 dim bottleneck -> proxy calculation for ||diff(old 512x512 gradient, new 512x512 gradient)||
        ergo ~ 1/2048 compression ratio (of gradient diff norm related information).
        3. A call to `super().step()`, which does more ordinary sorts of stuff to call modified async task generators,
        execute the 'Actor' optimizer's update and mutate update norm buffers.
        4. A **post-step evaluation phase** that estimates the counterfactual benefit of swapping optimizer 'policies' for the step, 
        then computing a cost function to implement a swap propensity control hyperparameter.
        5. A sparse, non-blocking initiation/polling of the distributed cost function check.
        ...
        """
        # Block 1: Check and Execute Deferred Swap
        if self.swap_is_pending:
            self._perform_actor_promotion()
            self.swap_is_pending = False
            self.cost_accumulator.fill_(0.0)

        # --- BLOCK 2: PRE-STEP OBSERVATION & LEARNING (ZERO-COPY) ---
        # This block operates on the live p.grad tensors (G[k]).
        # It learns from the past (k-1) and prepares for the future (k+1).
        hypothetical_norms = self._observe_and_learn_from_gradients()

        # --- BLOCK 3: EXECUTE ACTOR'S STEP ---
        # The superclass consumes p.grad and populates the telemetry buffers.
        loss = super().step(closure)

        # --- BLOCK 4: POST-STEP EVALUATION & COST UPDATE ---
        # This block uses the results from Block 2 and Block 3 to evaluate policies.
        self._evaluate_policies_and_update_cost(hypothetical_norms)

        # Block 5: Asynchronous Consensus Check (same as before)
        self._poll_or_initiate_consensus_check()

        return loss

    # --- METHOD OVERRIDES FOR NORM MEASUREMENT INJECTION ---
    # we basically need all 3 of these defined to reuse super.step().
    # they are needed because blah blah function signature of passing that update_norm buffer for in-place mutation.
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
                if group["algorithm"] not in ["muon", "gluon"]:
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
                # [SWAPPOMATIC]
                U_N = group["U_N"]
                # [GLUON] Get L0, L1, and the base LR for weight decay
                l0_val = group["l0"]
                l1_val = group["l1"]
                lr = torch.tensor(group["lr"])

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
                        swappomatic_gluon_update_batch_async(
                            X=pad_batch(params, self._world_size),
                            G=pad_batch(gradients, self._world_size),
                            M=pad_batch(momentums, self._world_size),
                            U_N=U_N,
                            l0=l0_val,
                            l1=l1_val,
                            lr=lr,
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
                    
    def _create_adamw_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "adamw",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for AdamW updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            # [SWAPPOMATIC]
            U_N = torch.tensor(group["U_N"])

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            step = torch.tensor(group["step"])

            yield AsyncTask(
                swappomatic_adamw_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    U_N=U_N,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    step=step,
                    epsilon=epsilon,
                )
            )

    #gemini is lazy and doesn't want to implement lion. neither do i!
    def _create_lion_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "lion",
    ) -> Generator["AsyncTask", None, None]:
        print(f"the lion does not concern himself with adaptive optimization in the distributed setting")
        pass

    def _perform_actor_promotion(self) -> None:
        """
        Executes the swap of the Actor and Understudy.

        This function promotes the current Understudy to become the new Actor. It involves:
        1. Transferring the Understudy's optimizer state (parameters, state_dict) to the Actor.
        2. Freezing the new Actor's (L0, L1) policy.
        3. Re-initializing a new Understudy, typically cloning the state of the newly
           promoted Actor as a starting point for its learning.
        4. Logging the swap event for later analysis.
        
        Returns:
            None. Modifies the self.actor and self.understudy attributes.
        """
        # ... implementation ...
        pass
        
    def _poll_or_initiate_consensus_check(self) -> None:
        """
        Manages the sparse, non-blocking communication for global consensus.

        On a sparse schedule (e.g., every 100 steps), this function will:
        1. If a handle to a previous async `all_reduce` exists, poll it. If it's complete,
           check the result and set the `self.swap_is_pending` flag if the global cost
           threshold is met.
        2. Initiate a *new* non-blocking `all_reduce` on the current local cost
           accumulator, storing the returned handle for the next check.

        This ensures the main training loop is never blocked by network I/O.
        
        Returns:
            None. Modifies `self.swap_check_handle` and `self.swap_is_pending`.
        """
        # ... implementation ...
        pass

    def _calculate_hypothetical_update_norm(self, group: Dict[str, Any], grad_cache: Dict[int, Tensor]) -> Tensor:
        """
        Calculates the L2 norm of the update the Understudy *would have* made.

        This is a lightweight, local calculation that uses the group's counterfactual
        (cfct) L0/L1 constants and the cached gradients to compute the theoretical
        Gluon update magnitude without applying it or running the full async pipeline.

        Args:
            group (Dict): A single parameter group from self.param_groups.
            grad_cache (Dict): A map from param ID to the gradient tensor for that param
                at the start of the step.

        Returns:
            Tensor: A scalar tensor containing the total L2 norm of the hypothetical
                update vector for this group.
        """
        pass

    def _observe_and_learn_from_gradients(self) -> Dict[int, Tensor]:
        """
        Performs the pre-step, zero-copy observation and meta-learning phase.

        This function is called *before* the main optimizer step. It operates directly
        on the live `p.grad` tensors (representing G[k]) to learn from the transition
        between step k-1 and k. For each parameter group, it:
        1. Calls the gradient sketch protocol (`_sketch_and_reconstruct_norm_from_seed`)
        to get the approximate gradient difference norm `‖G[k]-G[k-1]‖'`.
        2. Updates the group's persistent sketch history buffers for the next iteration.
        3. Calculates the smoothness proxy `L-hat` for the *previous* step (`k-1`) by
        normalizing the reconstructed norm by `‖ΔX[k-1]‖`.
        4. Trains the Understudy's meta-model using this newly computed `L-hat`.
        5. Calculates the hypothetical update norm for the Understudy's policy for the
        *current* step (`k`) and returns it for later evaluation.

        Returns:
            Dict[int, Tensor]: A dictionary mapping each group's index to the scalar tensor
                representing its hypothetical Understudy update norm for the current step.
                This is a transient output to be consumed later in the step.
        """
        hypothetical_norms = {}
        for i, group in enumerate(self.param_groups):
            current_grads = [p.grad for p in group['params'] if p.grad is not None]
            if not current_grads:
                continue

            # Step 2 & 3: Reconstruct norm and get new sketch state for the next step
            reconstructed_norm, new_sketch, new_norm_sq = self._sketch_and_reconstruct_norm_from_seed(
                current_gradients=current_grads,
                projection_seed=group['projection_seed'],
                sketch_dim=self.defaults['sketch_dim'],
                previous_sketch=group['grad_sketch_prev'],
                previous_norm_sq=group['grad_norm_sq_prev']
            )
            group['grad_sketch_prev'] = new_sketch
            group['grad_norm_sq_prev'] = new_norm_sq

            # Step 4: Calculate smoothness from previous step's data
            l_hat_actual = reconstructed_norm / (group['last_actor_norm'] + 1e-8)

            # Step 5: Train the meta-model
            grad_norm_k = torch.linalg.vector_norm(current_grads)
            self._train_understudy_meta_model(group, l_hat_actual, grad_norm_k)

            # Step 6: Calculate this step's hypothetical norm and store it transiently
            hypothetical_norms[i] = self._calculate_hypothetical_update_norm(group, grad_norm_k)

        return hypothetical_norms

    def _evaluate_policies_and_update_cost(self, hypothetical_norms: Dict[int, Tensor]) -> None:
        """
        Performs the post-step policy evaluation and updates the swap cost.

        This function is called *after* the Actor's step has been performed. It evaluates
        the policy disagreement for the transition from step `k` to `k+1`. It:
        1. Retrieves the Actor's update norm for step `k` (`‖ΔX[k]‖`) from the
        group's telemetry buffer, which was populated during `super().step()`.
        2. Compares it to the Understudy's hypothetical norm for step `k`, which was
        calculated pre-step in `_observe_and_learn_from_gradients`.
        ...
        5. Caches the Actor's norm (`‖ΔX[k]‖`) into `group['last_actor_norm']` so it can be
        used in the *next* iteration's smoothness calculation.

        Args:
            hypothetical_norms (Dict[int, Tensor]): The transient dictionary of Understudy
                update norms calculated in the pre-step phase.

        Returns:
            None. Modifies group statistics and the cost accumulator in-place.
        """
        total_local_pressure = 0.0
        num_groups_evaluated = 0

        for i, group in enumerate(self.param_groups):
            if i not in hypothetical_norms:
                continue
            
            # Step 1: Get Actor's norm for the step that just happened
            actor_norm = group['U_N'].item()
            
            # Step 2: Get Understudy's norm
            understudy_norm = hypothetical_norms[i].item()

            # Step 3: Update behavioral stats
            R = understudy_norm / (actor_norm + 1e-8)
            # ... call _update_ema and _update_online_variance for this group ...

            # Step 4: Calculate Swap Pressure
            pressure = self._calculate_swap_pressure(group)
            total_local_pressure += pressure
            num_groups_evaluated += 1

            # Step 5: Cache actor norm for the next step's L-hat calculation
            group['last_actor_norm'] = actor_norm

        # Aggregate local pressure into the cost accumulator
        if num_groups_evaluated > 0:
            avg_pressure = total_local_pressure / num_groups_evaluated
            # ... update self.cost_accumulator using avg_pressure ...

    def _train_understudy_meta_model(
        self, 
        group: Dict[str, Any], 
        l_hat_actual: Tensor, 
        grad_norm: Tensor
    ) -> None:
        """
        Performs a single gradient descent step on the Understudy's (L0, L1) meta-model.

        This function treats the group's 'cfct_l0' and 'cfct_l1' as trainable parameters.
        It uses a hardened SGD approach to ensure stability when learning from a noisy
        signal. The process involves:
        1. Calculating the gradient of the fitting loss (MSE with underestimation penalty).
        2. **Clipping the gradients** to a reasonable range to prevent explosive updates.
        3. Applying the update using a small, fixed `meta_lr`.
        4. Clamping the resulting `cfct_l0` and `cfct_l1` to be non-negative.

        Returns:
            None. Modifies the group dictionary in-place.
        """
        # ... implementation:
        # 1. Get meta_lr, penalty_lambda from self.defaults.
        # 2. l_hat_approx = group['cfct_l0'] + group['cfct_l1'] * grad_norm.
        # 3. error = l_hat_actual - l_hat_approx.
        # 4. grad_l0 = -2 * (error + penalty_lambda * max(0, error)).
        # 5. grad_l1 = grad_l0 * grad_norm.
        # 6. group['cfct_l0'] -= meta_lr * grad_l0.
        # 7. group['cfct_l1'] -= meta_lr * grad_l1.
        # 8. Clamp cfct_l0 and cfct_l1 to be non-negative.
        pass

    def _calculate_swap_pressure(self, group: Dict[str, Any]) -> float:
        """
        Calculates the instantaneous "Swap Pressure" for a single parameter group.

        The pressure is a scalar value that is non-zero only if both the "bias" and
        "confidence" conditions for a swap are met. It quantifies the strength of the
        evidence that the Understudy's policy is superior to the Actor's.

        Args:
            group (Dict): The parameter group being evaluated. Must contain the current
                'ema_R' and 'var_R_state' statistics.

        Returns:
            float: The unitless Swap Pressure value (P_i).
        """
        # ... implementation:
        # 1. Get threshold_bias, threshold_confidence from self.defaults.
        # 2. Get ema_R and the current variance from group['ema_R'] and group['var_R_state'].
        # 3. bias_term = max(0, abs(ema_R - 1) / threshold_bias - 1).
        # 4. confidence_term = max(0, threshold_confidence / current_variance - 1).
        # 5. return bias_term * confidence_term.
        pass

    # In swappomatic_gloptimizer.py, inside the SwappomaticGloptimizer class

    def _update_ema(self, old_ema: float, new_value: float, beta: float) -> float:
        """
        Computes a single step of an Exponential Moving Average.

        Args:
            old_ema (float): The EMA value from the previous step.
            new_value (float): The new data point to incorporate.
            beta (float): The smoothing factor, typically between 0.9 and 0.999.

        Returns:
            float: The updated EMA value.
        """
        return beta * old_ema + (1 - beta) * new_value

    def _update_online_variance(
        self, 
        existing_aggregate: tuple, 
        new_value: float
    ) -> tuple:
        """
        Updates running variance statistics using Welford's online algorithm.

        This avoids catastrophic cancellation and is numerically stable.

        Args:
            existing_aggregate (tuple): A tuple of (count, mean, M2), where M2 is the
                sum of squares of differences from the current mean.
            new_value (float): The new data point to incorporate.

        Returns:
            tuple: The updated (count, mean, M2) aggregate. The variance can be
                calculated as M2 / count.
        """
        (count, mean, M2) = existing_aggregate
        count += 1
        delta = new_value - mean
        mean += delta / count
        delta2 = new_value - mean
        M2 += delta * delta2
        return (count, mean, M2)

        # In swappomatic_gloptimizer.py, inside the SwappomaticGloptimizer class

def _sketch_and_reconstruct_norm_from_seed(
    current_gradients: List[Tensor],
    projection_seed: int,
    sketch_dim: int,
    previous_sketch: Tensor,
    previous_norm_sq: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Computes a new gradient sketch and reconstructs the norm of the difference.

    This function implements a memory-efficient streaming algorithm. It uses a
    deterministic seed to generate a random projection matrix on-the-fly, uses it for
    computation, and immediately discards it. This avoids storing the large O(d*D)
    matrix, ensuring the persistent memory overhead is only O(d) per group.

    The core of the method is the algebraic expansion:
    ‖G[k] - G[k-1]‖² = ‖G[k]‖² + ‖G[k-1]‖² - 2 * <G[k], G[k-1]>
    where the dot product is approximated via the random projection.

    Args:
        current_gradients (List[Tensor]): G[k] for the parameter group.
        projection_seed (int): A deterministic seed used to regenerate the projection matrix.
        sketch_dim (int): The dimensionality 'd' of the low-rank sketch.
        previous_sketch (Tensor): The sketch of the last step's gradient (S @ G[k-1]).
        previous_norm_sq (Tensor): The squared norm of the last step's gradient (‖G[k-1]‖²).

    Returns:
        tuple[Tensor, Tensor, Tensor]: A tuple containing:
            - The reconstructed norm of the difference (scalar `‖...‖'`).
            - The new sketch for the current gradient (`S @ G[k]`).
            - The new squared norm for the current gradient (`‖G[k]‖²`).
    """
    flat_grad = torch.cat([g.view(-1) for g in current_gradients])
    D = flat_grad.numel()
    d = sketch_dim
    
    # Generate S on-the-fly from the seed
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(projection_seed)
    S = torch.randn(d, D, device=flat_grad.device, dtype=flat_grad.dtype) / math.sqrt(d)
    torch.random.set_rng_state(rng_state)  # Restore original RNG state
    
    # Use it
    current_sketch = torch.mv(S, flat_grad)

    # S is automatically freed when function returns
    del S

    # --- Remainder of the function is the same, but with NO scaling factor ---
    current_norm_sq = torch.dot(flat_current_grad, flat_current_grad)
    
    # Dot product in sketch space - FJLT is constructed to be orthonormal
    dot_product_approx = torch.dot(current_sketch, previous_sketch)

    reconstructed_norm_sq = current_norm_sq + previous_norm_sq - 2 * dot_product_approx
    reconstructed_norm = torch.sqrt(torch.clamp(reconstructed_norm_sq, min=1e-12))

    return reconstructed_norm, current_sketch, current_norm_sq