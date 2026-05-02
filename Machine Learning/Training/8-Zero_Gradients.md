Zeroing gradients resets the accumulated gradient buffers to zero before computing new gradients for the next mini-batch. In PyTorch, this is done with `optimizer.zero_grad()`.

---
## Definition
### Why Gradients Accumulate
PyTorch accumulates gradients by **adding** each `.backward()` call into existing `.grad` buffers:
$$
g_{\text{buffer}} \;\leftarrow\; g_{\text{buffer}} + \nabla_\theta \mathcal{L}
$$

This is intentional — it allows gradient accumulation across multiple forward passes to simulate a larger effective batch size. However, for standard mini-batch training, gradients from the previous step must be explicitly zeroed before computing new ones.
### Discrete Formulation
At the start of each mini-batch step $t$, **before** the forward pass:
$$
g_t \leftarrow 0
$$
After `.backward()`, the buffer holds exactly $\nabla_\theta \mathcal{L}_t$ — the gradient of the current batch's loss only.
### What Happens Without `zero_grad()`
If `zero_grad()` is skipped, the effective gradient used in the [optimization step](7-Optimization_Step.md) becomes:
$$
g_{\text{effective}} = \sum_{\tau=1}^{t} \nabla_\theta \mathcal{L}_\tau
$$
an unintended accumulation over all previous batches, producing incorrect and increasingly large parameter updates.
### Implementation Notes
- **`optimizer.zero_grad()`:** Resets `.grad` of all tensors tracked by the optimizer.
- **`model.zero_grad()`:** Resets `.grad` of all parameters in the model. Equivalent for standard training loops.
- **`set_to_none=True`** (PyTorch default since 1.7): Sets gradients to `None` rather than writing zeros. Saves memory allocation and skips the zero-fill; PyTorch treats `None` as "no gradient yet" and handles it correctly.

---
## Example
### Setup
Suppose a single scalar parameter $w = 1.0$ is trained with learning rate $\eta = 0.1$. Each batch produces gradient $\nabla \mathcal{L} = 0.5$.

---
### Without `zero_grad()` (incorrect)

| Batch | `.backward()` accumulates | `.grad` value | Update ($\eta = 0.1$) | $w$ after |
|-------|--------------------------|---------------|------------------------|-----------|
| 1     | $0 + 0.5$                | $0.5$         | $w \leftarrow 1.0 - 0.05$ | $0.95$ |
| 2     | $0.5 + 0.5$              | $1.0$         | $w \leftarrow 0.95 - 0.10$ | $0.85$ |
| 3     | $1.0 + 0.5$              | $1.5$         | $w \leftarrow 0.85 - 0.15$ | $0.70$ |

Batch 2 uses a doubled gradient because Batch 1's gradient was never cleared. The effective step size grows with each batch.

---
### With `zero_grad()` (correct)

| Batch | `zero_grad()` | `.backward()` accumulates | `.grad` value | Update ($\eta = 0.1$) | $w$ after |
|-------|--------------|--------------------------|---------------|------------------------|-----------|
| 1     | $0$          | $0 + 0.5$                | $0.5$         | $w \leftarrow 1.0 - 0.05$ | $0.95$ |
| 2     | $0$          | $0 + 0.5$                | $0.5$         | $w \leftarrow 0.95 - 0.05$ | $0.90$ |
| 3     | $0$          | $0 + 0.5$                | $0.5$         | $w \leftarrow 0.90 - 0.05$ | $0.85$ |

Each batch contributes exactly its own gradient.

---
### Intentional Gradient Accumulation
To simulate a batch of size $k \cdot B$ using batches of size $B$, call `.backward()` $k$ times before calling `zero_grad()`:
$$
g_{\text{accumulated}} = \sum_{j=1}^{k} \nabla_\theta \mathcal{L}_{B_j} \;\approx\; \nabla_\theta \mathcal{L}_{kB}
$$
This is useful when a large batch does not fit in memory. The optimizer step is taken once after the $k$-th backward pass, then `zero_grad()` is called.

---
### Interpretation
- `zero_grad()` must be called **before** `loss.backward()`, not after — so each backward pass starts from a clean state.
- Missing `zero_grad()` is a silent bug: the model still trains, but the gradient signal grows unboundedly across batches, leading to overshooting and instability.
- This step completes one full iteration of the training loop. The loop then returns to performing a [forward pass](4-Forward_Pass.md) for the next batch.
