Neural Relational Inference (NRI), most well known from [Kipf2018NRI](Kipf2018NRI.pdf), is an unsupervised model that simultaneously infers the hidden interaction structure (a graph) between objects and learns their dynamics, purely from observational trajectories. It is structured as a [Variational Autoencoder](Variational%20Autoencoder.md) whose latent code is an interaction graph, using [Graph Neural Networks](Graph%20Neural%20Network.md) for both inference and prediction.

---
## Problem Setting

Given $N$ objects observed over $T$ timesteps, the full trajectory is:
$$
\mathbf{x} = \{x_i^t \mid i = 1,\ldots,N,\; t = 1,\ldots,T\}
$$
where $x_i^t \in \mathbb{R}^d$ is the feature vector (e.g. position, velocity) of object $i$ at time $t$.

The goal is to infer **which objects interact** (the latent graph $\mathbf{z}$) and **how they evolve** given those interactions — without any labels.

---
## Model

NRI factorizes the problem as a VAE:

$$
\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\!\left[\log p_\theta(\mathbf{x}\mid\mathbf{z})\right]}_{\text{reconstruction}} - \underbrace{D_{\mathrm{KL}}\!\left(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\right)}_{\text{regularization}}
$$

- **Encoder** $q_\phi(\mathbf{z}\mid\mathbf{x})$: infers a categorical distribution over edge types for every pair of objects.
- **Decoder** $p_\theta(\mathbf{x}\mid\mathbf{z})$: predicts future states given the inferred interaction graph.
- **Prior** $p(\mathbf{z})$: uniform categorical (no edge is preferred a priori).
- [KL Divergence](KL%20Divergence.md) penalizes the inferred graph from deviating unnecessarily from the prior.

---
## Encoder

The encoder runs two rounds of message passing on a **fully-connected** graph to produce edge-type logits from the trajectory.

### Round 1 — Node → Edge
Concatenate the features of each node pair and pass through an MLP $f_e^{(1)}$:
$$
\mathbf{h}_{(i,j)}^{(1)} = f_e^{(1)}\!\left([\mathbf{x}_i,\, \mathbf{x}_j]\right)
$$

### Aggregation — Edge → Node
Sum incoming edge messages at each node and pass through an MLP $f_v$:
$$
\mathbf{h}_i^{(1)} = f_v\!\left(\sum_{j \neq i} \mathbf{h}_{(j,i)}^{(1)}\right)
$$

### Round 2 — Node → Edge (logits)
Produce the final edge embedding and apply a softmax to obtain edge-type probabilities:
$$
\tilde{\mathbf{z}}_{(i,j)} = f_e^{(2)}\!\left([\mathbf{h}_i^{(1)},\, \mathbf{h}_j^{(1)}]\right)
$$
$$
q_\phi(\mathbf{z}_{ij} = k \mid \mathbf{x}) = \mathrm{softmax}_k\!\left(\tilde{\mathbf{z}}_{(i,j)}\right)
$$

Because $\mathbf{z}$ is **discrete**, the Gumbel-Softmax trick is used to draw differentiable samples during training.

---
## Decoder

Given sampled edge types $\mathbf{z}$, the decoder predicts the next state of each node by passing messages **only along active edges**:

$$
\mu_{(i,j)}^t = f_{e,k}^{\text{dec}}\!\left([x_i^t,\, x_j^t,\, x_i^{t-1},\, x_j^{t-1}]\right) \quad \text{for edge type } k = \mathbf{z}_{ij}
$$

$$
\hat{x}_i^{t+1} = x_i^t + f_v^{\text{dec}}\!\left(\sum_{j\neq i} \mu_{(i,j)}^t\right)
$$

The reconstruction term in the ELBO is then computed as the log-likelihood of the true future states under a Gaussian centred on $\hat{x}_i^{t+1}$.

---
## Example

Consider $N = 3$ particles with $K = 2$ edge types: **spring** (interacting) and **none** (independent).

After encoding the observed trajectories:

| Edge | $q(\mathbf{z}=\text{spring})$ | $q(\mathbf{z}=\text{none})$ |
|------|------|------|
| 1→2 | 0.92 | 0.08 |
| 1→3 | 0.11 | 0.89 |
| 2→3 | 0.87 | 0.13 |

The encoder has correctly identified that particles 1↔2 and 2↔3 interact (springs), while 1↔3 do not.

During decoding, messages are only sent along high-probability spring edges, so particle 3 receives a strong update from particle 2 but almost nothing from particle 1 — matching the true dynamics.

---
## Key Properties

| Property | Detail |
|---|---|
| **Unsupervised** | No ground-truth graph needed |
| **Discrete latent** | Edge types are categorical; Gumbel-Softmax enables gradients |
| **Permutation invariant** | Sum aggregation in message passing is order-independent |
| **Interpretable** | Learned edge types correspond to interaction types |
