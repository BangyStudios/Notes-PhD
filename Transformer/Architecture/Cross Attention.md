Cross attention is the mechanism by which the decoder attends to the encoder's output, allowing it to access source-sequence information when generating target tokens. Unlike [Self-Attention](Self-Attention.md), where queries, keys, and values all come from the same sequence, cross attention draws **Q from the decoder's current hidden state** and **K, V from the encoder output**.

---
## Definition
Consider the standard [Self-Attention](Self-Attention.md) formula:
$$
\text{Attention}(Q, K, V) = \text{softmax}_\text{row}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
In cross attention, let $X_1 \in \mathbb{R}^{n \times d}$ be the decoder's current state and $E \in \mathbb{R}^{m \times d}$ be the encoder output. The projections are:
$$
Q = X_1 W_Q, \qquad K = E\, W_K, \qquad V = E\, W_V
$$
So the cross-attention output is:
$$
\text{CrossAttention}(X_1, E) = \text{softmax}_\text{row}\!\left(\frac{(X_1 W_Q)(E W_K)^T}{\sqrt{d_k}}\right)(E W_V)
$$

Key difference from self-attention: **Q and K,V come from different sequences**, so each target token (row of $X_1$) can attend to all source tokens (rows of $E$).

---
## Example
### Setup
Suppose:
- **Encoder output** $E$ has 3 source tokens ("The", "cat", "sat"), $d=2$:
$$
E = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}
$$
- **Decoder state** $X_1$ has 2 target tokens, $d=2$:
$$
X_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$
- Use identity projections: $W_Q = W_K = W_V = I$, so $Q = X_1$, $K = V = E$.

---
### Step 1: Compute raw attention scores

$$
QK^T = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}
$$

Each entry $(i,j)$ is the dot product between target token $i$ and source token $j$.

---
### Step 2: Scale by $\sqrt{d_k}$

$$
\frac{QK^T}{\sqrt{2}} \approx \begin{bmatrix} 0.71 & 0.00 & 0.71 \\ 0.00 & 0.71 & 0.71 \end{bmatrix}
$$

---
### Step 3: Apply softmax row-wise

- Row 1: $[0.71, 0.00, 0.71]$ → softmax ≈ $[0.40, 0.20, 0.40]$
- Row 2: $[0.00, 0.71, 0.71]$ → softmax ≈ $[0.20, 0.40, 0.40]$

$$
A \approx \begin{bmatrix} 0.40 & 0.20 & 0.40 \\ 0.20 & 0.40 & 0.40 \end{bmatrix}
$$

---
### Step 4: Multiply by V

$$
\text{CrossAttention}(X_1, E) = A \cdot V =
\begin{bmatrix} 0.40 & 0.20 & 0.40 \\ 0.20 & 0.40 & 0.40 \end{bmatrix}
\begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}
= \begin{bmatrix} 0.80 & 0.60 \\ 0.60 & 0.80 \end{bmatrix}
$$

---
### Interpretation
- Target token 1 (row 1) focuses equally on source tokens 0 and 2 (weight 0.40 each) and less on source token 1 (weight 0.20).
- Target token 2 (row 2) focuses equally on source tokens 1 and 2 (weight 0.40 each) and less on source token 0 (weight 0.20).
- Each target token produces a weighted combination of the **encoder's value vectors**, allowing the decoder to selectively read from the full source context.
