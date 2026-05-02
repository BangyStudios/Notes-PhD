Backpropagation computes the gradient of the loss $\mathcal{L}$ with respect to every parameter $\theta$ by applying the **chain rule** backwards through the computation graph built during the [forward pass](4-Forward_Pass.md).

---
## Definition

### Continuous Formulation (Chain Rule)
For a loss $\mathcal{L}$ that is a composition of layer functions $f^{(1)}, \ldots, f^{(L)}$, the gradient with respect to the parameters $\theta^{(\ell)}$ at layer $\ell$ is:
$$
\frac{\partial \mathcal{L}}{\partial \theta^{(\ell)}} = \frac{\partial \mathcal{L}}{\partial h^{(\ell)}} \cdot \frac{\partial h^{(\ell)}}{\partial \theta^{(\ell)}}
$$
where $\partial \mathcal{L} / \partial h^{(\ell)}$ is itself computed recursively by propagating the gradient back from layer $L$ to layer $\ell + 1$.
### Discrete / Layer-by-Layer Formulation
Given the values $\{z^{(\ell)}, h^{(\ell)}\}$ stored during the forward pass, backpropagation sweeps from $\ell = L$ down to $\ell = 1$.
Define the **error signal** at layer $\ell$:
$$
\delta^{(\ell)} \;=\; \frac{\partial \mathcal{L}}{\partial z^{(\ell)}}
$$
**At the output layer** (softmax + cross-entropy), the error signal has a closed-form simplification:
$$
\delta^{(L)} = \hat{y} - y
$$
**At interior layers**, it is computed by propagating the error signal from the layer above:
$$
\delta^{(\ell)} = \left(W^{(\ell+1)}\right)^{\!T} \delta^{(\ell+1)} \;\odot\; \sigma'^{(\ell)}\!\left(z^{(\ell)}\right)
$$
**Parameter gradients** are then assembled from the error signal and the activations of the previous layer:
$$
\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} = \delta^{(\ell)}\!\left(h^{(\ell-1)}\right)^{\!T}, \qquad \frac{\partial \mathcal{L}}{\partial b^{(\ell)}} = \delta^{(\ell)}
$$

- **$\delta^{(\ell)} \in \mathbb{R}^{d_\ell}$:** Error signal at layer $\ell$ — how much $\mathcal{L}$ changes with respect to the pre-activation $z^{(\ell)}$.
- **$\odot$:** Element-wise (Hadamard) product.
- **$\sigma'^{(\ell)}$:** Derivative of the activation function at layer $\ell$ evaluated at $z^{(\ell)}$. For ReLU: $\sigma'(z) = \mathbf{1}[z > 0]$.
- **$\left(W^{(\ell+1)}\right)^T \delta^{(\ell+1)}$:** Upstream gradient routed back through the weight matrix. The transpose reverses the forward mapping $h^{(\ell)} \to z^{(\ell+1)}$.

---
## Example
### Setup
Continuing from [4-Forward_Pass](4-Forward_Pass.md) and [5-Loss_Function](5-Loss_Function.md):
$$
x = \begin{bmatrix}1\\0\end{bmatrix},\quad
z^{(1)} = \begin{bmatrix}1\\0\end{bmatrix},\quad
h^{(1)} = \begin{bmatrix}1\\0\end{bmatrix},\quad
\hat{y} \approx \begin{bmatrix}0.119\\0.881\end{bmatrix},\quad
y = \begin{bmatrix}0\\1\end{bmatrix}
$$
$$
W^{(1)} = \begin{bmatrix}1&0\\0&1\end{bmatrix}, \qquad W^{(2)} = \begin{bmatrix}1&2\\3&4\end{bmatrix}
$$

---
### Step 1: Output layer error signal
$$
\delta^{(2)} = \hat{y} - y = \begin{bmatrix}0.119\\0.881\end{bmatrix} - \begin{bmatrix}0\\1\end{bmatrix} = \begin{bmatrix}0.119\\-0.119\end{bmatrix}
$$
The symmetric values $\pm 0.119$ arise because softmax output sums to 1, so any increase in one class probability exactly decreases the other.

---
### Step 2: Gradients for $W^{(2)}$ and $b^{(2)}$
$$
\frac{\partial \mathcal{L}}{\partial W^{(2)}} = \delta^{(2)}\!\left(h^{(1)}\right)^T = \begin{bmatrix}0.119\\-0.119\end{bmatrix}\begin{bmatrix}1&0\end{bmatrix} = \begin{bmatrix}0.119&0\\-0.119&0\end{bmatrix}
$$
$$
\frac{\partial \mathcal{L}}{\partial b^{(2)}} = \delta^{(2)} = \begin{bmatrix}0.119\\-0.119\end{bmatrix}
$$
The second column of $\partial \mathcal{L}/\partial W^{(2)}$ is zero because $h^{(1)}_2 = 0$ — those weights have no influence on the output for this input.

---
### Step 3: Propagate error signal to layer 1
Route $\delta^{(2)}$ back through $W^{(2)}$:

$$
\left(W^{(2)}\right)^T \delta^{(2)} = \begin{bmatrix}1&3\\2&4\end{bmatrix}\begin{bmatrix}0.119\\-0.119\end{bmatrix} = \begin{bmatrix}0.119 - 0.357\\0.238 - 0.476\end{bmatrix} = \begin{bmatrix}-0.238\\-0.238\end{bmatrix}
$$
Apply the ReLU derivative, $\sigma'^{(1)}(z) = \mathbf{1}[z > 0]$, at $z^{(1)} = [1,\, 0]^T$:
$$
\sigma'^{(1)}\!\left(z^{(1)}\right) = \begin{bmatrix}\mathbf{1}[1>0]\\\mathbf{1}[0>0]\end{bmatrix} = \begin{bmatrix}1\\0\end{bmatrix}
$$
$$
\delta^{(1)} = \begin{bmatrix}-0.238\\-0.238\end{bmatrix} \odot \begin{bmatrix}1\\0\end{bmatrix} = \begin{bmatrix}-0.238\\0\end{bmatrix}
$$
The second unit receives zero error signal because $z^{(1)}_2 = 0$ is on the ReLU boundary — gradient is zero by convention, so no signal passes through this unit.

---
### Step 4: Gradients for $W^{(1)}$ and $b^{(1)}$
$$
\frac{\partial \mathcal{L}}{\partial W^{(1)}} = \delta^{(1)} x^T = \begin{bmatrix}-0.238\\0\end{bmatrix}\begin{bmatrix}1&0\end{bmatrix} = \begin{bmatrix}-0.238&0\\0&0\end{bmatrix}
$$
$$
\frac{\partial \mathcal{L}}{\partial b^{(1)}} = \delta^{(1)} = \begin{bmatrix}-0.238\\0\end{bmatrix}
$$

---
### Interpretation
- $\partial\mathcal{L}/\partial W^{(2)}_{11} > 0$: increasing this weight raises the class-0 probability, which increases loss — so it should decrease.
- $\partial\mathcal{L}/\partial W^{(2)}_{21} < 0$: increasing this weight raises the class-1 probability, which decreases loss — so it should increase.
- The entire second row of $\partial\mathcal{L}/\partial W^{(1)}$ is zero, and the second column of $\partial\mathcal{L}/\partial W^{(2)}$ is zero. This is a compound dead zone: $h^{(1)}_2 = 0$ (from ReLU at boundary) means these weights contribute nothing forward or backward for this input.
- These gradients are consumed by the [optimization step](7-Optimization_Step.md).
