A typical machine learning (ML) training loop is a repeated sequence of steps where a model learns from data by adjusting its parameters.

---
## 1. Initialize model, parameters, and setup

Allocate the model's weight matrices $W^{(\ell)}$ and bias vectors $b^{(\ell)}$ for each layer $\ell$. Weights are typically drawn from a small random distribution (e.g. Kaiming or Xavier initialization) to break symmetry; biases are usually initialized to zero. Also construct the optimizer (e.g. SGD, Adam) and attach it to the model parameters.
## 2. Loop over epochs
An **epoch** is one full pass over the entire training dataset. Training runs for a fixed number of epochs, or until a stopping criterion (e.g. validation loss plateaus) is met.
## 3. Loop over batches
The dataset is split into **mini-batches** of size $B$. Each iteration of the inner loop processes one batch. Using batches rather than single samples reduces gradient noise relative to stochastic updates while remaining computationally tractable.
## 4. [Forward pass](4-Forward_Pass.md)
Compute the model's prediction $\hat{y} = f(x;\,\theta)$ by passing the batch through each layer in sequence, storing intermediate activations for backpropagation.
## 5. [Compute loss](5-Loss_Function.md)
Evaluate a scalar loss $\mathcal{L}(\hat{y}, y)$ measuring how far the predictions are from the true targets (e.g. cross-entropy for classification, MSE for regression).
## 6. [Backward pass](6-Backpropagation.md)
Apply the chain rule backwards through the computation graph to compute $\nabla_\theta \mathcal{L}$ — the gradient of the loss with respect to every parameter.
## 7. [Update parameters](7-Optimization_Step.md)
Adjust each parameter in the direction that reduces the loss using an optimizer rule (e.g. $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$ for SGD, or the adaptive Adam update).
## 8. [Zero gradients](8-Zero_Gradients.md)
Reset gradient buffers to zero before the next batch. PyTorch accumulates gradients by addition, so without this step, gradients from previous batches corrupt the next update.
