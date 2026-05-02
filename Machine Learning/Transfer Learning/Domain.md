## Definition
A **domain** $D$ is composed of two elements $D = \{\mathcal{X}, P(X)\}$, where:
* $\mathcal{X}$ is the **feature space**
	* This is the set of all possible feature vectors, not just what's in the dataset
* $P(X)$ is the [**marginal probability distribution**](Marginal%20Probability%20Distribution.md) over $\mathcal{X}$
* $X = \{x_1, x_2, \dots, x_n\}$ is the **observed instance set** (samples drawn from $P(X)$), where each $x_i \in \mathcal{X}$

---
### Clarifying the Components
1. **Feature Space $\mathcal{X}$**:  
   The *mathematical universe* containing all possible feature representations.  
   *Example:* $\mathbb{R}^{2048}$ for 2048-dimensional CNN embeddings.
2. **Marginal Distribution $P(X)$**:  
   This represents the probability of seeing a specific input $x$ in your feature space. 
   *Example:* Distribution over pet images vs. wild animal images in $\mathbb{R}^{2048}$.
3. **Instance Set $X$**:  
   A finite *collection of samples* drawn i.i.d. from $P(X)$.  
   *Example:* $\{x_{\text{dog}}, x_{\text{cat}}, \dots, x_{\text{dog}}\}$ (n specific image embeddings).