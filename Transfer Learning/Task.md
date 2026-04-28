## Definition
A **task** $T$ is composed of two elements $T = \{\mathcal{Y}, P(Y|X)\}$, also expressable as $T = \{\mathcal{Y}, f\}$, where:
* $\mathcal{Y}$ is the **label space**
	* This is the set of all possible output vectors, not just what's is current being outputted
* $P(Y|X)$ is the **conditional distribution** of labels given features
* $Y = \{y_1, y_2, \dots, y_n\}$ is the **observed label set** (samples drawn according to $P(Y|X=x_i)$), where each $y_i \in \mathcal{Y}$

---
### Clarifying the Components:
1. **Label Space $\mathcal{Y}$**:  
   The *mathematical universe* containing all possible target values.  
   *Example:* $\{0, 1\}$ for binary classification, $\mathbb{R}$ for regression, or $\{1, 2, \dots, C\}$ for C-class classification.
2. **Conditional Distribution $P(Y|X)$**:  
   The *functional relationship* or *decision rule* that maps features to label probabilities.  
   *Example:* The probability that an image embedding corresponds to "cat" vs. "dog", or the mapping from house features to price distributions.
3. **Label Set $Y$**:  
   A finite *collection of observed labels* drawn according to $P(Y|X)$ given the instance set.  
   *Example:* $\{y_{\text{dog}}, y_{\text{cat}}, \dots, y_{\text{dog}}\}$ (n specific class labels).