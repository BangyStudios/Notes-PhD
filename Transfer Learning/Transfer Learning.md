## Definition
*From [Pan2009](../../Papers/Transfer%20Learning/Pan2009.pdf)*:
Given a source [domain](Domain.md) $\mathcal{D}_S$ and learning [task](Task.md) $\mathcal{T}_S$, a target domain $\mathcal{D}_T$ and learning task $\mathcal{T}_T$, transfer learning aims to help improve the learning of the target predictive function $f_T(\cdot) = P_T(Y|X)$ in $\mathcal{D}_T$ using the knowledge in $\mathcal{D}_S$ and $\mathcal{T}_S$, where $\mathcal{D}_S \neq \mathcal{D}_T$ or $\mathcal{T}_S \neq \mathcal{T}_T$ .

$\mathcal{D}_S \neq \mathcal{D}_T$ or $\mathcal{T}_S \neq \mathcal{T}_T$ entails that in each respective $\mathcal{D}$ or $\mathcal{T}$, either:
* $\mathcal{X}_S \neq \mathcal{X}_T$ or $P_S(X) \neq P_T(X)$, or
* $\mathcal{Y}_S \neq \mathcal{Y}_T$ or $P(Y_S|X_S) \neq P(Y_T|X_T)$, respectively.

As such, in the case where domain and tasks are the same, i.e. ($\mathcal{D}_S = \mathcal{D}_T$ and $\mathcal{T}_S = \mathcal{T}_T$), the learning problem reduces to a traditional machine learning problem.

As for in the case of domains, either the feature space (input schema) differs, or marginal distributions (data with the same input schema) differs. Consider the following cases for ($\mathcal{D}_S \neq \mathcal{D}_T$):
* **Case 1** ($\mathcal{X}_S \neq \mathcal{X}_T$): Feature spaces are different (as previously discussed), or
	* Example: Two sets of documents are described in different languages.
* **Case 2** ($P_S(X) \neq P_T(X)$): Marginal distributions are different (as previously discussed).
	* Example: When the source domain and target domain documents focus on different topics.

As for in the case of tasks, either the the label spaces differ, or conditional distributions differs. Consider the following cases:
* **Case 1** ($\mathcal{Y}_S \neq \mathcal{Y}_T$): Label spaces are different, or
	* Example: Source domain has $n$ document classes whereas target domain has $m$ document classes.
* **Case 2** ($P(Y_S|X_S) \neq P(Y_T|X_T)$): Conditional distributions are different
	* Example: Label shift. The same input features point to different labels across domains because the prevalence of those labels has shifted—like a model over-predicting "Urgent" in a casual setting just because urgency was common in its training data.