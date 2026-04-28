The **Kullback–Leibler divergence**, also known as relative entropy measures how much an approximating probability distribution $Q$ is different from a true probability distribution $P$. It is defined as:
$$
D_{KL}(P \mid\mid Q) = \sum_{x \in \mathcal{X}}P(x)\log\left(\frac{P(x)}{Q(x)}\right)
$$ Where:
* $D_{KL}(P \mid\mid Q)$ measures how much information is lost when $Q$ is used to approximate $P$
* $P$ is the **true** probability distribution.
* $Q$ is the **approximating** probability distribution.
* $\mathcal{X}$ is the sample space (set of all possible outcomes)
* $P(x)$ is probability that outcome $x$ occurs under the **true** distribution.
* $Q(x)$ is probability that outcome $x$ occurs under the **approximating** distribution.