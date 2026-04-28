A Relational Graph Convolution Network (R-GCN) is a type of [GNN](GNN.md) designed to operate on multi-relational graphs, such as [knowledge graphs](KG.md) ([Paper](../../Papers/2017/Schlichtkrull2017RGCN.pdf)).

## Update Rule
$$
h_i^{(l+1)} = \sigma\left[\left(\sum_{r\in\mathcal{R}} \sum_{j\in\mathcal{N}_i^r} \frac{1}{c_{i, r}} W_r^{(l)}h_j^{(l)}\right) + W_0^{(l)}h_i^{(l)} \right]
$$ Where:
* $h_i^{(l)} \in \mathbb{R}^{d^{(l)}}$ is the embedding of node $i$ at layer $l$
* $W_r^{(l)}$ is the weight matrix for relation $r$
* $\mathcal{N}_i^r$ are the set of neighbors of node $i$ connected with relation $r$
* $c_{i,r}$ is a normalization constant (that can be learned or chosen in advance, usually $c_{i,r} = |\mathcal{N}_i^r|$)
* $W_0$ is the self-loop weight
* $\sigma$ is an activation function (i.e. ReLU)
### Example of a R-GCN Update
Suppose node **France** has neighbors:
$(\text{Paris}, \text{capitalOf}, \text{France}), (\text{Napoleon}, \text{bornIn}, \text{France})$

Initialize embeddings:
$$
\text{Paris} = \begin{bmatrix}
0.2 & 0.5
\end{bmatrix},
\quad
\text{Napoleon} = \begin{bmatrix}
0.8 & 0.1
\end{bmatrix},
\quad
\text{France} = \begin{bmatrix}
0.3 & 0.7
\end{bmatrix}
$$
Initialize edge weights:
$$
W_\text{capitalOf} = \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix},
\quad
W_\text{bornIn} = \begin{bmatrix}
0.5 & 0 \\
0 & 0.5
\end{bmatrix},
\quad
W_0 = \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$
Compute messages from neighbors:
$$
W_\text{capitalOf} \cdot \text{Paris} = \begin{bmatrix} 0.2 & 0.5 \end{bmatrix}
\quad
W_\text{bornIn} \cdot \text{Napoleon} = \begin{bmatrix} 0.4 & 0.05 \end{bmatrix}
\quad
W_0 \cdot \text{France} = \begin{bmatrix} 0.3 & 0.7 \end{bmatrix}
$$
Aggregate:
$$
W_\text{capitalOf} \cdot \text{Paris} + 
W_\text{bornIn} \cdot \text{Napoleon} +
W_0 \cdot \text{France} = 
\begin{bmatrix} 0.9 & 1.25 \end{bmatrix}
$$
Apply activation, in the above case, the result remains the same.
## Regularization
To address the issue of overfitting on highly multi-relational data (too many parameters), we apply **[basis](../Linear%20Algebra/Basis.md) decomposition** and **block-diagonal decomposition**. 

Basis decomposition is satisfied by the identity:
$$
W_r^{(l)} = \sum_{b=1}^Ba_{rb}^{(l)}V_b^{(l)}
$$ where:
* $W_r^{(l)}$ is the weight matrix for relation $r$ (from original update rule)
* $a_{rb}^{(l)}$ are coefficients (such that only they depend on $r$)
* $V_b^{(l)}$ is a linear combination of basis transformations $V_b^{(l)} \in \mathbb{R}^{d^{(l+1)} \times d}$

[Block-diagonal decomposition](../Linear%20Algebra/Block-Diagonal%20Decomposition.md) is achieved by letting each $W_r^{(l)}$ (again, from original update rule) be defined through the direct sum over a set of low-dimensional matrices:
$$
W_r^{(l)} = \bigoplus_{b=1}^{B}Q_{br}^{(l)}
$$ where:
* $Q_{br}^{(l)} \in \mathbb{R}^{(d^{l+1}/B) \times (d^{(l)}/B)}$

Note: Values in the non-block regions of $W_r^{(l)}$ are truncated (set to 0) as a form of "dropout".
## Typical Pipeline
1. Text
2. Information extraction
3. Triples (subject, relation, object)
4. [Knowledge graph](KG.md)
5. R-GCN learning (this page)
6. Knowledge completion / reasoning