## Definition
A small category $\textbf{C}$ is a set $\textbf{C} = \{C_{obj}, C_{\rightarrow}\}$ where $C_{obj}$ is the set of objects in $\textbf{C}$ and $C_\rightarrow$ is the set of arrows (morphisms) in $\textbf{C}$ and satisfies the following properties:
* **Uniqueness**: $\forall f, g \in C_\rightarrow, \forall A, B \in C_{obj},  (\text{dom}(f) = A \land \text{codom}(f) = B) \implies \nexists g \in C_\rightarrow : (\text{dom}(g) = A \land \text{codom}(g) = B)$
* **Composition:** $\forall f, g \in C_\rightarrow, (f: A\rightarrow B \land g: B \rightarrow C) \implies \exists h\in C_\rightarrow : (h: A \rightarrow C)$
* **Identity:** $\forall A \in C_{obj}, \exists 1_A \in C_\rightarrow : \text{dom}(1_A) = \text{codom}(1_A) = A$
* **Associativity** $\forall f, g, h \in C_\rightarrow, h(g(f(x))) = (h \circ g)(f(x))$
* **Compositional identity:** $\forall (f: A \rightarrow B) \in \textbf{C}, f(1_A(x)) = f(x) = 1_B(f(x))$