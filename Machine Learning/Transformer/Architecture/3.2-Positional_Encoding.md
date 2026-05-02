## Definition

For a position index $pos$ and embedding dimension index $i$:
$$
\text{PE}(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$
$$
\text{PE}(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$
- **$pos$:** The position of the token in the sequence.
- **$i$:** The dimension index of the embedding.
- **$d_{\text{model}}$:** The total embedding dimension.
- PE: Positional encoding vector added to token embeddings.

This allows the model to incorporate word order without using recurrence or convolution, by injecting position-dependent patterns into embeddings.

---
## Example
### Setup
Suppose we have 3 tokens: "The", "cat", "sat".  
Assume:
- $d_{\text{model}} = 4$
- Positions: $pos = 0, 1, 2$
### Step 1: Compute Positional Encodings

For $pos = 0$:
$$
\text{PE}(0) = [\sin(0), \cos(0), \sin(0), \cos(0)] = [0, 1, 0, 1]
$$
For $pos = 1$:
$$
\text{PE}(1) = \left[
\sin\!\left(\frac{1}{10000^{0/4}}\right),
\cos\!\left(\frac{1}{10000^{0/4}}\right),
\sin\!\left(\frac{1}{10000^{2/4}}\right),
\cos\!\left(\frac{1}{10000^{2/4}}\right)
\right]
$$
$$
\approx [\sin(1), \cos(1), \sin(0.01), \cos(0.01)]
$$
$$
\approx [0.84, 0.54, 0.01, 0.9999]
$$
### Step 2: Add to Token Embeddings
If the embedding for "The" is:
$$
[0.2, 0.5, 0.1, 0.7]
$$
Then the final input becomes:
$$
[0.2, 0.5, 0.1, 0.7] + [0, 1, 0, 1] = [0.2, 1.5, 0.1, 1.7]
$$