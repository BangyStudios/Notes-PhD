### Autoregressive LLMs
> [!PDF|yellow] [Nie2025LLMDiffusion, p.1](Papers/2025/Nie2025LLMDiffusion.pdf#page=1&selection=202,0,203,71&color=yellow)
> > The predominant approach relies on the [autoregressive modeling (ARM)](../Statistics/Autoregression.md)—commonly referred to as the “next-token prediction” paradigm—to define the model distribution:

$$
p_\theta(x) = p_\theta(x^1) \prod_{i=2}^L p_\theta(x^i \mid x^1, ..., x^{i-1})
$$where:
* $x$ is a sequence of length $L$
* $x^i$ is the $i$-th token

#### Limitations of Autoregressive LLMs
> [!PDF|yellow] [Nie2025LLMDiffusion, p.2](Papers/2025/Nie2025LLMDiffusion.pdf#page=2&selection=256,14,257,5&color=yellow)
> > the left-to-right generation process restricts their ability to handle reversal reasoning tasks


