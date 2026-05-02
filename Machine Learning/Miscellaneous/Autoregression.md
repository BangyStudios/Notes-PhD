An autoregressive (AR) model factorizes a sequence probability in the form:
$$
P(x_1, x_2, ..., x_n) = \prod_{i=1}^n P(x_i \mid x_{<i})
$$Where:
* $P(x_1, x_2, ..., x_n)$ is the probability of the current sequence.
* $P(x_i \mid x_{<i})$ is the probability of the current token with respect to the previous sequence.

In plain language: 
> It predicts each next symbol given all previous ones.

### Lossless Compressive Property
If a model assigns probability $P(x)$ to data $x$, then the optimal number of bits needed to encode it is:
$$
L(x) = -\log_2 P(x)
$$ as per Shannon's source coding theorem.

Then, if an autoregressive model estimates:
$$
P(x) = \prod_i p(x_i \mid x_{<i})
$$ then the ideal code length becomes:
$$
L(x) = -\log_2 p(x_i \mid x_{<i})
$$ 
which is exactly the cross-entropy loss used to train these models.