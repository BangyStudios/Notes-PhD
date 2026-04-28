A diffusion model consists of 2 processes:
1. **Forward Process (Diffusion/Noising):** Gradually adds Gaussian noise to real data over many steps until it becomes pure noise.
2. **Reverse Process (Denoising / Generation):** Learns to reverse that noise step-by-step to generate new realistic data from random noise.
### Forward Diffusion Process
The forward process adds Gaussian noise gradually:
$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t ; \sqrt{1 - \beta_t} \, x_{t-1}, \beta_t I)
$$ where:
- $t \in \{1, ..., T\}$ = time step
- $T$ = total number of diffusion steps (often 1000)
- $x_t$​ = noisy version of data at step $t$
- $\beta_t \in (0,1)$ = noise variance schedule (small positive number)
- $I$ = identity matrix
- $\mathcal{N}(\mu, \Sigma)$ = Gaussian distribution with mean $\mu$, covariance $\Sigma$

### Reverse Process
We train a neural network to approximate:
$$
p_\theta(x_{t-1} \mid x_t)
$$ where:
* $p_\theta$ = learned reverse distribution
* $\theta$ = neural network parameters

It is modeled as a gaussian:
$$
q(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1} ; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$ where:
* $\mu_\theta(x_t, t)$ = predicted mean
* $\Sigma_\theta(x_t, t)$ = predicted covariance
* Neural network usually predicts **noise** instead of mean