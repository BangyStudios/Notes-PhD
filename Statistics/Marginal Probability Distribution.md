## Definition

The **marginal probability distribution** is the probability distribution of a subset of variables within a larger set of random variables, obtained by summing (or integrating, for continuous variables) over the possible values of the other variables.

Given a joint probability distribution $P(X, Y)$ for two discrete random variables:
- The marginal distribution of $X$ is:  
  $$
  P(X = x) = \sum_{y} P(X = x, Y = y)
  $$
- The marginal distribution of $Y$ is:  
  $$
  P(Y = y) = \sum_{x} P(X = x, Y = y)
  $$

For continuous variables with joint density $f_{X,Y}(x,y)$:
$$
f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x,y) \, dy
$$
$$
f_Y(y) = \int_{-\infty}^{\infty} f_{X,Y}(x,y) \, dx
$$

## Explanation

Marginalization "projects" the joint distribution onto a subset of variables by:
1. Considering all possible combinations with the variables of interest
2. Summing/integrating over the variables being eliminated
3. Preserving only the probability structure for the remaining variables

## Example: Discrete Case

**Scenario**: Students' study hours and exam results

Let:
- $X$: Study hours per week (1, 2, or 3 hours)
- $Y$: Exam result (Pass or Fail)

**Joint probability distribution** $P(X, Y)$:

| X \ Y | Pass | Fail | **Marginal P(X)** |
|-------|------|------|-------------------|
| 1 hr  | 0.1  | 0.3  | **0.4**          |
| 2 hr  | 0.2  | 0.2  | **0.4**          |
| 3 hr  | 0.3  | 0.1  | **0.4**          |
| **Marginal P(Y)** | **0.6** | **0.4** | **1.0**          |

### Calculating Marginal Distributions

**Marginal distribution of X (study hours)**:
- $P(X=1) = 0.1 + 0.3 = 0.4$
- $P(X=2) = 0.2 + 0.2 = 0.4$
- $P(X=3) = 0.3 + 0.1 = 0.4$

**Marginal distribution of Y (exam result)**:
- $P(Y=\text{Pass}) = 0.1 + 0.2 + 0.3 = 0.6$
- $P(Y=\text{Fail}) = 0.3 + 0.2 + 0.1 = 0.4$

## Interpretation

The marginal distributions tell us:
- Students are equally likely to study 1, 2, or 3 hours (each 0.4 probability)
- The overall pass rate is 60%, regardless of study hours

Without the joint distribution, we wouldn't know that students who study 3 hours have a higher pass rate ($0.3/0.4 = 75\%$) than those who study 1 hour ($0.1/0.4 = 25\%$). This relationship is captured by the **conditional distribution**, which differs from the marginal distribution.

## Key Properties

1. **Normalization**: Marginal probabilities sum to 1
2. **Information loss**: Marginal distributions contain less information than joint distributions
3. **Foundation**: Used in calculating conditional probabilities and expectations
4. **Application**: Essential in Bayesian statistics, regression analysis, and machine learning