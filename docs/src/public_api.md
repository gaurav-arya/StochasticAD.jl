# API walkthrough
 
The function [`derivative_estimate`](@ref) transforms a stochastic program containing discrete randomness into a new program whose average is the derivative of the original.
```@docs
derivative_estimate
```
While [`derivative_estimate`](@ref) is self-contained, we can also use the functions below to work with stochastic triples directly.
```@docs
StochasticAD.stochastic_triple
StochasticAD.derivative_contribution
StochasticAD.value
StochasticAD.delta
StochasticAD.perturbations
```
Note that [`derivative_estimate`](@ref) is simply the composition of [`stochastic_triple`](@ref) and [`derivative_contribution`](@ref). We also provide a convenience function for mimicking the behaviour
of standard AD, where derivatives of discrete random steps are dropped:
```@docs
StochasticAD.dual_number
```

## Smoothing

What happens if we were to run [`derivative_contribution`](@ref) after each step, instead of only at the end? This is *smoothing*, which combines the second and third components of a single stochastic triple into a single dual component. 
Smoothing no longer has a guarantee of unbiasedness, but is surprisingly accurate in a number of situations. 
For example, the popular [straight through gradient estimator](https://stackoverflow.com/questions/38361314/the-concept-of-straight-through-estimator-ste) can be viewed as a special case of smoothing.
Forward smoothing rules are provided through `ForwardDiff`, and backward rules through `ChainRules`, so that e.g. `Zygote.gradient` and `ForwardDiff.derivative` will use smoothed rules for discrete random variables rather than dropping the gradients entirely. 
Currently, special discrete->discrete constructs such as array indexing are not supported for smoothing.




## Optimization

We also provide utilities to make it easier to get started with forming and training a model via stochastic gradient descent:
```@docs
StochasticAD.StochasticModel
StochasticAD.stochastic_gradient
```
These are used in the [tutorial on stochastic optimization](tutorials/optimizations.md).
