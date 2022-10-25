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
Note that [`derivative_estimate`](@ref) is simply the composition of [`stochastic_triple`](@ref) and [`derivative_contribution`](@ref). 

## Smoothing

What happens if we run [`derivative_contribution`](@ref) after each step, instead of only at the end? This is *smoothing*, which combines the second and third components of a single stochastic triple into a single dual component. Smoothing no longer has a guarantee of unbiasedness, but is surprisingly accurate in a number of situations. 

[*Smoothing functionality coming soon.*]

## Optimization

We also provide utilities to make it easier to get started with forming and training a model via stochastic gradient descent:
```@docs
StochasticAD.StochasticModel
StochasticAD.stochastic_gradient
```
These are used in the [tutorial on stochastic optimization](tutorials/optimizations.md).
