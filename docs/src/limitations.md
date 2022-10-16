# Limitations of StochasticAD

`StochasticAD` has a number of limitations that are important to be aware of:

* `StochasticAD` uses operator-overloading just like [ForwardDiff](https://juliadiff.org/ForwardDiff.jl/stable/), so all of the [limitations](https://juliadiff.org/ForwardDiff.jl/stable/user/limitations/) listed there apply here too. Also note that some useful features of `ForwardDiff`, such as chunking for greater efficiency with a large number of parameters, have not yet been implemented here.
* We have limited support for reverse-mode AD, via smoothing, which cannot be guaranteed to be unbiased in all cases. [Note: smoothing functionality not yet added.]
* We do not yet support `if` statements with discrete random input (a workaround can be to use array indexing).
* We do not support discrete random variables that are implicitly implemented using continuous random variables, e.g. `rand() < p`.
* We have a limited assortment of discrete random variables: currently `Bernoulli`, `Binomial`, `Geometric`, `Poisson`, and `Categorical`. We are working on increasing coverage.
* Nested differentiation is not supported.

`StochasticAD` is still in active development! PRs are welcome.

