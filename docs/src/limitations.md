# Limitations of StochasticAD

`StochasticAD` has a number of limitations that are important to be aware of:

* `StochasticAD` uses operator-overloading just like [ForwardDiff](https://juliadiff.org/ForwardDiff.jl/stable/), so all of the [limitations](https://juliadiff.org/ForwardDiff.jl/stable/user/limitations/) listed there apply here too. Also note that some useful features of `ForwardDiff`, such as chunking for greater efficiency with a large number of parameters, have not yet been implemented here.
* We have limited support for reverse-mode AD via [smoothing](public_api#Smoothing), which cannot be guaranteed to be unbiased in all cases. 
* We do not yet support `if` statements with discrete random input. A workaround can be to use array indexing to express discrete random choices (see [the random walk tutorial](tutorials/random_walk.md) for an example).
* We do not yet support non-real values as intermediate values (e.g. a function such as `length(A[rand(Bernoulli(p))])` where `A` is an array of strings is in theory differentiable).
* We do not support discrete random variables that are implicitly implemented using continuous random variables, e.g. `rand() < p`.
* We support a limited assortment of discrete random variables: currently `Bernoulli`, `Binomial`, `Geometric`, `Poisson`, and `Categorical` from [Distributions](https://juliastats.org/Distributions.jl/). We are working on increasing coverage across `Distributions` as well as other libraries providing discrete random samplers such as [MeasureTheory](https://cscherrer.github.io/MeasureTheory.jl/stable/).
* Higher-order differentiation is not supported.

`StochasticAD` is still in active development! PRs are welcome.

