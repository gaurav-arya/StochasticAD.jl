abstract type AbstractStochasticADAlgorithm end

"""
    ForwardAlgorithm(backend::StochasticAD.AbstractFIsBackend) <: AbstractStochasticADAlgorithm
    
A differentiation algorithm relying on forward propagation of stochastic triples.

The `backend` argument controls the algorithm used by the third component of the stochastic triples.

!!! note 
    The required computation time for forward-mode AD scales linearly with the number of 
    parameters in `p` (but is unaffected by the number of parameters in `X(p)`).
"""
struct ForwardAlgorithm{B <: StochasticAD.AbstractFIsBackend} <: AbstractStochasticADAlgorithm
    backend::B
end

"""
    EnzymeReverseAlgorithm(backend::StochasticAD.AbstractFIsBackend) <: AbstractStochasticADAlgorithm

A differentiation algorithm relying on transposing the propagation of stochastic triples to
produce a reverse-mode algorithm. The transposition is performed by [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl),
which must be loaded for the algorithm to run.

Currently, only real- and vector-valued inputs are supported, and only real-valued outputs are supported.

The `backend` argument controls the algorithm used by the third component of the stochastic triples.

!!! warning
    For the reverse-mode algorithm to yield correct results, the employed `backend` cannot use input-dependent pruning  
    strategies. A suggested reverse-mode compatible backend is `PrunedFIsBackend(Val(:wins))`.
    
    Additionally, this algorithm relies on the ability of `Enzyme.jl` to differentiate the forward stochastic triple run.
    It is recommended to check that the primal function `X` is type stable for its input `p` using a tool such as
    [JET.jl](https://github.com/aviatesk/JET.jl), with all code executed in a function with no global state. 
    In addition, sometimes `X` may be type stable but stochastic triples introduce additional type stabilities.
    This can be debugged by checking type stability of Enzyme's target, which is
    `Base.get_extension(StochasticADExtra, :StochasticADExtraEnzymeExt).enzyme_target(u, X, p, backend)`,
    where `u` is a test direction.
"""
struct EnzymeReverseAlgorithm{B <: StochasticAD.AbstractFIsBackend}
    backend::B
end

function derivative_estimate(X, p, alg::ForwardAlgorithm; direction = nothing)
    return derivative_estimate(X, p; backend = alg.backend, direction)
end

@doc raw"""
    derivative_estimate(X, p, alg::AbstractStochasticADAlgorithm = ForwardAlgorithm(PrunedFIsBackend()); direction=nothing)

Compute an unbiased estimate of ``\frac{\mathrm{d}\mathbb{E}[X(p)]}{\mathrm{d}p}``, 
the derivative of the expectation of the random function `X(p)` with respect to its input `p`.

Both `p` and `X(p)` can be any object supported by [`Functors.jl`](https://fluxml.ai/Functors.jl/stable/),
e.g. scalars or abstract arrays. 
The output of `derivative_estimate` has the same outer structure as `p`, but with each
scalar in `p` replaced by a derivative estimate of `X(p)` with respect to that entry.
For example, if `X(p) <: AbstractMatrix` and `p <: Real`, then the output would be a matrix.

The `alg` keyword argument specifies the [algorithm](public_api.md#Algorithms) used to compute the derivative estimate.
For backward compatibility, an additional signature `derivative_estimate(X, p; backend, direction=nothing)`
is supported, which uses `ForwardAlgorithm` by default with the supplied `backend.`

When `direction` is provided, the output is only differentiated with respect to a perturbation
of `p` in that direction.

# Example
```jldoctest
julia> using Distributions, Random, StochasticAD; Random.seed!(4321);

julia> derivative_estimate(rand ∘ Bernoulli, 0.5) # A random quantity that averages to the true derivative.
2.0

julia> derivative_estimate(x -> [rand(Bernoulli(x * i/4)) for i in 1:3], 0.5)
3-element Vector{Float64}:
 0.2857142857142857
 0.6666666666666666
 0.0
```
"""

derivative_estimate
