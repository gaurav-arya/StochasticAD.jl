@doc raw"""
    StochasticModel(X, p)

Combine stochastic program `X` with parameter `p` into 
a trainable model using [Functors](https://fluxml.ai/Functors.jl/stable/), where
`p <: AbstractArray`.
Formulate as a minimization problem, i.e. find ``p`` that minimizes ``\mathbb{E}[X(p)]``.
"""
struct StochasticModel{S <: AbstractArray, T}
    X::T
    p::S
end
@functor StochasticModel (p,)

"""
    stochastic_gradient(m::StochasticModel)

Compute gradient with respect to the trainable parameter `p` of `StochasticModel(X, p)`.
"""
function stochastic_gradient(m::StochasticModel)
    fmap(p -> derivative_estimate(m.X, p), m)
end
