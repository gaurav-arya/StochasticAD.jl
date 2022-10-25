### Particle resampling

@doc raw"""
    new_weight(p::Real)

    Simulate a Bernoulli variable whose primal output is always 1. 
    Uses a smoothing rule for use in forward and reverse-mode AD, which is exactly unbiased when the quantity is only
    used in linear functions  (e.g. used as an [importance weight](https://en.wikipedia.org/wiki/Importance_sampling)).
"""
function new_weight(p::ForwardDiff.Dual{T}) where {T}
    ∂_p = ForwardDiff.partials(p)
    value_p = ForwardDiff.value(p)
    value_p = max(1e-5, value_p) # TODO: is this necessary?
    ForwardDiff.Dual{T}(1, ∂_p / value_p)
end
new_weight(p::Real) = 1

function ChainRulesCore.rrule(::typeof(new_weight), p)
    function new_weight_pullback(∇Ω)
        return (ChainRulesCore.NoTangent(), ∇Ω / p)
    end
    return (one(p), new_weight_pullback)
end

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
