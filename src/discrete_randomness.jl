## Rules for univariate uniparameter discrete distributions

"""
    δtoΔps(d, val::V, δ::Real, Δs::AbstractFIs)

Given the parameter `val` of a distribution `d` and an infinitesimal change `δ`,
return the discrete change in the output, with a similar representation to `Δs`.
"""
function δtoΔps(d::Geometric, val::V, δ::Real, Δs::AbstractFIs) where {V <: Signed}
    p = succprob(d)
    if δ > 0
        return val > 0 ? similar_new(Δs, -one(V), δ * val / p / (1 - p)) :
               similar_empty(Δs, V)
    elseif δ < 0
        return similar_new(Δs, one(V), -δ * (val + 1) / p)
    else
        return similar_empty(Δs, V)
    end
end

function δtoΔps(d::Bernoulli, val::V, δ::Real, Δs::AbstractFIs) where {V <: Signed}
    p = succprob(d)
    if δ > 0
        return isone(val) ? similar_empty(Δs, V) : similar_new(Δs, one(V), δ / (1 - p))
    elseif δ < 0
        return isone(val) ? similar_new(Δs, -one(V), -δ / p) : similar_empty(Δs, V)
    else
        return similar_empty(Δs, V)
    end
end

function δtoΔps(d::Binomial, val::V, δ::Real, Δs::AbstractFIs) where {V <: Signed}
    p = succprob(d)
    n = ntrials(d)
    if δ > 0
        return val == n ? similar_empty(Δs, V) :
               similar_new(Δs, one(V), δ * (n - val) / (1 - p))
    elseif δ < 0
        return !iszero(val) ? similar_new(Δs, -one(V), -δ * val / p) : similar_empty(Δs, V)
    else
        return similar_empty(Δs, V)
    end
end

function δtoΔps(d::Poisson, val::V, δ::Real, Δs::AbstractFIs) where {V <: Signed}
    p = mean(d) # rate
    if δ > 0
        return similar_new(Δs, 1, δ)
    elseif δ < 0
        return val > 0 ? similar_new(Δs, -1, -δ * val / p) : similar_empty(Δs, V)
    else
        return similar_empty(Δs, V)
    end
end

### Rules for univariate single-paramter distributions

for (dist, i) in [(:Geometric, :1), (:Bernoulli, :1), (:Binomial, :2), (:Poisson, :1)] # i = index of the parameter p
    @eval function Base.rand(rng::AbstractRNG,
                             d_st::$dist{StochasticTriple{T, V, FIs}}) where {T, V, FIs}
        st = params(d_st)[$i]
        d = $dist(params(d_st)[1:($i - 1)]..., st.value, params(d_st)[($i + 1):end]...)
        val = convert(Signed, rand(rng, d))
        Δs1 = δtoΔps(d, val, st.δ, st.Δs)

        low = cdf(d, val - 1)
        high = cdf(d, val)
        Δs2 = map(Δ -> convert(Signed,
                               quantile($dist(params(d_st)[1:($i - 1)]..., st.value + Δ,
                                              params(d_st)[($i + 1):end]...),
                                        rand(RNG) * (high - low) + low) - val), st.Δs)

        StochasticTriple{T}(val, zero(val), combine((Δs2, Δs1); rep = Δs1)) # ensure that tags are in order in combine, in case backend wishes to exploit this 
    end
end

"""
    DiscreteDeltaStochasticTriple{T, V, FIs <: AbstractFIs}

An experimental discrete stochastic triple type used internally for representing perturbations
to non-real quantities. Currently only used to represent a finite perturbation to the Binomial n.

## Constructor

- `value`: the primal value.
- `Δs``: some representation of the perturbation to the primal, which can have an unconventional
         interpretation depending on `T`.
"""
struct DiscreteDeltaStochasticTriple{T, V, FIs <: AbstractFIs}
    value::V
    Δs::FIs
    function DiscreteDeltaStochasticTriple{T, V, FIs}(value::V,
                                                      Δs::FIs) where {T, V,
                                                                      FIs <: AbstractFIs}
        new{T, V, FIs}(value, Δs)
    end
end

function DiscreteDeltaStochasticTriple{T}(val::V, Δs::FIs) where {T, V, FIs <: AbstractFIs}
    DiscreteDeltaStochasticTriple{T, V, FIs}(val, Δs)
end

### Handling finite perturbation to Binomial number of trials

function Distributions.Binomial(n::StochasticTriple{T}, p::Real) where {T}
    return DiscreteDeltaStochasticTriple{T}(Binomial(n.value, p), n.Δs)
end

# TODO: Support functions other than `rand` called on a perturbed Binomial.
function Base.rand(rng::AbstractRNG,
                   d_st::DiscreteDeltaStochasticTriple{T, <:Binomial}) where {T}
    d = d_st.value
    val = rand(rng, d)
    function map_func(Δ)
        if Δ >= 0
            return rand(StochasticAD.RNG, Binomial(Δ, value(succprob(d))))
        else
            return -rand(StochasticAD.RNG,
                         Hypergeometric(value(val), ntrials(d) - value(val), -Δ))
        end
    end
    Δs = map(map_func, d_st.Δs)
    if val isa StochasticTriple
        return StochasticTriple{T}(val.value, val.δ, combine((Δs, val.Δs); rep = Δs))
    else
        return StochasticTriple{T}(val, zero(val), Δs)
    end
end

# TODO: add Categorical rule