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

### Rules for univariate single-parameter distributions

for (dist, i) in [(:Geometric, :1), (:Bernoulli, :1), (:Binomial, :2), (:Poisson, :1)] # i = index of the parameter p
    @eval function Base.rand(rng::AbstractRNG,
                             d_st::$dist{StochasticTriple{T, V, FIs}}) where {T, V, FIs}
        st = params(d_st)[$i]
        d = $dist(params(d_st)[1:($i - 1)]..., st.value, params(d_st)[($i + 1):end]...)
        val = convert(Signed, rand(rng, d))
        Δs1 = δtoΔps(d, val, st.δ, st.Δs)

        low = cdf(d, val - 1)
        high = cdf(d, val)

        function map_func(Δ)
            alt_d = $dist(params(d_st)[1:($i - 1)]..., st.value + Δ,
                          params(d_st)[($i + 1):end]...)
            alt_val = quantile(alt_d, rand(RNG) * (high - low) + low)
            convert(Signed, alt_val - val)
        end
        Δs2 = map(map_func, st.Δs)

        StochasticTriple{T}(val, zero(val), combine((Δs2, Δs1); rep = Δs1)) # ensure that tags are in order in combine, in case backend wishes to exploit this 
    end
end

"""
    DiscreteDeltaStochasticTriple{T, V, FIs <: AbstractFIs}

An experimental discrete stochastic triple type used internally for representing perturbations
to non-real quantities. Currently only used to represent a finite perturbation to the Binomial 
parameter n.

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

### Rule for Categorical variable

function δtoΔs(d::Categorical, val::V, δs, Δs::AbstractFIs) where {V <: Signed}
    p = params(d)[1]
    left_sum = sum(δs[1:(val - 1)], init = zero(V))
    right_sum = -sum(δs[(val + 1):end], init = zero(V))

    if left_sum > 0
        stop = rand() * left_sum
        upto = zero(eltype(δs)) # The "upto" logic handles an edge case of probability 0 events that have non-zero derivative.
        # It's a lot of logic to handle an edge case, but hopefully it's optimized away.
        local left_nonzero
        for i in (val - 1):-1:1
            if !iszero(p[i]) || ((upto += δs[i]) > stop)
                left_nonzero = i
                break
            end
        end
        Δs_left = similar_new(Δs, left_nonzero - val, left_sum / p[val])
    else
        Δs_left = similar_empty(Δs, typeof(val))
    end

    if right_sum < 0
        stop = -rand() * right_sum
        upto = zero(eltype(δs))
        local right_nonzero
        for i in (val + 1):length(p)
            if !iszero(p[i]) || ((upto += δs[i]) > stop)
                right_nonzero = i
                break
            end
        end
        Δs_right = similar_new(Δs, right_nonzero - val, -right_sum / p[val])
    else
        Δs_right = similar_empty(Δs, typeof(val))
    end

    return combine((Δs_left, Δs_right); rep = Δs)
end

# what if some elements in vector are not stochastic triples... promotion should take care of that?
function Base.rand(rng::AbstractRNG,
                   d_st::Categorical{<:StochasticTriple{T},
                                     <:AbstractVector{<:StochasticTriple{T, V}}}) where {T,
                                                                                         V}
    sts = params(d_st)[1] # stochastic triple for each probability
    p = map(st -> st.value, sts) # try to keep the same type. e.g. static array -> static array. TODO: avoid allocations 
    d = Categorical(p)
    val = convert(Signed, rand(rng, d))

    Δs_all = map(st -> st.Δs, sts)
    Δs_rep = get_rep(Δs_all)

    Δs1 = δtoΔs(d, val, map(st -> st.δ, sts), Δs_rep)

    low = cdf(d, val - 1)
    high = cdf(d, val)
    Δs_coupled = couple(Δs_all; rep = Δs_rep) # TODO: again, there are possible allocations here

    function map_func(Δ)
        alt_val = quantile(Categorical(p .+ Δ), rand(RNG) * (high - low) + low)
        convert(Signed, alt_val - val)
    end
    Δs2 = map(map_func, Δs_coupled)

    StochasticTriple{T}(val, zero(val), combine((Δs2, Δs1); rep = Δs_rep))
end
