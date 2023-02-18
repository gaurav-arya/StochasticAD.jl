### Particle resampling

@doc raw"""
    new_weight(p::Real)

    Simulate a Bernoulli variable whose primal output is always 1. 
    Uses a smoothing rule for use in forward and reverse-mode AD, which is exactly unbiased when the quantity is only
    used in linear functions  (e.g. used as an [importance weight](https://en.wikipedia.org/wiki/Importance_sampling)).
"""
new_weight(p::Real) = 1

function new_weight(p::ForwardDiff.Dual{T}) where {T}
    Δp = ForwardDiff.partials(p)
    val_p = ForwardDiff.value(p)
    val_p = max(1e-5, val_p) # TODO: is this necessary?
    ForwardDiff.Dual{T}(one(val_p), Δp / val_p)
end

function ChainRulesCore.frule((_, Δp), ::typeof(new_weight), p::Real)
    val_p = max(1e-5, p) # TODO: is this necessary?
    return one(p), Δp / val_p
end

function ChainRulesCore.rrule(::typeof(new_weight), p)
    function new_weight_pullback(∇Ω)
        return (ChainRulesCore.NoTangent(), ∇Ω / p)
    end
    return (one(p), new_weight_pullback)
end

# Smoothed rules for univariate single-parameter distributions. 

function smoothed_delta(d, val, δ)
    Δs_empty = SmoothedFIs{typeof(val)}(0.0)
    partial_right = derivative_contribution(δtoΔs(d, val, δ, Δs_empty))
    partial_left = -derivative_contribution(δtoΔs(d, val, -δ, Δs_empty))
    return (partial_left + partial_right) / 2
end

for (dist, i, field) in [
    (:Geometric, :1, :p),
    (:Bernoulli, :1, :p),
    (:Binomial, :2, :p),
    (:Poisson, :1, :λ),
    (:Categorical, :1, :p),
] # i = index of parameter p
    @eval function Base.rand(rng::AbstractRNG,
                             d_dual::$dist{<:ForwardDiff.Dual{T}}) where {T}
        dual = params(d_dual)[$i]
        # dual could represent an array of duals or a single one; map handles both cases.
        p = map(value, dual)
        # Generate a δ for each partial component.
        partials_indices = ntuple(identity, length(first(dual).partials))
        δs = map(i -> map(d -> ForwardDiff.partials(d)[i], dual), partials_indices)
        d = $dist(params(d_dual)[1:($i - 1)]..., p,
                  params(d_dual)[($i + 1):end]...)
        val = convert(Signed, rand(rng, d))
        partials = ForwardDiff.Partials(map(δ -> smoothed_delta(d, val, δ), δs))
        ForwardDiff.Dual{T}(val, partials)
    end
    @eval function ChainRulesCore.rrule(::typeof(rand), rng::AbstractRNG, d::$dist)
        val = convert(Signed, rand(rng, d))
        function rand_pullback(∇out)
            p = params(d)[$i]
            if p isa Real
                Δp = smoothed_delta(d, val, one(val))
            else
                # TODO: this rule is O(length(p)^2), whereas we should be able to do O(length(p)) by reversing through δtoΔs.
                I = eachindex(p)
                V = eltype(p)
                onehot(i) = map(j -> j == i ? one(V) : zero(V), I)
                Δp = map(i -> smoothed_delta(d, val, onehot(i)), I)
            end
            # rrule_via_ad approach below not used because slow.
            # Δp = rrule_via_ad(config, smoothed_delta, d, val, map(one, p))[2](∇out)[4]
            Δd = ChainRulesCore.Tangent{typeof(d)}(; $field = ∇out * Δp)
            return (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), Δd)
        end
        return (val, rand_pullback)
    end
end
