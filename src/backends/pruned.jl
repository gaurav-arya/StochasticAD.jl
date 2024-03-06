module PrunedFIsModule

import ..StochasticAD

export PrunedFIsBackend, PrunedFIs

"""
    PrunedFIsBackend <: StochasticAD.AbstractFIsBackend

A backend algorithm that prunes between perturbations as soon as they clash (e.g. added together).
"""
struct PrunedFIsBackend <: StochasticAD.AbstractFIsBackend end

"""
    PrunedFIsState

State maintained by pruning backend.
"""
mutable struct PrunedFIsState
    # tag is in place to avoid relying on mutability for uniqueness. 
    tag::Int32
    weight::Float64
    valid::Bool
    function PrunedFIsState(valid = true)
        state = new(0, 0.0, valid)
        state.tag = objectid(state) % typemax(Int32)
        return state
    end
end

Base.:(==)(state1::PrunedFIsState, state2::PrunedFIsState) = state1.tag == state2.tag
# c.f. https://github.com/JuliaLang/julia/blob/61c3521613767b2af21dfa5cc5a7b8195c5bdcaf/base/hashing.jl#L38C45-L38C51
Base.hash(state::PrunedFIsState) = state.tag

"""
    PrunedFIs{V} <: StochasticAD.AbstractFIs{V}

The implementing backend structure for PrunedFIsBackend.
"""
struct PrunedFIs{V} <: StochasticAD.AbstractFIs{V}
    Δ::V
    state::PrunedFIsState
    # directly called when propagating an existing perturbation
end

### Empty / no perturbation

PrunedFIs{V}(state::PrunedFIsState) where {V} = PrunedFIs{V}(zero(V), state)
# TODO: avoid allocations here
StochasticAD.similar_empty(Δs::PrunedFIs, V::Type) = PrunedFIs{V}(PrunedFIsState(false))
Base.empty(Δs::PrunedFIs{V}) where {V} = StochasticAD.similar_empty(Δs::PrunedFIs, V::Type)
# we truly have no clue what the state is here, so use an invalidated state
function Base.empty(::Type{<:PrunedFIs{V}}) where {V}
    PrunedFIs{V}(PrunedFIsState(false))
end

### Create a new perturbation with infinitesimal probability

function StochasticAD.similar_new(Δs::PrunedFIs, Δ::V, w::Real) where {V}
    state = PrunedFIsState()
    state.weight += w
    Δs = PrunedFIs{V}(Δ, state)
    return Δs
end

### Create Δs backend for the first stochastic triple of computation

StochasticAD.create_Δs(::PrunedFIsBackend, V) = PrunedFIs{V}(PrunedFIsState(false))

### Convert type of a backend

function Base.convert(::Type{PrunedFIs{V}}, Δs::PrunedFIs) where {V}
    PrunedFIs{V}(convert(V, Δs.Δ), Δs.state)
end

### Getting information about perturbations

# "empty" here means no perturbation or a perturbation that has been pruned away
Base.isempty(Δs::PrunedFIs) = !Δs.state.valid
Base.length(Δs::PrunedFIs) = isempty(Δs) ? 0 : 1
Base.iszero(Δs::PrunedFIs) = isempty(Δs) || iszero(Δs.Δ)
Base.iszero(Δs::PrunedFIs{<:Tuple}) = isempty(Δs) || all(iszero.(Δs.Δ))
isapproxzero(Δs::PrunedFIs) = isempty(Δs) || isapprox(Δs.Δ, zero(Δs.Δ))

# we lazily prune, so check if empty first
# TODO: possibly generalize these methods to all structures for Δs.Δ, as needed.
pruned_value(Δs::PrunedFIs{V}) where {V} = isempty(Δs) ? zero(V) : Δs.Δ
pruned_value(Δs::PrunedFIs{<:Tuple}) = isempty(Δs) ? zero.(Δs.Δ) : Δs.Δ
pruned_value(Δs::PrunedFIs{<:AbstractArray}) = isempty(Δs) ? zero.(Δs.Δ) : Δs.Δ

StochasticAD.derivative_contribution(Δs::PrunedFIs) = pruned_value(Δs) * Δs.state.weight
function StochasticAD.perturbations(Δs::PrunedFIs)
    return ((; Δ = pruned_value(Δs),
        weight = Δs.state.valid ? Δs.state.weight : zero(Δs.state.weight),
        state = Δs.state),)
end

### Unary propagation

function StochasticAD.weighted_map_Δs(f, Δs::PrunedFIs; kwargs...)
    Δ_out, weight_out = f(Δs.Δ, Δs.state)
    # TODO: we could add a direct overload for map_Δs that elides the below line
    Δs.state.weight *= weight_out
    PrunedFIs(Δ_out, Δs.state)
end

StochasticAD.alltrue(f, Δs::PrunedFIs) = f(Δs.Δ)

### Coupling

function StochasticAD.get_rep(FIs::Type{<:PrunedFIs}, Δs_all)
    return empty(FIs)
end

function get_pruned_state(Δs_all)
    function op(reduced, Δs)
        total_weight, new_state = reduced
        isapproxzero(Δs) && return (total_weight, new_state)
        candidate_state = Δs.state
        if !candidate_state.valid ||
           ((new_state !== nothing) && (candidate_state == new_state))
            return (total_weight, new_state)
        end
        w = candidate_state.weight
        total_weight += abs(w)
        if rand(StochasticAD.RNG) * total_weight < abs(w)
            new_state !== nothing && (new_state.valid = false)
            new_state = candidate_state
        else
            candidate_state.valid = false
        end
        return (total_weight, new_state)
    end
    (_total_weight, _new_state) = foldl(op, StochasticAD.structural_iterate(Δs_all);
        init = (0.0, nothing))
    if _new_state !== nothing
        _new_state.weight = _total_weight * sign(_new_state.weight)
    else
        _new_state = PrunedFIsState(false) # TODO: can this be avoided?
    end
    return _new_state::PrunedFIsState
end

# for pruning, coupling amounts to getting rid of perturbed values that have been
# lazily kept around even after (aggressive or lazy) pruning made the perturbation invalid.
# rep is unused.
function StochasticAD.couple(
        ::Type{<:PrunedFIs}, Δs_all; rep = nothing, out_rep = nothing, kwargs...)
    state = get_pruned_state(Δs_all)
    Δ_coupled = StochasticAD.structural_map(pruned_value, Δs_all) # TODO: perhaps a performance optimization possible here
    PrunedFIs(Δ_coupled, state)
end

# basically couple combined with a sum.
function StochasticAD.combine(::Type{<:PrunedFIs}, Δs_all; rep = nothing, kwargs...)
    state = get_pruned_state(Δs_all)
    Δ_combined = sum(pruned_value, StochasticAD.structural_iterate(Δs_all))
    PrunedFIs(Δ_combined, state)
end

function StochasticAD.scalarize(Δs::PrunedFIs; out_rep = nothing)
    return StochasticAD.structural_map(Δs.Δ) do Δ
        return PrunedFIs(Δ, Δs.state)
    end
end

function StochasticAD.filter_state(Δs::PrunedFIs{V}, state) where {V}
    Δs.state == state ? pruned_value(Δs) : zero(V)
end

### Miscellaneous

StochasticAD.similar_type(::Type{<:PrunedFIs}, V::Type) = PrunedFIs{V}
StochasticAD.valtype(::Type{<:PrunedFIs{V}}) where {V} = V

function Base.show(io::IO, Δs::PrunedFIs{V}) where {V}
    print(io, "$(pruned_value(Δs)) with probability $(Δs.state.weight)ε")
end

end
