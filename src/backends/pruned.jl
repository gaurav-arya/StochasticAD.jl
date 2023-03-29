module PrunedFIsModule

import ..StochasticAD

export PrunedFIsBackend

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
    active_tag::Int64 # 0 is always a dummy tag
    weight::Float64
    tag_count::Int64
    valid::Bool
    PrunedFIsState(valid = true) = new(0, 0.0, 0, valid)
end

"""
    PrunedFIs{V} <: StochasticAD.AbstractFIs{V}

The implementing backend structure for PrunedFIsBackend.
"""
struct PrunedFIs{V} <: StochasticAD.AbstractFIs{V}
    Δ::V
    tag::Int
    state::PrunedFIsState
    # directly called when propagating an existing perturbation
end

### Empty / no perturbation

PrunedFIs{V}(state::PrunedFIsState) where {V} = PrunedFIs{V}(zero(V), -1, state)
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
    state.tag_count = 1
    state.active_tag = 1
    Δs = PrunedFIs{V}(Δ, 1, state)
    return Δs
end

### Create Δs backend for the first stochastic triple of computation

StochasticAD.create_Δs(::PrunedFIsBackend, V) = PrunedFIs{V}(PrunedFIsState(false))

### Convert type of a backend

function PrunedFIs{V}(Δs::PrunedFIs) where {V}
    PrunedFIs{V}(convert(V, Δs.Δ), Δs.tag, Δs.state)
end

### Getting information about perturbations

# "empty" here means no perturbation or a perturbation that has been pruned away
Base.isempty(Δs::PrunedFIs) = !Δs.state.valid || (Δs.tag != Δs.state.active_tag)
Base.length(Δs::PrunedFIs) = isempty(Δs) ? 0 : 1
Base.iszero(Δs::PrunedFIs) = isempty(Δs) || iszero(Δs.Δ)
Base.iszero(Δs::PrunedFIs{<:Tuple}) = isempty(Δs) || all(iszero.(Δs.Δ))
isapproxzero(Δs::PrunedFIs) = isempty(Δs) || isapprox(Δs.Δ, zero(Δs.Δ))

# we lazily prune, so check if empty first
# TODO: possibly generalize these methods to all structures for Δs.Δ, as needed.
pruned_value(Δs::PrunedFIs{V}) where {V} = isempty(Δs) ? zero(V) : Δs.Δ
pruned_value(Δs::PrunedFIs{<:Tuple}) = isempty(Δs) ? zero.(Δs.Δ) : Δs.Δ
pruned_value(Δs::PrunedFIs{<:AbstractArray}) = isempty(Δs) ? zero.(Δs.Δ) : Δs.Δ
function StochasticAD.filter_state(Δs::PrunedFIs{V}, state) where {V}
    Δs.state === state ? pruned_value(Δs) : zero(V)
end
StochasticAD.derivative_contribution(Δs::PrunedFIs) = pruned_value(Δs) * Δs.state.weight
StochasticAD.perturbations(Δs::PrunedFIs) = ((pruned_value(Δs), Δs.state.weight),)

### Unary propagation

function StochasticAD.map_Δs(f, Δs::PrunedFIs; kwargs...)
    PrunedFIs(f(Δs.Δ, Δs.state), Δs.tag, Δs.state)
end

StochasticAD.alltrue(Δs::PrunedFIs{Bool}) = Δs.Δ

### Coupling

function StochasticAD.get_rep(::Type{<:PrunedFIs}, Δs_all)
    # The code below is a bit ridiculous, but it's faster than `first` for small structures:)
    foldl((Δs1, Δs2) -> Δs1, StochasticAD.structural_iterate(Δs_all))
end

function get_pruned_state(Δs_all)
    function op(reduced, Δs)
        total_weight, new_state = reduced
        isapproxzero(Δs) && return (total_weight, new_state)
        candidate_state = Δs.state
        if !candidate_state.valid ||
           ((new_state !== nothing) && (candidate_state === new_state))
            return (total_weight, new_state)
        end
        w = candidate_state.weight
        total_weight += w
        if rand(StochasticAD.RNG) * total_weight < w
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
        _new_state.weight = _total_weight
    else
        _new_state = PrunedFIsState(false) # TODO: can this be avoided?
    end
    return _new_state::PrunedFIsState
end

# for pruning, coupling amounts to getting rid of perturbed values that have been
# lazily kept around even after (aggressive or lazy) pruning made the perturbation invalid.
# rep is unused.
function StochasticAD.couple(::Type{<:PrunedFIs}, Δs_all; rep = nothing)
    state = get_pruned_state(Δs_all)
    Δ_coupled = StochasticAD.structural_map(pruned_value, Δs_all) # TODO: perhaps a performance optimization possible here
    PrunedFIs(Δ_coupled, state.active_tag, state)
end

# basically couple combined with a sum.
function StochasticAD.combine(::Type{<:PrunedFIs}, Δs_all; rep = nothing)
    state = get_pruned_state(Δs_all)
    Δ_combined = sum(pruned_value, StochasticAD.structural_iterate(Δs_all))
    PrunedFIs(Δ_combined, state.active_tag, state)
end

function StochasticAD.scalarize(Δs::PrunedFIs)
    return StochasticAD.structural_map(Δs.Δ) do Δ
        return PrunedFIs(Δ, Δs.tag, Δs.state)
    end
end

### Miscellaneous

StochasticAD.similar_type(::Type{<:PrunedFIs}, V::Type) = PrunedFIs{V}
StochasticAD.valtype(::Type{<:PrunedFIs{V}}) where {V} = V

function Base.show(io::IO, Δs::PrunedFIs{V}) where {V}
    print(io, "$(pruned_value(Δs)) with probability $(Δs.state.weight)ε, tag $(Δs.tag)")
end

end
