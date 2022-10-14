module PrunedFIsAggressiveBackend

export PrunedFIsAggressive

import ..StochasticAD

"""
    PrunedFIsAggressiveState

State maintained by aggressive pruning backend.
"""
mutable struct PrunedFIsAggressiveState
    active_tag::Int64 # 0 is always a dummy tag
    weight::Float64
    tag_count::Int64
    valid::Bool
    PrunedFIsAggressiveState(valid = true) = new(0, 0.0, 0, valid)
end

"""
    PrunedFIsAggressive{V} <: StochasticAD.AbstractFIs{V}

A backend that aggressively prunes between perturbations as soon as they are created.
"""
struct PrunedFIsAggressive{V} <: StochasticAD.AbstractFIs{V}
    Δ::V
    tag::Int
    state::PrunedFIsAggressiveState
    # directly called when propagating an existing perturbation
end

### Empty / no perturbation

function PrunedFIsAggressive{V}(state::PrunedFIsAggressiveState) where {V}
    PrunedFIsAggressive{V}(zero(V), -1, state)
end
function StochasticAD.similar_empty(Δs::PrunedFIsAggressive, V::Type)
    PrunedFIsAggressive{V}(Δs.state)
end
function Base.empty(Δs::PrunedFIsAggressive{V}) where {V}
    similar_empty(Δs::PrunedFIsAggressive, V::Type)
end
# we truly have no clue what the state is here, so use an invalidated state
function Base.empty(::Type{<:PrunedFIsAggressive{V}}) where {V}
    PrunedFIsAggressive{V}(PrunedFIsAggressiveState(false))
end

### Create a new perturbation with infinitesimal probability

function new_perturbation(Δ::V, w::Real, state::PrunedFIsAggressiveState) where {V}
    total_weight = state.weight + w
    if rand(StochasticAD.RNG) * total_weight < state.weight
        state.weight += w
        return PrunedFIsAggressive{V}(state)
    else
        state.tag_count += 1
        state.active_tag = state.tag_count
        state.weight += w
        return PrunedFIsAggressive{V}(Δ, state.active_tag, state)
    end
end
function StochasticAD.similar_new(Δs::PrunedFIsAggressive, Δ::V, w::Real) where {V}
    new_perturbation(Δ, w, Δs.state)
end

### Create Δs backend for the first stochastic triple of computation

PrunedFIsAggressive{V}() where {V} = PrunedFIsAggressive{V}(PrunedFIsAggressiveState())

### Convert type of a backend

function PrunedFIsAggressive{V}(Δs::PrunedFIsAggressive) where {V}
    PrunedFIsAggressive{V}(convert(V, Δs.Δ), Δs.tag, Δs.state)
end

### Getting information about perturbations

# "empty" here means no perturbation or a perturbation that has been pruned away
Base.isempty(Δs::PrunedFIsAggressive) = Δs.tag != Δs.state.active_tag
Base.length(Δs::PrunedFIsAggressive) = isempty(Δs) ? 0 : 1
Base.iszero(Δs::PrunedFIsAggressive) = isempty(Δs) || iszero(Δs.Δ)

# we lazily prune, so check if empty first
pruned_value(Δs::PrunedFIsAggressive{V}) where {V} = isempty(Δs) ? zero(V) : Δs.Δ
StochasticAD.derivative_contribution(Δs::PrunedFIsAggressive) = pruned_value(Δs) * Δs.state.weight

perturbations(Δs::PrunedFIsAggressive) = ((pruned_value(Δs), Δs.state.weight),)

### Unary propagation

function Base.map(f, Δs::PrunedFIsAggressive)
    PrunedFIsAggressive(f(Δs.Δ), Δs.tag, Δs.state)
end

StochasticAD.alltrue(Δs::PrunedFIsAggressive{Bool}) = Δs.Δ

### Coupling

function StochasticAD.get_rep(::Type{<:PrunedFIsAggressive}, Δs_all)
    for Δs in Δs_all
        if Δs.state.valid
            return Δs
        end
    end
    return first(Δs_all)
end

# for pruning, coupling amounts to getting rid of perturbed values that have been
# lazily kept around even after (aggressive or lazy) pruning made the perturbation invalid.
function StochasticAD.couple(::Type{<:PrunedFIsAggressive}, Δs_all;
                             rep = StochasticAD.get_rep(Δs_all))
    state = rep.state
    Δ_coupled = map(pruned_value, Δs_all) # TODO: perhaps a performance optimization possible here
    PrunedFIsAggressive(Δ_coupled, state.active_tag, state)
end

# basically couple combined with a sum.
function StochasticAD.combine(::Type{<:PrunedFIsAggressive}, Δs_all;
                              rep = StochasticAD.get_rep(Δs_all...))
    state = rep.state
    Δ_combined = sum(pruned_value(Δs) for Δs in Δs_all)
    PrunedFIsAggressive(Δ_combined, state.active_tag, state)
end

StochasticAD.similar_type(::Type{<:PrunedFIsAggressive}, V::Type) = PrunedFIsAggressive{V}

### Miscellaneous

# should I have a mime input?
function Base.show(io::IO, mime::MIME"text/plain",
                   Δs::PrunedFIsAggressive{V}) where {V}
    print(io, "$(pruned_value(Δs)) with probability $(Δs.state.weight)ε, tag $(Δs.tag)")
end

function Base.show(io::IO, Δs::PrunedFIsAggressive{V}) where {V}
    print(io, "$(pruned_value(Δs)) with probability $(Δs.state.weight)ε, tag $(Δs.tag)")
end

end
