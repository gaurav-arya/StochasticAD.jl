module PrunedFIsAggressiveModule

import ..StochasticAD

export PrunedFIsAggressiveBackend, PrunedFIsAggressive

"""
    PrunedFIsAggressiveBackend <: StochasticAD.AbstractFIsBackend

A backend algorithm that aggressively prunes between perturbations as soon as they are created.
"""
struct PrunedFIsAggressiveBackend <: StochasticAD.AbstractFIsBackend end

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

The implementing backend structure for PrunedFIsAggressiveBackend.
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
    StochasticAD.similar_empty(Δs, V)
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

function StochasticAD.create_Δs(::PrunedFIsAggressiveBackend, V)
    PrunedFIsAggressive{V}(PrunedFIsAggressiveState())
end

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

function StochasticAD.derivative_contribution(Δs::PrunedFIsAggressive)
    pruned_value(Δs) * Δs.state.weight
end

StochasticAD.perturbations(Δs::PrunedFIsAggressive) = ((pruned_value(Δs), Δs.state.weight),)

### Unary propagation

function StochasticAD.weighted_map_Δs(f, Δs::PrunedFIsAggressive; kwargs...)
    Δ_out, weight_out = f(Δs.Δ, nothing)
    Δs.state.weight *= weight_out
    PrunedFIsAggressive(Δ_out, Δs.tag, Δs.state)
end

StochasticAD.alltrue(f, Δs::PrunedFIsAggressive) = f(Δs.Δ)

### Coupling

function StochasticAD.get_rep(::Type{<:PrunedFIsAggressive}, Δs_all)
    # Get some Δs with a valid state, or any if all are invalid.
    return reduce((Δs1, Δs2) -> Δs1.state.valid ? Δs1 : Δs2,
        StochasticAD.structural_iterate(Δs_all))
end

# for pruning, coupling amounts to getting rid of perturbed values that have been
# lazily kept around even after (aggressive or lazy) pruning made the perturbation invalid.
function StochasticAD.couple(FIs::Type{<:PrunedFIsAggressive}, Δs_all;
        rep = StochasticAD.get_rep(FIs, Δs_all),
        out_rep = nothing)
    state = rep.state
    Δ_coupled = StochasticAD.structural_map(pruned_value, Δs_all) # TODO: perhaps a performance optimization possible here
    PrunedFIsAggressive(Δ_coupled, state.active_tag, state)
end

# basically couple combined with a sum.
function StochasticAD.combine(FIs::Type{<:PrunedFIsAggressive}, Δs_all;
        rep = StochasticAD.get_rep(FIs, Δs_all))
    state = rep.state
    Δ_combined = sum(pruned_value, StochasticAD.structural_iterate(Δs_all))
    PrunedFIsAggressive(Δ_combined, state.active_tag, state)
end

function StochasticAD.scalarize(Δs::PrunedFIsAggressive; out_rep = nothing)
    return StochasticAD.structural_map(Δs.Δ) do Δ
        return PrunedFIsAggressive(Δ, Δs.tag, Δs.state)
    end
end

StochasticAD.filter_state(Δs::PrunedFIsAggressive, _) = pruned_value(Δs)

### Miscellaneous

StochasticAD.similar_type(::Type{<:PrunedFIsAggressive}, V::Type) = PrunedFIsAggressive{V}
StochasticAD.valtype(::Type{<:PrunedFIsAggressive{V}}) where {V} = V

# should I have a mime input?
function Base.show(io::IO, mime::MIME"text/plain",
        Δs::PrunedFIsAggressive{V}) where {V}
    print(io, "$(pruned_value(Δs)) with probability $(Δs.state.weight)ε, tag $(Δs.tag)")
end

function Base.show(io::IO, Δs::PrunedFIsAggressive{V}) where {V}
    print(io, "$(pruned_value(Δs)) with probability $(Δs.state.weight)ε, tag $(Δs.tag)")
end

end
