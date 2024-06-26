module PrunedFIsModule

import ..StochasticAD

export PrunedFIsBackend, PrunedFIs

"""
    PrunedFIsBackend <: StochasticAD.AbstractFIsBackend

A backend algorithm that prunes between perturbations as soon as they clash (e.g. added together).
Currently chooses uniformly between all perturbations.
"""
struct PrunedFIsBackend{M <: Val} <: StochasticAD.AbstractFIsBackend
    pruning_mode::M
    function PrunedFIsBackend(pruning_mode::M = Val(:weights)) where {M}
        if pruning_mode isa Val{:weights} || pruning_mode isa Val{:wins}
            return new{M}(pruning_mode)
        else
            error("Unsupported pruning_mode $pruning_mode for `PrunedFIsBackend.")
        end
    end
end

"""
    PrunedFIsState

State maintained by pruning backend.
"""
mutable struct PrunedFIsState{M, W}
    tag::Int32
    weight::Float64
    valid::Bool
    # TODO: generalize (wins, pruning_mode) into a general interface for accumulating state
    # that informs future pruning decisions.
    wins::W
    pruning_mode::M
    function PrunedFIsState(pruning_mode::M, valid = true) where {M <: Val}
        wins = pruning_mode isa Val{:wins} ? (valid ? 1 : 0) : nothing
        state::PrunedFIsState = new{M, typeof(wins)}(0, 0.0, valid, wins)
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
struct PrunedFIs{V, S <: PrunedFIsState} <: StochasticAD.AbstractFIs{V}
    Δ::V
    state::S
end

### Empty / no perturbation

PrunedFIs{V}(Δ::V, state::S) where {V, S <: PrunedFIsState} = PrunedFIs{V, S}(Δ, state)
PrunedFIs{V}(state::PrunedFIsState) where {V} = PrunedFIs{V}(zero(V), state)
# TODO: avoid allocations here
function StochasticAD.similar_empty(Δs::PrunedFIs, V::Type)
    PrunedFIs{V}(PrunedFIsState(Δs.state.pruning_mode, false))
end
Base.empty(Δs::PrunedFIs{V}) where {V} = StochasticAD.similar_empty(Δs::PrunedFIs, V::Type)
# we truly have no clue what the state is here, so use an invalidated state
function Base.empty(::Type{<:PrunedFIs{V, S}}) where {V, M, S <: PrunedFIsState{M}}
    PrunedFIs{V}(PrunedFIsState(M(), false))
end

### Create a new perturbation with infinitesimal probability

function StochasticAD.similar_new(Δs::PrunedFIs, Δ::V, w::Real) where {V}
    if iszero(w)
        return StochasticAD.similar_empty(Δs, V)
    end
    state = PrunedFIsState(Δs.state.pruning_mode)
    state.weight += w
    Δs = PrunedFIs{V}(Δ, state)
    return Δs
end

### Create Δs backend for the first stochastic triple of computation

function StochasticAD.create_Δs(backend::PrunedFIsBackend, V)
    PrunedFIs{V}(PrunedFIsState(backend.pruning_mode, false))
end

### Convert type of a backend

function Base.convert(::Type{<:PrunedFIs{V}}, Δs::PrunedFIs) where {V}
    PrunedFIs{V}(convert(V, Δs.Δ), Δs.state)
end

### Getting information about perturbations

# "empty" here means no perturbation or a perturbation that has been pruned away
Base.isempty(Δs::PrunedFIs) = !Δs.state.valid
Base.length(Δs::PrunedFIs) = isempty(Δs) ? 0 : 1
function Base.iszero(Δs::PrunedFIs)
    isempty(Δs) || all(iszero, StochasticAD.structural_iterate(Δs.Δ))
end
Base.iszero(Δs::PrunedFIs{<:Real}) = isempty(Δs) || iszero(Δs.Δ)
Base.iszero(Δs::PrunedFIs{<:Tuple}) = isempty(Δs) || all(iszero.(Δs.Δ))
isapproxzero(Δs::PrunedFIs) = isempty(Δs) || isapprox(Δs.Δ, zero(Δs.Δ))

# we lazily prune, so check if empty first
function pruned_value(Δs::PrunedFIs{V}) where {V}
    isempty(Δs) ? StochasticAD.structural_map(zero, Δs.Δ) : Δs.Δ
end
pruned_value(Δs::PrunedFIs{<:Real}) = isempty(Δs) ? zero(Δs.Δ) : Δs.Δ
pruned_value(Δs::PrunedFIs{<:Tuple}) = isempty(Δs) ? zero.(Δs.Δ) : Δs.Δ
pruned_value(Δs::PrunedFIs{<:AbstractArray}) = isempty(Δs) ? zero.(Δs.Δ) : Δs.Δ

StochasticAD.derivative_contribution(Δs::PrunedFIs) = pruned_value(Δs) * Δs.state.weight
function StochasticAD.perturbations(Δs::PrunedFIs)
    ((; Δ = pruned_value(Δs), weight = Δs.state.weight, state = Δs.state),)
end

### Unary propagation

function StochasticAD.weighted_map_Δs(f, Δs::PrunedFIs; kwargs...)
    Δ_out, weight_out = f(pruned_value(Δs), Δs.state)
    # TODO: we could add a direct overload for map_Δs that elides the below line
    Δs.state.weight *= weight_out
    PrunedFIs(Δ_out, Δs.state)
end

StochasticAD.alltrue(f, Δs::PrunedFIs) = f(pruned_value(Δs))

### Coupling

function StochasticAD.get_rep(FIs::Type{<:PrunedFIs}, Δs_all)
    return empty(FIs) #StochasticAD.get_any(Δs_all)
end

function get_pruned_state(Δs_all; Δ_func = nothing, rep, out_rep = nothing)
    if !isnothing(Δ_func) && isnothing(out_rep)
        error("Specifying Δ_func requires out_rep to be specified.")
    end
    function op(cur_state, Δs)
        # lazy pruning optimization temporarily disabled with custom Δ_func 
        # (because custom Δ_func's may prefer not to lazily prune)
        (isnothing(Δ_func) && isapproxzero(Δs)) && return cur_state
        candidate_state = Δs.state
        if !candidate_state.valid ||
           (candidate_state == cur_state)
            return cur_state
        end
        if !cur_state.valid
            return candidate_state
        end

        # Compute "strength" of each perturbation for pruning proposal
        if !isnothing(Δ_func)
            # TODO: structural_map for each state can take asymptotically more time than necessary when combining many distinct states
            candidate_Δ = StochasticAD.structural_map(
                Base.Fix2(StochasticAD.filter_state, candidate_state), Δs_all)
            candidate_Δ_func::Float64 = Δ_func(candidate_Δ, candidate_state, out_rep)
            cur_Δ = StochasticAD.structural_map(
                Base.Fix2(StochasticAD.filter_state, cur_state), Δs_all)
            cur_Δ_func::Float64 = Δ_func(cur_Δ, cur_state, out_rep)
        else
            candidate_Δ_func = 1.0
            cur_Δ_func = 1.0
        end
        candidate_intrinsic_strength = Δs.state.pruning_mode isa Val{:wins} ?
                                       candidate_state.wins : abs(candidate_state.weight)
        cur_intrinsic_strength = Δs.state.pruning_mode isa Val{:wins} ? cur_state.wins :
                                 abs(cur_state.weight)
        candidate_strength = candidate_intrinsic_strength * candidate_Δ_func
        cur_strength = cur_intrinsic_strength * cur_Δ_func

        both_states_bad = iszero(candidate_strength) && iszero(cur_strength)
        if both_states_bad
            cur_state.valid = false
            candidate_state.valid = false
            return cur_state
        end

        # Prune between perturbations
        total_strength = cur_strength + candidate_strength
        p = candidate_strength / total_strength
        if isone(p) || (rand(StochasticAD.RNG) < p)
            cur_state.valid = false
            if Δs.state.pruning_mode isa Val{:wins}
                candidate_state.wins += 1
            end
            candidate_state.weight *= 1 / p
            return candidate_state
        else
            candidate_state.valid = false
            if Δs.state.pruning_mode isa Val{:wins}
                cur_state.wins += 1
            end
            cur_state.weight *= 1 / (1 - p)
            return cur_state
        end
    end
    dummy_state = PrunedFIsState(rep.state.pruning_mode, false) # For type stability, as well as retval if no better state found. TODO: can this be avoided?
    _new_state = foldl(op, StochasticAD.structural_iterate(Δs_all); init = dummy_state)
    return _new_state::PrunedFIsState
end

# for pruning, coupling amounts to getting rid of perturbed values that have been
# lazily kept around even after (aggressive or lazy) pruning made the perturbation invalid.
function StochasticAD.couple(
        FIs::Type{<:PrunedFIs}, Δs_all; rep = StochasticAD.get_rep(FIs, Δs_all),
        out_rep = nothing, Δ_func = nothing, kwargs...)
    state = get_pruned_state(Δs_all; rep, Δ_func)
    Δ_coupled = StochasticAD.structural_map(pruned_value, Δs_all) # TODO: perhaps a performance optimization possible here
    PrunedFIs(Δ_coupled, state)
end

# basically couple combined with a sum.
function StochasticAD.combine(
        FIs::Type{<:PrunedFIs}, Δs_all; rep = StochasticAD.get_rep(FIs, Δs_all),
        Δ_func = nothing, out_rep = nothing, kwargs...)
    state = get_pruned_state(Δs_all;
        rep,
        out_rep,
        Δ_func = !isnothing(Δ_func) ? (Δ, state, val) -> Δ_func(sum(Δ), state, val) :
                 Δ_func)
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

function StochasticAD.similar_type(::Type{<:PrunedFIs{V0, M}}, V::Type) where {V0, M}
    PrunedFIs{V, M}
end
StochasticAD.valtype(::Type{<:PrunedFIs{V}}) where {V} = V

function Base.show(io::IO, Δs::PrunedFIs{V}) where {V}
    print(io, "$(pruned_value(Δs)) with probability $(Δs.state.weight)ε")
end

end
