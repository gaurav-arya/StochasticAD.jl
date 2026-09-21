module DictFIsModule

export DictFIsBackend, DictFIs

import ..StochasticAD
using Dictionaries

"""
    DictFIsBackend <: StochasticAD.AbstractFIsBackend

A dictionary backend algorithm which keeps entries for each perturbation that has occurred without pruning. 
An index tracks the alternatives that have not recoupled, so backend operations do not
need to revisit the retained values of dead alternatives.
"""
struct DictFIsBackend <: StochasticAD.AbstractFIsBackend end

"""
    DictFIsState    

State maintained by dictionary backend.
"""
mutable struct DictFIsState
    tag_count::Int64
    valid::Bool
    DictFIsState(valid = true) = new(0, valid)
end

struct InfinitesimalEvent
    tag::Any # unique identifier
    w::Float64 # weight (infinitesimal probability wε) 
end

Base.:<(event1::InfinitesimalEvent, event2::InfinitesimalEvent) = event1.tag < event2.tag
function Base.:(==)(event1::InfinitesimalEvent, event2::InfinitesimalEvent)
    event1.tag == event2.tag
end
function Base.isequal(event1::InfinitesimalEvent, event2::InfinitesimalEvent)
    isequal(event1.tag, event2.tag)
end
Base.hash(event::InfinitesimalEvent, hsh::UInt) = hash(event.tag, hsh)
Base.:isless(event1::InfinitesimalEvent, event2::InfinitesimalEvent) = event1 < event2

"""
    DictFIs{V} <: StochasticAD.AbstractFIs{V}

The implementing backend structure for DictFIsBackend.
"""
struct DictFIs{V} <: StochasticAD.AbstractFIs{V}
    dict::Dictionary{InfinitesimalEvent, V}
    # The dictionary may retain recoupled values for compatibility with perturbation inspection.
    active::Indices{InfinitesimalEvent}
    state::DictFIsState
end

state(Δs::DictFIs) = Δs.state

function _iszero_value(value)
    all(iszero, StochasticAD.structural_iterate(value))
end

_isactive(event::InfinitesimalEvent, value) =
    !iszero(event.w) && !_iszero_value(value)

function _active_index(events, values)
    active = Indices{InfinitesimalEvent}(sizehint = length(events))
    for (event, value) in zip(events, values)
        _isactive(event, value) && insert!(active, event)
    end
    active
end

function _dictfis(dict::Dictionary{InfinitesimalEvent, V}, active, state) where {V}
    active isa Indices{InfinitesimalEvent} ||
        (active = Indices{InfinitesimalEvent}(active))
    DictFIs{V}(dict, active, state)
end

function _dictfis(dict::Dictionary{InfinitesimalEvent, V}, state) where {V}
    _dictfis(dict, Indices{InfinitesimalEvent}(keys(dict)), state)
end

# Keep the old constructor available for callers that construct DictFIs directly.
function DictFIs(dict::Dictionary{InfinitesimalEvent, V}, state::DictFIsState) where {V}
    _dictfis(dict, state)
end

function DictFIs{V}(dict::Dictionary{InfinitesimalEvent}, state::DictFIsState) where {V}
    _dictfis(convert(Dictionary{InfinitesimalEvent, V}, dict), state)
end

### Empty / no perturbation

function DictFIs{V}(state::DictFIsState) where {V}
    DictFIs{V}(Dictionary{InfinitesimalEvent, V}(), Indices{InfinitesimalEvent}(), state)
end
StochasticAD.similar_empty(Δs::DictFIs, V::Type) = DictFIs{V}(Δs.state)
Base.empty(Δs::DictFIs{V}) where {V} = StochasticAD.similar_empty(Δs::DictFIs, V::Type)
function Base.empty(::Type{<:DictFIs{V}}) where {V}
    DictFIs{V}(DictFIsState(false))
end

### Create a new perturbation with infinitesimal probability

function new_perturbation(Δ::V, w::Real, state::DictFIsState) where {V}
    state.tag_count += 1
    event = InfinitesimalEvent(state.tag_count, w)
    active = iszero(event.w) ? Indices{InfinitesimalEvent}() : Indices([event])
    _dictfis(Dictionary([event], [Δ]), active, state)
end
function StochasticAD.similar_new(Δs::DictFIs, Δ::V, w::Real) where {V}
    new_perturbation(Δ, w, Δs.state)
end

### Create Δs backend for the first stochastic triple of computation

StochasticAD.create_Δs(::DictFIsBackend, V) = DictFIs{V}(DictFIsState())

### Convert type of a backend

function Base.convert(::Type{DictFIs{V}}, Δs::DictFIs) where {V}
    _dictfis(convert(Dictionary{InfinitesimalEvent, V}, Δs.dict), Δs.active, Δs.state)
end

### Getting information about Δs

Base.isempty(Δs::DictFIs) = isempty(Δs.active)
Base.length(Δs::DictFIs) = length(Δs.active)
Base.iszero(Δs::DictFIs) = isempty(Δs) ||
                            all(_iszero_value(Δs.dict[event]) for event in Δs.active)
function StochasticAD.derivative_contribution(Δs::DictFIs{V}) where {V}
    sum((Δs.dict[event] * event.w for event in Δs.active), init = zero(V) * 0.0)
end

function StochasticAD.perturbations(Δs::DictFIs)
    [(; Δ, weight = event.w, state = event) for (event, Δ) in pairs(Δs.dict)]
end

### Unary propagation

function StochasticAD.weighted_map_Δs(f, Δs::DictFIs; kwargs...)
    if isempty(Δs)
        out_rep = get(kwargs, :out_rep, nothing)
        V = (isnothing(out_rep) || out_rep isa StochasticAD.AbstractFIs) ?
            StochasticAD.valtype(Δs) : typeof(out_rep)
        return StochasticAD.similar_empty(Δs, V)
    end
    # Pass key as state in map
    mapped_values_and_weights = [f(Δs.dict[event], event) for event in Δs.active]
    mapped_values = first.(mapped_values_and_weights)
    mapped_weights = last.(mapped_values_and_weights)
    scaled_events = map((event, a) -> InfinitesimalEvent(event.tag, event.w * a),
        Δs.active,
        mapped_weights) # TODO: should original events (with old tag) also be modified?
    dict = Dictionary(scaled_events, mapped_values)
    _dictfis(dict, _active_index(scaled_events, mapped_values), Δs.state)
end

StochasticAD.alltrue(f, Δs::DictFIs) = all(f(Δs.dict[event]) for event in Δs.active)

### Coupling

function StochasticAD.get_rep(::Type{<:DictFIs}, Δs_all)
    for Δs in StochasticAD.structural_iterate(Δs_all)
        if Δs.state.valid
            return Δs
        end
    end
    return first(Δs_all)
end

function StochasticAD.couple(FIs::Type{<:DictFIs}, Δs_all;
        rep = StochasticAD.get_rep(FIs, Δs_all),
        out_rep = nothing,
        kwargs...)
    distinct_keys = Indices{InfinitesimalEvent}()
    for Δs in StochasticAD.structural_iterate(Δs_all)
        union!(distinct_keys, Δs.active)
    end
    Δs_coupled_dict = [StochasticAD.structural_map(
                           Δs -> StochasticAD.filter_state(Δs, key), Δs_all)
                       for key in distinct_keys]
    dict = Dictionary(distinct_keys, Δs_coupled_dict)
    _dictfis(dict, distinct_keys, rep.state)
end

function StochasticAD.combine(FIs::Type{<:DictFIs}, Δs_all;
        rep = StochasticAD.get_rep(FIs, Δs_all), out_rep = nothing, kwargs...)
    Δs_iter = StochasticAD.structural_iterate(Δs_all)
    first_value = nothing
    for Δs in Δs_iter
        if !isempty(Δs.active)
            first_value = Δs.dict[first(Δs.active)]
            break
        end
    end
    if isnothing(first_value)
        V = isnothing(out_rep) || out_rep isa StochasticAD.AbstractFIs ?
            StochasticAD.valtype(rep) : typeof(out_rep)
        return DictFIs{V}(rep.state)
    end

    Δs_combined_dict = Dictionary{InfinitesimalEvent, typeof(first_value)}()
    for Δs in StochasticAD.structural_iterate(Δs_all)
        for event in Δs.active
            if haskey(Δs_combined_dict, event)
                Δs_combined_dict[event] = StochasticAD.structural_map(
                    +, Δs_combined_dict[event], Δs.dict[event])
            else
                insert!(Δs_combined_dict, event, Δs.dict[event])
            end
        end
    end
    active = _active_index(keys(Δs_combined_dict), values(Δs_combined_dict))
    _dictfis(Δs_combined_dict, active, rep.state)
end

function StochasticAD.scalarize(Δs::DictFIs; out_rep = nothing)
    if isempty(Δs)
        isnothing(out_rep) && error("Cannot scalarize empty DictFIs without out_rep")
        return StochasticAD.structural_map(
            x -> StochasticAD.similar_empty(Δs, typeof(x)), out_rep)
    end

    first_event = first(Δs.active)
    first_value = Δs.dict[first_event]
    first_value isa Real && return Δs
    if first_value isa AbstractVector
        events = collect(Δs.active)
        event_values = [Δs.dict[event] for event in events]
        return map(eachindex(first_value)) do index
            coordinate_values = [value[index] for value in event_values]
            dict = Dictionary(events, coordinate_values)
            _dictfis(dict, _active_index(events, coordinate_values), Δs.state)
        end
    end

    # TODO: use vcat here?
    tupleify(Δ1, Δ2) = StochasticAD.structural_map(tuple, Δ1, Δ2)
    Δ_all_allkeys = foldl(tupleify, (Δs.dict[event] for event in Δs.active))
    Δ_all_rep = first_value
    _keys = Δs.active
    return StochasticAD.structural_map(Δ_all_rep, Δ_all_allkeys) do _, Δ_allkeys
        dict = Dictionary(_keys, Δ_allkeys)
        _dictfis(dict, _active_index(_keys, Δ_allkeys), Δs.state)
    end
end

function StochasticAD.filter_state(Δs::DictFIs{V}, key) where {V}
    key in Δs.active ? Δs.dict[key] : zero(V)
end

### Miscellaneous

StochasticAD.similar_type(::Type{<:DictFIs}, V::Type) = DictFIs{V}
StochasticAD.valtype(::Type{<:DictFIs{V}}) where {V} = V

end
