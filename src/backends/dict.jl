module DictFIsBackend

export DictFIs

import ..StochasticAD
using Dictionaries

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
Base.:isless(event1::InfinitesimalEvent, event2::InfinitesimalEvent) = event1 < event2

"""
    DictFIs{V} <: StochasticAD.AbstractFIs{V}

A dictionary backend which keeps entries for each perturbation that has occurred without pruning. 
Currently very unoptimized.
"""
struct DictFIs{V} <: StochasticAD.AbstractFIs{V}
    dict::Dictionary{InfinitesimalEvent, V}
    state::DictFIsState
end

state(Δs::DictFIs) = Δs.state

### Empty / no perturbation

function DictFIs{V}(state::DictFIsState) where {V}
    DictFIs{V}(Dictionary{InfinitesimalEvent, V}(), state)
end
StochasticAD.similar_empty(Δs::DictFIs, V::Type) = DictFIs{V}(Δs.state)
Base.empty(Δs::DictFIs{V}) where {V} = similar_empty(Δs::DictFIs, V::Type) # abstracted away
function Base.empty(::Type{<:DictFIs{V}}) where {V}
    DictFIs{V}(DictFIsState(false))
end

### Create a new perturbation with infinitesimal probability

function new_perturbation(Δ::V, w::Real, state::DictFIsState) where {V}
    state.tag_count += 1
    event = InfinitesimalEvent(state.tag_count, w)
    DictFIs{V}(Dictionary([event], [Δ]), state)
end
function StochasticAD.similar_new(Δs::DictFIs, Δ::V, w::Real) where {V}
    new_perturbation(Δ, w, Δs.state)
end

### Create Δs backend for the first stochastic triple of computation

DictFIs{V}() where {V} = DictFIs{V}(DictFIsState())

### Convert type of a backend

function DictFIs{V}(Δs::DictFIs) where {V}
    DictFIs{V}(convert(Dictionary{InfinitesimalEvent, V}, Δs.dict), Δs.state)
end

### Getting information about Δs

Base.isempty(Δs::DictFIs) = isempty(Δs.dict)
Base.length(Δs::DictFIs) = length(Δs.dict)
Base.iszero(Δs::DictFIs) = isempty(Δs) || all(iszero.(Δs.dict))
function StochasticAD.derivative_contribution(Δs::DictFIs{V}) where {V}
    sum((Δ * event.w for (event, Δ) in pairs(Δs.dict)), init = zero(V))
end

perturbations(Δs::DictFIs) = [(Δ, event.w) for (event, Δ) in pairs(Δs.dict)]

### Unary propagation

function Base.map(f, Δs::DictFIs)
    dict = Dictionary(keys(Δs.dict), map(f, collect(Δs.dict)))
    DictFIs(dict, Δs.state)
end

StochasticAD.alltrue(Δs::DictFIs{Bool}) = all(Δs.dict)

### Coupling

function StochasticAD.get_rep(::Type{<:DictFIs}, Δs_all)
    for Δs in Δs_all
        if Δs.state.valid
            return Δs
        end
    end
    return first(Δs_all)
end

function StochasticAD.couple(::Type{<:DictFIs}, Δs_all; rep = StochasticAD.get_rep(Δs_all))
    all_keys = union(keys.(getfield.(Δs_all, :dict))...)
    Δs_coupled_dict = [map(Δs -> isassigned(Δs.dict, key) ? Δs.dict[key] :
                                 zero(eltype(Δs.dict)), Δs_all) for key in all_keys]
    DictFIs(Dictionary(all_keys, Δs_coupled_dict), rep.state)
end

function StochasticAD.combine(::Type{<:DictFIs}, Δs_all; rep = get_rep(Δs_all))
    Δs_combined_dict = reduce((Δs1_dict, Δs2_dict) -> mergewith(+, Δs1_dict, Δs2_dict),
                              getfield.(Δs_all, :dict))
    DictFIs(Δs_combined_dict, rep.state)
end

### Miscellaneous

StochasticAD.similar_type(::Type{<:DictFIs}, V::Type) = DictFIs{V}

end
