## 
"""
    AbstractFIsBackend

An abstract type for backend strategies of Finite perturbations that occur with Infinitesimal probability (FIs).
"""
abstract type AbstractFIsBackend end

"""
    AbstractFIs{V}

An abstract type for concrete backend representations of Finite Infinitesimals. 
"""
abstract type AbstractFIs{V} end

### Some of the necessary interface notes below.
# TODO: document

function create_Δs end

function similar_new end
function similar_empty end
function similar_type end

valtype(Δs::AbstractFIs) = valtype(typeof(Δs))

# TODO: typeof ∘ first is a loose check, should make more robust.
# Done rather than eltype to avoid unnecessary unions over the type params, when we ultimately
# only essentially care about the parameterless type.
couple(Δs_all; kwargs...) = couple(typeof(first(Δs_all)), Δs_all; kwargs...) 
combine(Δs_all; kwargs...) = combine(typeof(first(Δs_all)), Δs_all; kwargs...)
get_rep(Δs_all; kwargs...) = get_rep(typeof(first(Δs_all)), Δs_all; kwargs...)
function scalarize end

function derivative_contribution end

function alltrue end

function perturbations end

function filter_state end

function map_Δs end
function Base.map(f, Δs::AbstractFIs; kwargs...)
    StochasticAD.map_Δs((Δs, _) -> f(Δs), Δs; kwargs...)
end

function new_Δs_strategy end

# Currently only supported / thought through for SmoothedFIs.
function scale end

# utility function useful e.g. for get_rep in some backends
function get_any(Δs_all)
    # The code below is a bit ridiculous, but it's faster than `first` for small structures:)
    foldl((Δs1, Δs2) -> Δs1, StochasticAD.structural_iterate(Δs_all))
end

### Strategies

abstract type AbstractPerturbationStrategy end
Base.empty(S::Type{<:AbstractPerturbationStrategy}) = S() # TODO: handle strategies containing data.

struct StrategyWrapperBackend{B <: AbstractFIsBackend, S <: AbstractPerturbationStrategy} <:
       AbstractFIsBackend
    backend::B
    strategy::S
end

struct StrategyWrapperFIs{V, FIs <: AbstractFIs{V}, S <: AbstractPerturbationStrategy} <:
       AbstractFIs{V}
    Δs::FIs
    strategy::S
end

## wrap the full interface, manually for now:)

function create_Δs(backend::StrategyWrapperBackend, V)
    return StrategyWrapperFIs(create_Δs(backend.backend, V), backend.strategy)
end

function similar_new(Δs::StrategyWrapperFIs, Δ, w)
    return StrategyWrapperFIs(similar_new(Δs.Δs, Δ, w), Δs.strategy)
end

function similar_empty(Δs::StrategyWrapperFIs, V)
    return StrategyWrapperFIs(similar_empty(Δs.Δs, V), Δs.strategy)
end

function similar_type(::Type{<:StrategyWrapperFIs{V0, FIs, S}}, V) where {V0, FIs, S}
    return StrategyWrapperFIs{V, similar_type(FIs, V), S}
end

function valtype(Δs::StrategyWrapperFIs)
    return valtype(Δs.Δs)
end

function couple(::Type{<:StrategyWrapperFIs{V, FIs}}, Δs_all; kwargs...) where {V, FIs}
    _Δs_all = structural_map(Δs -> Δs.Δs, Δs_all)
    return StrategyWrapperFIs(couple(FIs, _Δs_all; kwargs...), get_any(Δs_all).strategy)
end

function combine(::Type{<:StrategyWrapperFIs{V, FIs}}, Δs_all; kwargs...) where {V, FIs}
    _Δs_all = structural_map(Δs -> Δs.Δs, Δs_all)
    return StrategyWrapperFIs(combine(FIs, _Δs_all; kwargs...), get_any(Δs_all).strategy)
end

function get_rep(::Type{<:StrategyWrapperFIs{V, FIs}}, Δs_all; kwargs...) where {V, FIs}
    _Δs_all = structural_map(Δs -> Δs.Δs, Δs_all)
    return StrategyWrapperFIs(get_rep(FIs, _Δs_all; kwargs...), get_any(Δs_all).strategy)
end

function scalarize(Δs::StrategyWrapperFIs; kwargs...)
    return structural_map(scalarize(Δs.Δs; kwargs...)) do _Δs
        StrategyWrapperFIs(_Δs, Δs.strategy)
    end
end

function derivative_contribution(Δs::StrategyWrapperFIs, Δs_all; kwargs...)
    return derivative_contribution(Δs.Δs, Δs_all; kwargs...)
end

function alltrue(f, Δs::StrategyWrapperFIs)
    return alltrue(f, Δs.Δs)
end

function perturbations(Δs::StrategyWrapperFIs)
    return perturbations(Δs.Δs)
end

function filter_state(Δs::StrategyWrapperFIs, state)
    return filter_state(Δs.Δs, state)
end

function map_Δs(f, Δs::StrategyWrapperFIs; kwargs...)
    return StrategyWrapperFIs(map_Δs(f, Δs.Δs; kwargs...), Δs.strategy)
end

function new_Δs_strategy(Δs::StrategyWrapperFIs)
    return Δs.strategy
end

function Base.empty(::Type{<:StrategyWrapperFIs{V, FIs, S}}) where {V, FIs, S}
    return StrategyWrapperFIs(empty(FIs), empty(S)) 
end

function Base.empty(Δs::StrategyWrapperFIs)
    return StrategyWrapperFIs(empty(Δs.Δs), Δs.strategy)
end

function Base.isempty(Δs::StrategyWrapperFIs)
    return isempty(Δs.Δs)
end

function Base.length(Δs::StrategyWrapperFIs)
    return length(Δs.Δs)
end

function Base.iszero(Δs::StrategyWrapperFIs)
    return iszero(Δs.Δs)
end

function scale(Δs::StrategyWrapperFIs, a)
    return StrategyWrapperFIs(scale(Δs.Δs, a), Δs.strategy)
end

derivative_contribution(Δs::StrategyWrapperFIs) = derivative_contribution(Δs.Δs)

function (::Type{<:StrategyWrapperFIs{V}})(Δs::StrategyWrapperFIs) where {V}
    StrategyWrapperFIs(similar_type(typeof(Δs.Δs), V)(Δs.Δs), Δs.strategy)
end

function Base.show(io::IO, Δs::StrategyWrapperFIs)
    return show(io, Δs.Δs)
end
