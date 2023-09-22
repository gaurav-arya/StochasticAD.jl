module AbstractWrapperFIsModule

import ..StochasticAD

export AbstractWrapperFIs

"""
    AbstractWrapperFIs{V, FIs} <: StochasticAD.AbstractFIs{V}

A convenience type for backend strategies that wrap another backend. A subtype `WrapperFIs <: AbstractWrapperFIs`
should have a field called Δs containing the wrapped backend, and should also define the following methods:
* `StochasticAD.similar_type(::Type{<:WrapperFIs}, V, FIs)`: return the type of a new
    `WrapperFIs` with value type `V` and wrapped backend type `FIs`,
* `AbstractWrapperFIsModule.reconstruct_wrapper(wrapper_Δs::WrapperFIs, Δs::AbstractFIs)`: construct
a new `WrapperFIs` wrapping `Δs` given an existing wrapped instace `wrapper_Δs`. 
* `AbstractWrapperFIsModule.reconstruct_wrapper(::Type{<:WrapperFIs}, Δs::AbstractFIs)`: construct
a new `WrapperFIs` wrapping `Δs` given the type of an existing `WrapperFIs`.

Then, all other methods will generically be forwarded to the inner backend, except those overloaded by the
specific wrapper type.
"""
abstract type AbstractWrapperFIs{V, FIs} <: StochasticAD.AbstractFIs{V} end

function reconstruct_wrapper end

function StochasticAD.similar_new(Δs::AbstractWrapperFIs, Δ, w)
    reconstruct_wrapper(Δs, StochasticAD.similar_new(Δs.Δs, Δ, w))
end
function StochasticAD.similar_empty(Δs::AbstractWrapperFIs, V)
    reconstruct_wrapper(Δs, StochasticAD.similar_empty(Δs.Δs, V))
end

function StochasticAD.similar_type(WrapperFIs::Type{<:AbstractWrapperFIs{V0, FIs}},
    V) where {V0, FIs}
    return StochasticAD.similar_type(WrapperFIs, V, StochasticAD.similar_type(FIs, V))
end

StochasticAD.valtype(Δs::AbstractWrapperFIs) = StochasticAD.valtype(Δs.Δs)

function StochasticAD.couple(WrapperFIs::Type{<:AbstractWrapperFIs{V, FIs}},
    Δs_all;
    kwargs...) where {V, FIs}
    _Δs_all = StochasticAD.structural_map(Δs -> Δs.Δs, Δs_all)
    return reconstruct_wrapper(StochasticAD.get_any(Δs_all),
        StochasticAD.couple(FIs, _Δs_all; kwargs...))
end

function StochasticAD.combine(WrapperFIs::Type{<:AbstractWrapperFIs{V, FIs}},
    Δs_all;
    kwargs...) where {V, FIs}
    _Δs_all = StochasticAD.structural_map(Δs -> Δs.Δs, Δs_all)
    return reconstruct_wrapper(StochasticAD.get_any(Δs_all),
        StochasticAD.combine(FIs, _Δs_all; kwargs...))
end

function StochasticAD.get_rep(WrapperFIs::Type{<:AbstractWrapperFIs{V, FIs}},
    Δs_all;
    kwargs...) where {V, FIs}
    _Δs_all = StochasticAD.structural_map(Δs -> Δs.Δs, Δs_all)
    return reconstruct_wrapper(StochasticAD.get_any(Δs_all),
        StochasticAD.get_rep(FIs, _Δs_all; kwargs...))
end

function StochasticAD.scalarize(Δs::AbstractWrapperFIs; kwargs...)
    return StochasticAD.structural_map(StochasticAD.scalarize(Δs.Δs; kwargs...)) do _Δs
        reconstruct_wrapper(Δs, _Δs)
    end
end

function StochasticAD.derivative_contribution(Δs::AbstractWrapperFIs, Δs_all; kwargs...)
    StochasticAD.derivative_contribution(Δs.Δs, Δs_all; kwargs...)
end

StochasticAD.alltrue(f, Δs::AbstractWrapperFIs) = StochasticAD.alltrue(f, Δs.Δs)

StochasticAD.perturbations(Δs::AbstractWrapperFIs) = StochasticAD.perturbations(Δs.Δs)

function StochasticAD.filter_state(Δs::AbstractWrapperFIs, state)
    StochasticAD.filter_state(Δs.Δs, state)
end

function StochasticAD.map_Δs(f, Δs::AbstractWrapperFIs; kwargs...)
    reconstruct_wrapper(Δs, StochasticAD.map_Δs(f, Δs.Δs; kwargs...))
end

StochasticAD.new_Δs_strategy(Δs::AbstractWrapperFIs) = StochasticAD.new_Δs_strategy(Δs.Δs)

function Base.empty(WrapperFIs::Type{<:AbstractWrapperFIs{V, FIs}}) where {V, FIs}
    return reconstruct_wrapper(WrapperFIs, empty(FIs))
end

Base.empty(Δs::AbstractWrapperFIs) = reconstruct_wrapper(Δs, empty(Δs.Δs))
Base.isempty(Δs::AbstractWrapperFIs) = isempty(Δs.Δs)
Base.length(Δs::AbstractWrapperFIs) = length(Δs.Δs)
Base.iszero(Δs::AbstractWrapperFIs) = iszero(Δs.Δs)

function StochasticAD.scale(Δs::AbstractWrapperFIs, a)
    reconstruct_wrapper(Δs, StochasticAD.scale(Δs.Δs, a))
end

function StochasticAD.derivative_contribution(Δs::AbstractWrapperFIs)
    StochasticAD.derivative_contribution(Δs.Δs)
end

function (::Type{<:AbstractWrapperFIs{V}})(Δs::AbstractWrapperFIs) where {V}
    reconstruct_wrapper(Δs, StochasticAD.similar_type(typeof(Δs.Δs), V)(Δs.Δs))
end

function Base.show(io::IO, Δs::AbstractWrapperFIs)
    return show(io, Δs.Δs)
end

end
