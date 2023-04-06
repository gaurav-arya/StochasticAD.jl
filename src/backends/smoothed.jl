module SmoothedFIsModule

import ..StochasticAD

export SmoothedFIsBackend, SmoothedFIs

"""
    SmoothedFIsBackend <: StochasticAD.AbstractFIsBackend

A backend algorithm that smooths perturbations togethers. 
"""
struct SmoothedFIsBackend <: StochasticAD.AbstractFIsBackend end

"""
    SmoothedFIs{V} <: StochasticAD.AbstractFIs{V}

The implementing backend structure for SmoothedFIsBackend.
"""
# TODO: make type of δ generic
struct SmoothedFIs{V, V_float} <: StochasticAD.AbstractFIs{V}
    δ::V_float
    function SmoothedFIs{V}(δ) where {V}
        # hardcode Float64 representation for now, for simplicity.
        δ_f64 = StochasticAD.structural_map(Base.Fix1(convert, Float64), δ)
        return new{V, typeof(δ_f64)}(δ_f64)
    end
end

### Empty / no perturbation

StochasticAD.similar_empty(::SmoothedFIs, V::Type) = SmoothedFIs{V}(0.0)
Base.empty(::Type{<:SmoothedFIs{V}}) where {V} = SmoothedFIs{V}(0.0)
Base.empty(Δs::SmoothedFIs) = empty(typeof(Δs))

### Create a new perturbation with infinitesimal probability

function StochasticAD.similar_new(::SmoothedFIs, Δ::V, w::Real) where {V}
    SmoothedFIs{V}(Δ * w)
end

StochasticAD.new_Δs_strategy(::SmoothedFIs) = StochasticAD.TwoSidedStrategy()

### Scale a perturbation

function StochasticAD.scale(Δs::SmoothedFIs{V}, scale::Real) where {V}
    SmoothedFIs{V}(Δs.δ * scale)
end

### Create Δs backend for the first stochastic triple of computation

StochasticAD.create_Δs(::SmoothedFIsBackend, V) = SmoothedFIs{V}(0.0)

### Convert type of a backend

function (::Type{<:SmoothedFIs{V}})(Δs::SmoothedFIs) where {V}
    SmoothedFIs{V}(Δs.δ)
end
(::Type{SmoothedFIs{V}})(Δs::SmoothedFIs) where {V} = SmoothedFIs{V}(Δs.δ)

### Getting information about perturbations

Base.isempty(Δs::SmoothedFIs) = false
Base.iszero(Δs::SmoothedFIs) = iszero(Δs.δ)
Base.iszero(Δs::SmoothedFIs{<:Tuple}) = all(iszero.(Δs.δ))
StochasticAD.derivative_contribution(Δs::SmoothedFIs) = Δs.δ

### Unary propagation

function StochasticAD.map_Δs(f, Δs::SmoothedFIs; deriv, out_rep)
    SmoothedFIs{typeof(out_rep)}(deriv(Δs.δ))
end

StochasticAD.alltrue(f, Δs::SmoothedFIs) = true

### Coupling

StochasticAD.get_rep(::Type{<:SmoothedFIs}, Δs_all) = first(Δs_all)

function StochasticAD.couple(::Type{<:SmoothedFIs}, Δs_all; rep = nothing, out_rep)
    SmoothedFIs{typeof(out_rep)}(StochasticAD.structural_map(Δs -> Δs.δ, Δs_all))
end

function StochasticAD.combine(::Type{<:SmoothedFIs}, Δs_all; rep = nothing)
    V_out = StochasticAD.valtype(first(StochasticAD.structural_iterate(Δs_all)))
    Δ_combined = sum(Δs -> Δs.δ, StochasticAD.structural_iterate(Δs_all))
    SmoothedFIs{V_out}(Δ_combined)
end

function StochasticAD.scalarize(Δs::SmoothedFIs; out_rep)
    return StochasticAD.structural_map(out_rep, Δs.δ) do out, δ
        return SmoothedFIs{typeof(out)}(δ)
    end
end

### Miscellaneous

StochasticAD.similar_type(::Type{<:SmoothedFIs}, V::Type) = SmoothedFIs{V}
StochasticAD.valtype(::Type{<:SmoothedFIs{V}}) where {V} = V

function Base.show(io::IO, Δs::SmoothedFIs)
    print(io, "$(Δs.δ)ε")
end

end
