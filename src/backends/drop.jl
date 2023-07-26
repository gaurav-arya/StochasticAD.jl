module DropFIsModule

import ..StochasticAD

export DropFIsBackend, DropFIs

"""
    DropFIsBackend <: StochasticAD.AbstractFIsBackend

A backend algorithm that smooths perturbations togethers. 
"""
struct DropFIsBackend <: StochasticAD.AbstractFIsBackend end

"""
    DropFIs{V} <: StochasticAD.AbstractFIs{V}

The implementing backend structure for DropFIsBackend.
"""
# TODO: make type of δ generic
struct DropFIs{V, V_float} <: StochasticAD.AbstractFIs{V}
    δ::V_float
    function DropFIs{V}(δ) where {V}
        # hardcode Float64 representation for now, for simplicity.
        δ_f64 = StochasticAD.structural_map(Base.Fix1(convert, Float64), δ)
        return new{V, typeof(δ_f64)}(δ_f64)
    end
end

### Empty / no perturbation

StochasticAD.similar_empty(::DropFIs, V::Type) = DropFIs{V}(0.0)
Base.empty(::Type{<:DropFIs{V}}) where {V} = DropFIs{V}(0.0)
Base.empty(Δs::DropFIs) = empty(typeof(Δs))

### Create a new perturbation with infinitesimal probability

function StochasticAD.similar_new(::DropFIs, Δ::V, w::Real) where {V}
    DropFIs{V}(Δ * w)
end

StochasticAD.new_Δs_strategy(::DropFIs) = StochasticAD.IgnoreDiscreteStrategy()

### Scale a perturbation

function StochasticAD.scale(Δs::DropFIs{V}, scale::Real) where {V}
    DropFIs{V}(Δs.δ * scale)
end

### Create Δs backend for the first stochastic triple of computation

StochasticAD.create_Δs(::DropFIsBackend, V) = DropFIs{V}(0.0)

### Convert type of a backend

function (::Type{<:DropFIs{V}})(Δs::DropFIs) where {V}
    DropFIs{V}(Δs.δ)
end
(::Type{DropFIs{V}})(Δs::DropFIs) where {V} = DropFIs{V}(Δs.δ)

### Getting information about perturbations

Base.isempty(Δs::DropFIs) = true
Base.iszero(Δs::DropFIs) = iszero(Δs.δ)
Base.iszero(Δs::DropFIs{<:Tuple}) = all(iszero.(Δs.δ))
StochasticAD.derivative_contribution(Δs::DropFIs) = Δs.δ

### Unary propagation

function StochasticAD.map_Δs(f, Δs::DropFIs; deriv, out_rep)
    DropFIs{typeof(out_rep)}(deriv(Δs.δ))
end

StochasticAD.alltrue(f, Δs::DropFIs) = true

### Coupling

StochasticAD.get_rep(::Type{<:DropFIs}, Δs_all) = first(Δs_all)

function StochasticAD.couple(::Type{<:DropFIs}, Δs_all; rep = nothing, out_rep)
    DropFIs{typeof(out_rep)}(StochasticAD.structural_map(Δs -> Δs.δ, Δs_all))
end

function StochasticAD.combine(::Type{<:DropFIs}, Δs_all; rep = nothing)
    V_out = StochasticAD.valtype(first(StochasticAD.structural_iterate(Δs_all)))
    Δ_combined = sum(Δs -> Δs.δ, StochasticAD.structural_iterate(Δs_all))
    DropFIs{V_out}(Δ_combined)
end

function StochasticAD.scalarize(Δs::DropFIs; out_rep)
    return StochasticAD.structural_map(out_rep, Δs.δ) do out, δ
        return DropFIs{typeof(out)}(δ)
    end
end

### Miscellaneous

StochasticAD.similar_type(::Type{<:DropFIs}, V::Type) = DropFIs{V}
StochasticAD.valtype(::Type{<:DropFIs{V}}) where {V} = V

function Base.show(io::IO, Δs::DropFIs)
    print(io, "")
end

end
