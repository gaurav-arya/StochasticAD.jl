module SmoothedFIsBackend

import ..StochasticAD

export SmoothedFIs

"""
    SmoothedFIs{V} <: StochasticAD.AbstractFIs{V}

A backend that smooths perturbations together.
The full backend interface is not supported, rather only the functions necessary for defining chain rules.
"""
struct SmoothedFIs{V, Vfloat} <: StochasticAD.AbstractFIs{V}
    δ::Vfloat
end

SmoothedFIs{V}(δ::Vfloat) where {V, Vfloat} = SmoothedFIs{V, Vfloat}(δ)

StochasticAD.similar_empty(::SmoothedFIs, V::Type) = SmoothedFIs{V}(zero(float(V)))
function StochasticAD.similar_new(::SmoothedFIs, Δ::V, w::Real) where {V}
    SmoothedFIs{V}(float(Δ) * w)
end

function StochasticAD.combine(::Type{<:SmoothedFIs}, Δs_all; rep = nothing)
    Δ_combined = sum(Δs -> Δs.δ, StochasticAD.structural_iterate(Δs_all))
    # TODO: using eltype below will not work in general, and the proper fix
    # could be a caller-provided type. This is not yet needed for this function's
    # limited internal use.
    eltype(Δs_all)(Δ_combined)
end

StochasticAD.derivative_contribution(Δs::SmoothedFIs) = Δs.δ

function Base.show(io::IO, mime::MIME"text/plain", Δs::SmoothedFIs)
    print(io, "$(Δs.δ)ε")
end

end
