## 
"""
    AbstractFIs{V}

An abstract type for backend representations of Finite perturbations that occur with Infinitesimal probability (FIs).
"""
abstract type AbstractFIs{V} end

### Some of the necessary interface notes below.
# TODO: document

function similar_new end
function similar_empty end
function similar_type end

get_valtype(Δs::AbstractFIs) = get_valtype(typeof(Δs))

couple(Δs_all; kwargs...) = couple(eltype(Δs_all), Δs_all; kwargs...)
combine(Δs_all; kwargs...) = combine(eltype(Δs_all), Δs_all; kwargs...)
get_rep(Δs_all; kwargs...) = get_rep(eltype(Δs_all), Δs_all; kwargs...)

function derivative_contribution end

function alltrue end

function perturbations end
