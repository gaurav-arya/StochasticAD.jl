# TODO: make this a module, with the interface exported?

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
# TODO: perhaps deprecate these methods in favor of an explicit first argument?
couple(Δs_all; kwargs...) = couple(typeof(first(Δs_all)), Δs_all; kwargs...)
combine(Δs_all; kwargs...) = combine(typeof(first(Δs_all)), Δs_all; kwargs...)
get_rep(Δs_all; kwargs...) = get_rep(typeof(first(Δs_all)), Δs_all; kwargs...)
function scalarize end

function derivative_contribution end

function alltrue end

function perturbations end

function filter_state end

function weighted_map_Δs end
function map_Δs(f, Δs::AbstractFIs; kwargs...)
    StochasticAD.weighted_map_Δs((Δs, state) -> (f(Δs, state), 1.0), Δs; kwargs...)
end
function Base.map(f, Δs::AbstractFIs; kwargs...)
    StochasticAD.map_Δs((Δs, _) -> f(Δs), Δs; kwargs...)
end
# We also add a scale to deriv for scaling smoothed perturbations 
function scale(Δs::AbstractFIs, a::Real)
    StochasticAD.weighted_map_Δs((Δ, state) -> (Δ, a),
        Δs;
        deriv = Base.Fix1(*, a),
        out_rep = Δs)
end

function new_Δs_strategy end

# utility function useful e.g. for get_rep in some backends
function get_any(Δs_all)
    # The code below is a bit ridiculous, but it's faster than `first` for small structures:)
    foldl((Δs1, Δs2) -> Δs1, StochasticAD.structural_iterate(Δs_all))
end

abstract type AbstractPerturbationStrategy end

abstract type AbstractPerturbationSignal end

function send_signal end
