module StochasticAD

### Public API

export stochastic_triple, derivative_contribution, perturbations, smooth_triple, dual_number # For working with stochastic triples
export derivative_estimate, StochasticModel, stochastic_gradient # Higher level functionality
export new_weight # Particle resampling
export PrunedFIsBackend, PrunedFIsAggressiveBackend, DictFIsBackend, SmoothedFIsBackend

### Imports

using Random
using Distributions
using DistributionsAD
using ChainRulesCore
using ChainRulesOverloadGeneration
using ExprTools
using ForwardDiff
using Functors
import ChainRulesCore
# resolve conflicts while this code exists in both.
const on_new_rule = ChainRulesOverloadGeneration.on_new_rule
const refresh_rules = ChainRulesOverloadGeneration.refresh_rules

const RNG = copy(Random.default_rng())

### Files responsible for backends

include("finite_infinitesimals.jl")

""" 
    StochasticTriple{T, V <: Real, FIs <: AbstractFIs{V}}

Stores the primal value of the computation, alongside a "dual" component
representing an infinitesimal change, and a "triple" component that tracks
discrete change(s) with infinitesimal probability. 

Pretty printed as "value + δε + ({pretty print of Δs})".

## Constructor

- `value`: the primal value.
- `δ`: the value of the almost-sure derivative, i.e. the rate of "infinitesimal" change.
- `Δs`: alternate values with associated weights, i.e. Finite perturbations with Infinitesimal probability,
        represented by a backend `FIs <: AbstractFIs`.
"""
struct StochasticTriple{T, V <: Real, FIs <: AbstractFIs{V}} <: Real
    value::V
    δ::V # infinitesimal change
    Δs::FIs # finite changes with infinitesimal probabilities # (Δ = 3, p = 1*h)
    function StochasticTriple{T, V, FIs}(value::V, δ::V,
        Δs::FIs) where {T, V, FIs <: AbstractFIs{V}}
        new{T, V, FIs}(value, δ, Δs)
    end
end

include("backends/pruned.jl")
include("backends/pruned_aggressive.jl")
include("backends/dict.jl")
include("backends/smoothed.jl")
include("backends/drop.jl")
using .PrunedFIsModule
using .PrunedFIsAggressiveModule
using .DictFIsModule
using .SmoothedFIsModule
using .DropFIsModule

include("prelude.jl") # Defines global constants
include("smoothing.jl") # Smoothing rules. Placed before general rules so that new_weight frule is caught by overload generation.
include("stochastic_triple.jl") # Defines stochastic triple object and higher level functions
include("general_rules.jl") # Defines rules for propagation through deterministic functions
include("discrete_randomness.jl") # Defines rules for propagation through discrete random functions
include("propagate.jl") # Experimental generalized forward propagation functionality
include("misc.jl") # Miscellaneous functions that do not fit in the usual flow

end
