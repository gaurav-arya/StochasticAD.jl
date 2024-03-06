module StrategyWrapperFIsModule

using ..StochasticAD
using ..StochasticAD.AbstractWrapperFIsModule

export StrategyWrapperFIsBackend, StrategyWrapperFIs

struct StrategyWrapperFIsBackend{
    B <: StochasticAD.AbstractFIsBackend,
    S <: StochasticAD.AbstractPerturbationStrategy
} <:
       StochasticAD.AbstractFIsBackend
    backend::B
    strategy::S
end

struct StrategyWrapperFIs{
    V,
    FIs <: StochasticAD.AbstractFIs{V},
    S <: StochasticAD.AbstractPerturbationStrategy
} <:
       AbstractWrapperFIs{V, FIs}
    Δs::FIs
    strategy::S
end

function StochasticAD.create_Δs(backend::StrategyWrapperFIsBackend, V)
    return StrategyWrapperFIs(StochasticAD.create_Δs(backend.backend, V), backend.strategy)
end

function StochasticAD.similar_type(::Type{<:StrategyWrapperFIs{V0, FIs0, S}},
        V,
        FIs) where {V0, FIs0, S}
    return StrategyWrapperFIs{V, FIs, S}
end

function AbstractWrapperFIsModule.reconstruct_wrapper(wrapper_Δs::StrategyWrapperFIs, Δs)
    return StrategyWrapperFIs(Δs, wrapper_Δs.strategy)
end

function AbstractWrapperFIsModule.reconstruct_wrapper(
        ::Type{
            <:StrategyWrapperFIs{V, FIs, S},
        },
        Δs) where {V, FIs, S}
    return StrategyWrapperFIs(Δs, S())
end

StochasticAD.new_Δs_strategy(Δs::StrategyWrapperFIs) = Δs.strategy

end
