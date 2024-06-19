abstract type AbstractStochasticADAlgorithm end

struct ForwardAlgorithm{B <: StochasticAD.AbstractFIsBackend} <: AbstractStochasticADAlgorithm
    backend::B
end

function StochasticAD.derivative_estimate(X, p, alg::ForwardAlgorithm; direction = nothing)
    return derivative_estimate(X, p; backend = alg.backend, direction)
end
