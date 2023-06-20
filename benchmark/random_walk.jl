module RandomWalkBenchmark

using BenchmarkTools

using StochasticAD
using Statistics
using ForwardDiff: derivative
include("../tutorials/random_walk/core.jl")
using .RandomWalkCore: n, p, nsamples
using .RandomWalkCore: fX, get_dfX

const suite = BenchmarkGroup()

suite["original"] = @benchmarkable $(fX)($p)
suite["PrunedFIs"] = @benchmarkable derivative_estimate($fX, $p;
    backend = PrunedFIsBackend())
suite["PrunedFIsAggressive"] = @benchmarkable derivative_estimate($fX, $p;
    backend = PrunedFIsAggressiveBackend())
suite["SmoothedFIs"] = @benchmarkable derivative_estimate($fX, $p;
    backend = SmoothedFIsBackend())
forwarddiff_func = p -> fX(p; hardcode_leftright_step = true)
suite["ForwardDiff_smoothing"] = @benchmarkable derivative($forwarddiff_func, $p)

end
