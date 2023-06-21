module GoLBenchmark

using BenchmarkTools

using StochasticAD
using Statistics
using ForwardDiff: derivative
include("../tutorials/game_of_life/core.jl")
using .GoLCore: play, p

const suite = BenchmarkGroup()

suite["original"] = @benchmarkable $play($p)
suite["PrunedFIs"] = @benchmarkable derivative_estimate($play, $p;
    backend = PrunedFIsBackend())
suite["PrunedFIsAggressive"] = @benchmarkable derivative_estimate($play, $p;
    backend = PrunedFIsAggressiveBackend())
suite["SmoothedFIs"] = @benchmarkable derivative_estimate($play, $p;
    backend = SmoothedFIsBackend())

end
